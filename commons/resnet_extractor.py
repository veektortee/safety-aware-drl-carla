"""ImageNet ResNet-50 image branch for SAC 'MultiInputPolicy' (5q CNN experiment, --cnn resnet50).

Drop-in replacement for SB3's CombinedExtractor (NatureCNN on the image + flattened nav vector) on
the CnnNavObservationWrapper observation Dict(image (3,84,84) uint8, nav (NAV_DIM,) float32). The
complete ResNet-50 is used, from the stem to its fully connected tail; only torchvision's ImageNet
classifier (fc 2048 -> 1000 classes) is replaced by a new fc 2048 -> 256 + ReLU, so the image
features have NatureCNN's size and the SAC decision layers (pi / qf MLPs, the critic ensemble) are
unchanged:

    image (BGR, /255 by SB3) -> RGB, ImageNet mean/std
      -> conv1 bn1 relu maxpool layer1 layer2 layer3 layer4 avgpool   (ImageNet weights)
      -> fc Linear(2048, 256) + ReLU                                  (new, trained by SAC)
    concat nav                                                        -> 256 + NAV_DIM features

`trainable` chooses how much of the ResNet the SAC losses train (the fc is always trained):

    "head"    fc only. The whole conv trunk + avgpool is frozen.
    "layer4"  layer4 + fc. conv1..layer3 are frozen. (default)
    "all"     every layer.

Measured on the RTX 5060 Laptop (batch 64, 5 critics, one SAC gradient step): NatureCNN 22 ms,
head 32 ms, layer4 49 ms, all 107 ms. layer4 is the default: it adapts the most task-specific block
to driving at about half the cost of training everything (lower blocks hold generic edge/texture
filters that RL gradients from a 64-sample batch rarely improve).

Frozen layers run without gradients, BatchNorm folded into the convolutions, fp16 channels-last on
CUDA. They are not part of the SB3 policy: one copy per device is shared by the actor, critic and
critic-target extractors, loaded from pretrained/feature_extractor/pretrained_resnet50.pth
(commons.stt.RESNET50_WEIGHTS), and excluded from the state dict, so checkpoints, the optimizers and
the target-network update only see the trained part. Because a SAC gradient step feeds the same
batch to several extractors, the shared frozen trunk remembers its last two inputs and reuses the
output when a batch repeats (an exact torch.equal test; frozen layers always give the same output).

Trained ResNet layers start from the ImageNet weights, are copied into each extractor (actor,
critic, critic target), are saved in the checkpoint .zip and run in bf16 autocast on CUDA (fp32
where bf16 is unsupported). Every BatchNorm keeps the ImageNet running statistics (eval mode, also
during training); in trained layers only its affine weight and bias learn.
"""

import copy
import functools
import json
import zipfile
from collections import OrderedDict
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

TRAINABLE_CHOICES = ("head", "layer4", "all")
DEFAULT_TRAINABLE = "layer4"
CNN_OUTPUT_DIM = 256        # NatureCNN's / CombinedExtractor's image feature size

# torchvision ResNet children before the classifier, in forward order
STAGES = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool")
_FIRST_TRAINED_STAGE = {"head": len(STAGES), "layer4": STAGES.index("layer4"), "all": 0}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ------------------------------------------------------------------------------------------------
# Backbones (torchvision-style ResNets with ImageNet weights)
# ------------------------------------------------------------------------------------------------

def _imagenet_resnet50() -> nn.Module:
    from torchvision import models

    from commons.stt import RESNET50_WEIGHTS, _resnet50_state_dict

    net = models.resnet50(weights=None)
    net.load_state_dict(_resnet50_state_dict(RESNET50_WEIGHTS), strict=True)
    return net


# name -> builder() returning a torchvision ResNet (children conv1 .. avgpool, fc). Kept as a name so
# the pickled policy_kwargs stay small; tests register a tiny stand-in with register_backbone().
BACKBONES: Dict[str, Callable[[], nn.Module]] = {"resnet50": _imagenet_resnet50}


def register_backbone(name: str, builder: Callable[[], nn.Module]):
    BACKBONES[name] = builder
    _pretrained.cache_clear()
    _FROZEN.clear()


@functools.lru_cache(maxsize=None)
def _pretrained(backbone: str) -> nn.Module:
    """CPU master copy of the pretrained backbone, loaded once per process."""
    if backbone not in BACKBONES:
        raise ValueError(f"unknown backbone {backbone!r}; known: {sorted(BACKBONES)}")
    net = BACKBONES[backbone]()
    for name in STAGES + ("fc",):
        if not hasattr(net, name):
            raise ValueError(f"backbone {backbone!r} has no '{name}' layer (torchvision ResNet layout expected)")
    return net.eval()


def _stages(net: nn.Module, start: int, stop: int) -> nn.Sequential:
    return nn.Sequential(OrderedDict((n, copy.deepcopy(getattr(net, n))) for n in STAGES[start:stop]))


def _amp_dtype(device: torch.device) -> Optional[torch.dtype]:
    """Autocast dtype of the trained ResNet layers: bf16 on CUDA (no grad scaler needed), else fp32."""
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


def fuse_conv_bn(module: nn.Module) -> nn.Module:
    """Fold every eval-mode BatchNorm2d into the Conv2d registered just before it (torchvision ResNet
    order: conv1/bn1, conv2/bn2, conv3/bn3, downsample[0]/[1]). Same outputs, half the kernels."""
    from torch.nn.utils.fusion import fuse_conv_bn_eval

    for parent in list(module.modules()):
        children = list(parent.named_children())
        for (conv_name, conv), (bn_name, bn) in zip(children, children[1:]):
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                setattr(parent, conv_name, fuse_conv_bn_eval(conv, bn))
                setattr(parent, bn_name, nn.Identity())
    return module


class FrozenTrunk:
    """The frozen first stages of a backbone on one device; no gradients, always eval mode.

    BatchNorm is folded into the convolutions (ImageNet running statistics) and, on CUDA, the
    weights are fp16 and channels-last. Shared by every extractor with the same backbone and split.
    Remembers its last `slots` inputs and returns the stored output when a batch repeats (a SAC step
    passes the same batch to the actor and critic extractors several times).
    """

    def __init__(self, net: nn.Sequential, device: torch.device, slots: int = 2):
        net = fuse_conv_bn(net.eval())
        self.dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.net = net.to(device=device, dtype=self.dtype).to(memory_format=torch.channels_last).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device = device
        self.slots = int(slots)
        self.cache = []                 # [(input, output)], most recent first
        self.hits = 0
        self.misses = 0

    @property
    def training(self) -> bool:
        return self.net.training

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for i, (key, out) in enumerate(self.cache):
            if key.shape == x.shape and key.dtype == x.dtype and torch.equal(key, x):
                self.hits += 1
                if i:
                    self.cache.insert(0, self.cache.pop(i))
                return out
        self.misses += 1
        out = self.net(x.to(self.dtype))
        if self.slots > 0:
            self.cache.insert(0, (x, out))
            del self.cache[self.slots:]
        return out


_FROZEN: Dict[tuple, FrozenTrunk] = {}


def frozen_trunk(backbone: str, stop: int, device) -> FrozenTrunk:
    """The shared FrozenTrunk with stages[:stop] of `backbone` on `device` (built on first use)."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    key = (backbone, int(stop), str(device))
    if key not in _FROZEN:
        _FROZEN[key] = FrozenTrunk(_stages(_pretrained(backbone), 0, stop), device)
    return _FROZEN[key]


# ------------------------------------------------------------------------------------------------
# SB3 features extractor
# ------------------------------------------------------------------------------------------------

class ResNetNavExtractor(BaseFeaturesExtractor):
    """Dict(image (3,H,W) uint8 BGR, nav) -> [ResNet features (cnn_output_dim), nav].

    Same output layout as CombinedExtractor with NatureCNN (image features first, then the nav
    vector), so features_dim = cnn_output_dim + NAV_DIM. Expects SB3's normalize_images=True (the
    image arrives as float in [0, 1]).
    """

    def __init__(self, observation_space: spaces.Dict, cnn_output_dim: int = CNN_OUTPUT_DIM,
                 trainable: str = DEFAULT_TRAINABLE, backbone: str = "resnet50"):
        image_space = observation_space["image"]
        if len(image_space.shape) != 3 or image_space.shape[0] != 3:
            raise ValueError(f"ResNetNavExtractor needs a channel-first 3-channel image, got {image_space.shape}")
        if trainable not in TRAINABLE_CHOICES:
            raise ValueError(f"trainable must be one of {TRAINABLE_CHOICES}, got {trainable!r}")
        nav_dim = int(torch.Size(observation_space["nav"].shape).numel())
        super().__init__(observation_space, features_dim=int(cnn_output_dim) + nav_dim)
        self.backbone = backbone
        self.trainable = trainable
        self.split = _FIRST_TRAINED_STAGE[trainable]
        net = _pretrained(backbone)
        # trained ResNet stages (ImageNet init) followed by the new fc; empty body for "head"
        self.body = _stages(net, self.split, len(STAGES)).to(memory_format=torch.channels_last)
        self.fc = nn.Sequential(nn.Linear(net.fc.in_features, int(cnn_output_dim)), nn.ReLU())
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.train(True)

    def train(self, mode: bool = True):
        """BatchNorm layers always stay in eval mode (ImageNet running statistics)."""
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
        return self

    @property
    def frozen(self) -> Optional[FrozenTrunk]:
        """The shared frozen trunk on this extractor's device (None when everything is trained)."""
        if self.split == 0:
            return None
        return frozen_trunk(self.backbone, self.split, self.fc[0].weight.device)

    def forward(self, observations: dict) -> torch.Tensor:
        img = observations["image"]                                  # (B, 3, H, W) BGR in [0, 1]
        x = (img.flip(1) - self.mean) / self.std                     # RGB, ImageNet normalisation
        x = x.contiguous(memory_format=torch.channels_last)
        if self.split > 0:
            x = frozen_trunk(self.backbone, self.split, x.device)(x)
        if len(self.body) > 0:
            amp = _amp_dtype(x.device)
            if amp is None:
                x = x.float()                                        # frozen fp16 output -> fp32
            with torch.autocast(device_type=x.device.type, dtype=amp, enabled=amp is not None):
                x = self.body(x)
        image_features = self.fc(torch.flatten(x, 1).float())
        return torch.cat([image_features, observations["nav"].flatten(1)], dim=1)


def resnet_extractor_kwargs(trainable: str = DEFAULT_TRAINABLE) -> dict:
    """policy_kwargs entries that switch SAC 'MultiInputPolicy' to the ResNet-50 image branch."""
    return {"features_extractor_class": ResNetNavExtractor,
            "features_extractor_kwargs": {"cnn_output_dim": CNN_OUTPUT_DIM, "trainable": trainable}}


# ------------------------------------------------------------------------------------------------
# Checkpoint inspection (no model is built)
# ------------------------------------------------------------------------------------------------

def checkpoint_image_extractor(checkpoint_path: str) -> dict:
    """{'cnn': 'nature' | 'resnet50' | <other extractor>, 'resnet_train': mode or None} of an SB3 .zip.

    Reads the human-readable copy of policy_kwargs that SB3 writes next to the pickled one, so
    nothing is unpickled. Raises OSError / ValueError if the file is not an SB3 checkpoint.
    """
    with zipfile.ZipFile(checkpoint_path) as zf:
        data = json.loads(zf.read("data").decode("utf-8"))
    policy_kwargs = data.get("policy_kwargs") or {}
    cls = str(policy_kwargs.get("features_extractor_class") or "")
    if not cls or "CombinedExtractor" in cls:
        return {"cnn": "nature", "resnet_train": None}
    if "ResNetNavExtractor" in cls:
        kwargs = policy_kwargs.get("features_extractor_kwargs") or {}
        return {"cnn": "resnet50", "resnet_train": kwargs.get("trainable", DEFAULT_TRAINABLE)}
    return {"cnn": cls, "resnet_train": None}


def extractor_mismatch(checkpoint_path: str, cnn: str, resnet_train: Optional[str]) -> Optional[str]:
    """None if the checkpoint's image branch is the requested one, else a message saying why not."""
    try:
        found = checkpoint_image_extractor(checkpoint_path)
    except Exception as exc:  # noqa: BLE001
        return f"cannot read the image extractor of {checkpoint_path}: {exc}"
    wanted = {"cnn": cnn, "resnet_train": resnet_train if cnn == "resnet50" else None}
    if found == wanted:
        return None

    def describe(d):
        return "resnet50 --resnet-train " + str(d["resnet_train"]) if d["cnn"] == "resnet50" else d["cnn"]

    return (f"checkpoint {checkpoint_path} was trained with --cnn {describe(found)}, but this run asks for "
            f"--cnn {describe(wanted)}. The image branches have different weights, so it cannot be "
            f"resumed; pass the matching options or start a fresh run in another --log-dir.")
