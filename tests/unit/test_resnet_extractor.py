"""Offline tests for commons/resnet_extractor.py (no CARLA server).

Most tests use a tiny randomly initialised torchvision ResNet (BasicBlock, one block per stage)
registered as backbone "tiny", so they run in seconds on the CPU. One test loads the real ImageNet
ResNet-50 (skipped if pretrained/feature_extractor/pretrained_resnet50.pth is missing).

Covers: the feature size and layout match CombinedExtractor/NatureCNN (image features, then nav),
BGR -> RGB + ImageNet normalisation + folded BatchNorm give the torchvision ResNet's own output,
which parameters are frozen / trained for each --resnet-train mode, BatchNorm stays in eval mode
under policy.set_training_mode(True), gradients reach the trained part only, the shared frozen trunk
reuses its output for repeated batches (2 trunk passes per SAC gradient step instead of 5), SAC
trains with the extractor and a checkpoint round-trips, checkpoint inspection / mismatch messages,
the experiment's CLI names and the train_all entries.

Run:  python tests/unit/test_resnet_extractor.py      or      python -m pytest tests/unit/test_resnet_extractor.py
"""

import os
import sys

sys.dont_write_bytecode = True  # tracked __pycache__/*.pyc files must not change
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
for path in (REPO_ROOT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

import copy
import importlib.util
import tempfile

import numpy as np
import torch
import torch.nn as nn

from commons.nav_layout import NAV_DIM
from commons.obs_wrappers import CnnNavObservationWrapper
from commons.resnet_extractor import (CNN_OUTPUT_DIM, IMAGENET_MEAN, IMAGENET_STD, STAGES, TRAINABLE_CHOICES,
                                      ResNetNavExtractor, checkpoint_image_extractor, extractor_mismatch,
                                      register_backbone, resnet_extractor_kwargs)
from test_exp_glue import FakeCarlaEnv

TINY = "tiny"


def _tiny_resnet():
    """ResNet(BasicBlock, [1, 1, 1, 1]) with random weights and non-trivial BatchNorm statistics."""
    from torchvision.models.resnet import BasicBlock, ResNet

    with torch.random.fork_rng():
        torch.manual_seed(0)
        net = ResNet(BasicBlock, [1, 1, 1, 1])
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.uniform_(-0.2, 0.2)
                m.running_var.uniform_(0.5, 2.0)
                m.weight.data.uniform_(0.5, 1.5)
                m.bias.data.uniform_(-0.2, 0.2)
    return net


register_backbone(TINY, _tiny_resnet)


def _env():
    return CnnNavObservationWrapper(FakeCarlaEnv(), occlusion="none")


def _obs(batch=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"image": torch.rand(batch, 3, 84, 84, generator=g),          # SB3 hands over image / 255
            "nav": torch.randn(batch, NAV_DIM, generator=g)}


def _reference_pooled(net, image):
    """torchvision forward (stem .. avgpool) on the RGB, ImageNet-normalised image."""
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    x = (image[:, [2, 1, 0]] - mean) / std
    with torch.no_grad():
        for name in STAGES:
            x = getattr(net, name)(x)
    return torch.flatten(x, 1)


def _sac(trainable="layer4", backbone=TINY, batch_size=8, env=None):
    from stable_baselines3 import SAC

    policy_kwargs = {"net_arch": {"pi": [32, 32], "qf": [32, 32]}, "n_critics": 5, "normalize_images": True}
    policy_kwargs.update(resnet_extractor_kwargs(trainable))
    policy_kwargs["features_extractor_kwargs"]["backbone"] = backbone
    return SAC("MultiInputPolicy", env or _env(), buffer_size=200, learning_starts=0, batch_size=batch_size,
               policy_kwargs=policy_kwargs, device="cpu", verbose=0, seed=0)


def _bn_layers(module):
    return [m for m in module.modules() if isinstance(m, nn.BatchNorm2d)]


# --------------------------------------------------------------------------------------------------

def test_feature_dim_and_layout_match_nature_cnn():
    from stable_baselines3.common.torch_layers import CombinedExtractor

    space = _env().observation_space
    nature = CombinedExtractor(space)
    assert nature.features_dim == CNN_OUTPUT_DIM + NAV_DIM
    obs = _obs()
    for mode in TRAINABLE_CHOICES:
        ext = ResNetNavExtractor(space, trainable=mode, backbone=TINY)
        assert ext.features_dim == nature.features_dim, mode
        out = ext(obs)
        assert out.shape == nature(obs).shape == (4, CNN_OUTPUT_DIM + NAV_DIM)
        assert torch.equal(out[:, CNN_OUTPUT_DIM:], obs["nav"]), "nav is concatenated after the image features"
        assert (out[:, :CNN_OUTPUT_DIM] >= 0).all(), "fc + ReLU like NatureCNN's last layer"


def test_output_equals_torchvision_resnet_on_rgb_imagenet_input():
    from commons.resnet_extractor import _pretrained

    space = _env().observation_space
    obs = _obs(batch=3, seed=1)
    ref = _reference_pooled(_pretrained(TINY), obs["image"])
    for mode in TRAINABLE_CHOICES:
        ext = ResNetNavExtractor(space, trainable=mode, backbone=TINY).eval()
        expected = torch.relu(ext.fc[0](ref))
        got = ext(obs)[:, :CNN_OUTPUT_DIM]
        assert torch.allclose(got, expected, atol=1e-4, rtol=1e-4), (mode, (got - expected).abs().max())
        swapped = dict(obs, image=obs["image"][:, [2, 1, 0]])               # an RGB image would differ
        assert not torch.allclose(ext(swapped)[:, :CNN_OUTPUT_DIM], expected, atol=1e-3)


def test_frozen_and_trained_parts_per_mode():
    expected_body = {"head": [], "layer4": ["layer4", "avgpool"], "all": list(STAGES)}
    for mode in TRAINABLE_CHOICES:
        model = _sac(mode)
        policy = model.policy
        actor_ext = model.actor.features_extractor
        critic_ext, target_ext = model.critic.features_extractor, model.critic_target.features_extractor
        assert isinstance(actor_ext, ResNetNavExtractor) and actor_ext.trainable == mode
        assert actor_ext is not critic_ext and critic_ext is not target_ext, "separate extractors (SB3 SAC)"
        assert [n for n, _ in actor_ext.body.named_children()] == expected_body[mode]
        assert all(p.requires_grad for p in actor_ext.parameters()), "everything in the module trains"

        frozen = actor_ext.frozen
        if mode == "all":
            assert frozen is None
        else:
            assert frozen is critic_ext.frozen is target_ext.frozen, "one frozen trunk shared by all three"
            frozen_ids = {id(p) for p in frozen.net.parameters()}
            assert frozen_ids and all(not p.requires_grad for p in frozen.net.parameters())
            assert not frozen_ids & {id(p) for p in policy.parameters()}, "frozen weights are not policy params"
            assert not _bn_layers(frozen.net), "BatchNorm folded into the frozen convolutions"
        keys = list(policy.state_dict())
        assert not any(k.startswith("actor.features_extractor.mean") for k in keys), "constants not saved"
        assert any(".features_extractor.fc.0.weight" in k for k in keys)
        assert any("layer4" in k for k in keys) == (mode != "head")
        assert any(".layer3." in k for k in keys) == (mode == "all")

        policy.set_training_mode(True)
        assert actor_ext.training and actor_ext.fc.training and critic_ext.fc.training
        assert all(not bn.training for ext in (actor_ext, critic_ext, target_ext) for bn in _bn_layers(ext)), \
            "BatchNorm keeps the ImageNet running statistics while training"
        assert mode == "head" or _bn_layers(actor_ext.body), "trained ResNet layers keep their BatchNorm"
        if frozen is not None:
            assert not frozen.training
        policy.set_training_mode(False)
        assert not actor_ext.fc.training


def test_gradients_reach_only_the_trained_part():
    space = _env().observation_space
    for mode in TRAINABLE_CHOICES:
        ext = ResNetNavExtractor(space, trainable=mode, backbone=TINY)
        ext(_obs())[:, :CNN_OUTPUT_DIM].pow(2).sum().backward()
        assert ext.fc[0].weight.grad is not None and ext.fc[0].weight.grad.abs().sum() > 0
        for name, p in ext.body.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), (mode, name)
        if ext.frozen is not None:
            assert all(p.grad is None for p in ext.frozen.net.parameters())


def test_frozen_trunk_reuses_repeated_batches():
    model = _sac("head", batch_size=8)
    model.learn(total_timesteps=12, log_interval=None)
    frozen = model.actor.features_extractor.frozen
    trunk_net = frozen.net
    obs = _obs(batch=2, seed=3)
    x = torch.rand(2, 3, 84, 84)
    out1 = frozen(x)
    hits, misses = frozen.hits, frozen.misses
    out2 = frozen(x.clone())                                   # same content, another tensor
    assert frozen.hits == hits + 1 and frozen.misses == misses and out2 is out1
    frozen(x + 1e-3)
    assert frozen.misses == misses + 1

    # one SAC gradient step: actor(obs), actor(next_obs), target(next_obs), critic(obs) x2
    hits, misses = frozen.hits, frozen.misses
    model.train(gradient_steps=1, batch_size=8)
    assert (frozen.hits - hits, frozen.misses - misses) == (3, 2), (frozen.hits - hits, frozen.misses - misses)
    assert frozen.net is trunk_net
    assert model.actor.features_extractor(obs).shape == (2, CNN_OUTPUT_DIM + NAV_DIM)


def test_sac_trains_and_checkpoint_round_trips():
    from stable_baselines3 import SAC

    model = _sac("layer4")
    actor_ext, critic_ext = model.actor.features_extractor, model.critic.features_extractor
    frozen_before = copy.deepcopy(actor_ext.frozen.net.state_dict())
    before = {name: {k: v.clone() for k, v in ext.state_dict().items()}
              for name, ext in (("actor", actor_ext), ("critic", critic_ext))}
    model.learn(total_timesteps=40, log_interval=None)
    for name, ext in (("actor", actor_ext), ("critic", critic_ext)):
        after = ext.state_dict()
        for key in ("fc.0.weight", "body.layer4.0.conv1.weight", "body.layer4.0.bn1.weight"):
            assert not torch.equal(before[name][key], after[key]), f"{name} {key} did not train"
        for key in ("body.layer4.0.bn1.running_mean", "body.layer4.0.bn1.running_var"):
            assert torch.equal(before[name][key], after[key]), f"{name} {key} changed (BatchNorm must stay frozen)"
    for k, v in actor_ext.frozen.net.state_dict().items():
        assert torch.equal(frozen_before[k], v), f"frozen trunk changed: {k}"

    obs = _env().observation_space.sample()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = os.path.join(tmp, "sac_5q_cnn_resnet_layer4_40_steps.zip")
        model.save(path)
        loaded = SAC.load(path, env=_env(), device="cpu")
        ext = loaded.actor.features_extractor
        assert isinstance(ext, ResNetNavExtractor) and ext.trainable == "layer4" and ext.backbone == TINY
        sd0, sd1 = model.policy.state_dict(), loaded.policy.state_dict()
        assert sd0.keys() == sd1.keys() and all(torch.equal(sd0[k], sd1[k]) for k in sd0)
        a0, _ = model.predict(obs, deterministic=True)
        a1, _ = loaded.predict(obs, deterministic=True)
        assert np.allclose(a0, a1, atol=1e-6)
        loaded.learn(total_timesteps=3, log_interval=None, reset_num_timesteps=False)

        assert checkpoint_image_extractor(path) == {"cnn": "resnet50", "resnet_train": "layer4"}
        assert extractor_mismatch(path, "resnet50", "layer4") is None
        msg = extractor_mismatch(path, "nature", "layer4")
        assert msg and "--cnn resnet50 --resnet-train layer4" in msg and "--cnn nature" in msg
        assert "--resnet-train head" in extractor_mismatch(path, "resnet50", "head")

        nature = SAC("MultiInputPolicy", _env(), buffer_size=50, learning_starts=0, device="cpu", verbose=0,
                     policy_kwargs={"net_arch": [16], "n_critics": 2, "normalize_images": True})
        nature_path = os.path.join(tmp, "sac_5q_cnn_1_steps.zip")
        nature.save(nature_path)
        assert checkpoint_image_extractor(nature_path) == {"cnn": "nature", "resnet_train": None}
        assert extractor_mismatch(nature_path, "nature", "layer4") is None     # --resnet-train ignored
        msg = extractor_mismatch(nature_path, "resnet50", "layer4")
        assert msg and "trained with --cnn nature" in msg
        assert "cannot read" in extractor_mismatch(os.path.join(tmp, "missing.zip"), "nature", None)


def test_real_resnet50_layout_and_output():
    from commons.stt import RESNET50_WEIGHTS

    if not os.path.isfile(RESNET50_WEIGHTS):
        print(f"  (skipped: {RESNET50_WEIGHTS} not found)")
        return
    from commons.resnet_extractor import _pretrained

    space = _env().observation_space
    ext = ResNetNavExtractor(space, trainable="layer4").eval()
    assert ext.features_dim == CNN_OUTPUT_DIM + NAV_DIM
    assert ext.fc[0].in_features == 2048 and ext.fc[0].out_features == CNN_OUTPUT_DIM
    n_trained = sum(p.numel() for p in ext.parameters())
    assert 15.0e6 < n_trained < 16.0e6, n_trained                      # layer4 (14.96M) + fc (0.52M)
    obs = _obs(batch=2, seed=4)
    ref = _reference_pooled(_pretrained("resnet50"), obs["image"])
    expected = torch.relu(ext.fc[0](ref))
    got = ext(obs)[:, :CNN_OUTPUT_DIM]
    assert torch.allclose(got, expected, atol=1e-3, rtol=1e-3), (got - expected).abs().max()


def _load_experiment():
    spec = importlib.util.spec_from_file_location(
        "exp_5q_cnn", os.path.join(REPO_ROOT, "experiments", "5qCNNsac_experiment.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_experiment_cli_names_and_train_all_entries():
    exp = _load_experiment()
    args = exp.parse_args([])
    assert (args.cnn, args.log_dir, exp.run_names(args)) == ("nature", "./logs/5q_cnn", ("5q_cnn", "sac_5q_cnn"))
    for mode in TRAINABLE_CHOICES:
        args = exp.parse_args(["--cnn", "resnet50", "--resnet-train", mode])
        tag = f"5q_cnn_resnet_{mode}"
        assert args.log_dir == f"./logs/{tag}" and exp.run_names(args) == (tag, f"sac_{tag}")
    assert exp.parse_args(["--cnn", "resnet50", "--log-dir", "x"]).log_dir == "x"

    import train_all

    default_run = [n for n, cfg in train_all.EXPERIMENTS.items() if not cfg.get("opt_in")]
    assert default_run == ["2q_cnn", "5q_cnn", "2q_stt", "5q_stt"]
    args = train_all.build_parser().parse_args(["--only", "5q_cnn_resnet_layer4"])
    for mode in TRAINABLE_CHOICES:
        cfg = train_all.EXPERIMENTS[f"5q_cnn_resnet_{mode}"]
        cmd = train_all.build_command(cfg["script"], 2000, cfg["log_dir"], None, args, cfg["args"])
        parsed = exp.parse_args(cmd[2:])
        assert parsed.cnn == "resnet50" and parsed.resnet_train == mode
        assert exp.run_names(parsed)[1] == cfg["prefix"], "train_all finds the script's checkpoints"


if __name__ == "__main__":
    import traceback

    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
