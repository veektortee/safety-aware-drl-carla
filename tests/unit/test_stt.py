"""Offline tests for commons/stt.py (no CARLA server, no ResNet: a small fake backbone stands in).

Covers: the per-frame feature cache and repeat-padding of SttNavObservationWrapper, the streaming PCA
projection and its fit from expert driving, the trainable SttFeaturesExtractor (gradients reach the
transformer), SAC training with it (the transformer weights change), and saving/loading (projection
and transformer weights survive a .zip round trip). Also checks the two exact speed-ups: the
single-token attention path and the detached critic features in the actor loss.

Run:  python tests/unit/test_stt.py      or      python -m pytest tests/unit/test_stt.py
"""

import os
import sys

sys.dont_write_bytecode = True  # tracked __pycache__/*.pyc files must not change
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
for path in (REPO_ROOT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

import tempfile

import numpy as np
import torch

from commons.nav_layout import NAV_DIM
from commons.stt import (GRID, NUM_FRAMES, TRUNK_CHANNELS, ProjectionFitter, SttFeaturesExtractor,
                         SttNavObservationWrapper, fit_stt_projection, stt_policy_kwargs)
from test_exp_glue import EXPERT, FakeCarlaEnv

K = 8          # small projection for the tests
CELLS = GRID[0] * GRID[1]


class FakeBackbone:
    """Frame -> (28, 2048) tokens with rank-3 structure driven by the frame brightness."""

    device = torch.device("cpu")

    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.basis = torch.randn(3, TRUNK_CHANNELS, generator=g)
        self.calls = 0

    def __call__(self, bgr):
        self.calls += 1
        v = float(np.asarray(bgr).mean()) / 255.0
        cells = torch.linspace(0.0, 1.0, CELLS)[:, None]
        coeffs = torch.cat([v * cells, v * v * (1.0 - cells), 0.2 + cells], dim=1)   # (28, 3)
        return coeffs @ self.basis


def _wrapper(env=None, backbone=None):
    return SttNavObservationWrapper(env or FakeCarlaEnv(), num_frames=NUM_FRAMES, channels=K,
                                    occlusion="none", backbone=backbone or FakeBackbone())


def _fitted_wrapper(env=None):
    w = _wrapper(env)
    fitter = ProjectionFitter()
    for t in range(12):
        fitter.add(w.backbone(np.full((360, 640, 3), t * 20, np.uint8)))
    proj = fitter.projection(K)
    w.set_projection(proj)
    w.backbone.calls = 0
    return w, proj


# --------------------------------------------------------------------------------------------------

def test_projection_fitter_recovers_low_rank_structure():
    rng = np.random.default_rng(0)
    basis = rng.standard_normal((4, TRUNK_CHANNELS))
    fitter = ProjectionFitter()
    data = []
    for _ in range(40):
        x = rng.standard_normal((CELLS, 4)) * [5.0, 3.0, 2.0, 1.0] @ basis
        x += 0.01 * rng.standard_normal(x.shape) + 3.0                  # small noise, non-zero mean
        fitter.add(torch.as_tensor(x))
        data.append(x)
    proj = fitter.projection(4)
    assert proj["components"].shape == (4, TRUNK_CHANNELS)
    assert proj["explained_variance"] > 0.999
    comps = proj["components"].astype(np.float64)
    assert np.allclose(comps @ comps.T, np.eye(4), atol=1e-4), "components are orthonormal"
    codes = (np.concatenate(data) - proj["mean"]) @ comps.T / proj["scale"]
    assert np.allclose(codes.std(axis=0), 1.0, atol=0.05), "codes are whitened"
    assert proj["n_tokens"] == 40 * CELLS


def test_wrapper_caches_one_encode_per_frame_and_repeat_pads():
    w = _wrapper()
    try:
        w.reset()
        raise AssertionError("an observation without a projection must fail loudly")
    except RuntimeError:
        pass

    w, _ = _fitted_wrapper()
    space = w.observation_space
    assert space["frames"].shape == (NUM_FRAMES, K, *GRID) and space["frames"].dtype == np.float16
    obs, _ = w.reset()
    assert space.contains(obs)
    assert w.backbone.calls == 1, "one ResNet pass per frame"
    first = obs["frames"]
    assert all(np.array_equal(first[i], first[0]) for i in range(NUM_FRAMES)), "repeat-padded, not black"

    for t in range(1, 4):
        obs, *_ = w.step(EXPERT)
        assert w.backbone.calls == 1 + t, "earlier frames come from the cache"
    frames = obs["frames"]
    assert all(np.array_equal(frames[i], first[0]) for i in range(NUM_FRAMES - 3)), "history kept in order"
    assert not np.array_equal(frames[-1], frames[-2]) and not np.array_equal(frames[-2], frames[-3])
    assert obs["nav"][0] == 3.0 and obs["nav"].dtype == np.float32

    obs, _ = w.reset()                                  # new episode: no frames of the previous one
    assert all(np.array_equal(obs["frames"][i], obs["frames"][0]) for i in range(NUM_FRAMES))
    assert np.array_equal(obs["frames"][0], first[0])   # FakeCarlaEnv's first frame is always the same


def test_fit_stt_projection_drives_the_expert_and_restores_the_stage():
    fake = FakeCarlaEnv(episode_len=7)
    w = _wrapper(fake)
    proj = fit_stt_projection(w, n_steps=20, reset_every=5, stage=3, verbose=0)
    assert fake.stage_calls == [3, 0] and fake.curriculum_stage == 0
    assert len(fake.actions) == 20 and all(np.allclose(a, EXPERT) for a in fake.actions)
    assert proj["n_tokens"] == 20 * CELLS and proj["explained_variance"] > 0.99
    assert w.projection is proj
    obs, _ = w.reset()
    assert w.observation_space.contains(obs)


def test_extractor_output_and_gradients_reach_the_transformer():
    w, _ = _fitted_wrapper()
    ext = SttFeaturesExtractor(w.observation_space, embed_dim=32)
    assert ext.features_dim == 32 + NAV_DIM
    obs = {"frames": torch.randn(4, NUM_FRAMES, K, *GRID), "nav": torch.randn(4, NAV_DIM)}
    out = ext(obs)
    assert out.shape == (4, 32 + NAV_DIM)
    assert torch.equal(out[:, 32:], obs["nav"]), "nav passes through unchanged"
    out[:, :32].pow(2).sum().backward()
    for name in ("encoder.proj.weight", "encoder.blocks.0.attn_time.in_proj_weight",
                 "encoder.blocks.0.attn_space.in_proj_weight", "cnn_proj.weight",
                 "transformer.blocks.4.mlp.0.weight", "transformer.proj_in.weight"):
        grad = dict(ext.named_parameters())[name].grad
        assert grad is not None and grad.abs().sum() > 0, f"no gradient for {name}"


def test_sac_trains_the_transformer_and_saves_everything():
    from stable_baselines3 import SAC

    w, proj = _fitted_wrapper(FakeCarlaEnv(episode_len=9))
    model = SAC("MultiInputPolicy", w, buffer_size=100, learning_starts=10, batch_size=8, device="cpu",
                verbose=0, seed=0, policy_kwargs={**stt_policy_kwargs(2, embed_dim=32),
                                                   "net_arch": {"pi": [32, 32], "qf": [32, 32]}})
    model.stt_projection = proj
    critic_before = {k: v.clone() for k, v in model.critic.features_extractor.state_dict().items()}
    actor_before = {k: v.clone() for k, v in model.actor.features_extractor.state_dict().items()}
    model.learn(total_timesteps=30, log_interval=None)
    assert model.replay_buffer.observations["frames"].dtype == np.float16, "compact float16 replay buffer"
    for name in ("encoder.proj.weight", "cnn_proj.weight", "transformer.proj_mid.weight"):
        assert not torch.equal(model.critic.features_extractor.state_dict()[name], critic_before[name]), name
        assert not torch.equal(model.actor.features_extractor.state_dict()[name], actor_before[name]), name

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = os.path.join(tmp, "sac_stt_30_steps.zip")
        model.save(path)
        w2 = _wrapper(FakeCarlaEnv(episode_len=9))
        loaded = SAC.load(path, env=w2, device="cpu")
    for key in ("mean", "components", "scale"):
        assert np.array_equal(loaded.stt_projection[key], proj[key]), f"projection '{key}' not restored"
    assert loaded.stt_projection["explained_variance"] == proj["explained_variance"]
    for part in ("actor", "critic"):
        saved = getattr(model, part).features_extractor.state_dict()
        restored = getattr(loaded, part).features_extractor.state_dict()
        assert saved.keys() == restored.keys()
        assert all(torch.equal(saved[k], restored[k]) for k in saved), f"{part} STT weights not restored"
    w2.set_projection(loaded.stt_projection)
    obs, _ = w2.reset()
    action, _ = loaded.predict(obs, deterministic=True)
    expected, _ = model.predict(obs, deterministic=True)
    assert np.allclose(action, expected, atol=1e-6), "restored model acts like the saved one"


def test_single_token_attention_fast_path_is_exact():
    from commons.spatiotemporal_transformer import TransformerBlock

    torch.manual_seed(0)
    blk = TransformerBlock(32, 4).double()
    x = torch.randn(5, 1, 32, dtype=torch.double)
    h = blk.norm1(x)
    ref = x + blk.attn(h, h, h)[0]
    ref = ref + blk.mlp(blk.norm2(ref))
    assert torch.allclose(blk(x), ref, atol=1e-12)


def test_actor_loss_critic_features_detached():
    from stable_baselines3 import SAC

    w, proj = _fitted_wrapper()
    model = SAC("MultiInputPolicy", w, buffer_size=50, device="cpu", verbose=0, seed=0,
                policy_kwargs={**stt_policy_kwargs(3, embed_dim=16), "net_arch": [16]})
    critic = model.critic
    obs = {"frames": torch.randn(4, NUM_FRAMES, K, *GRID), "nav": torch.randn(4, NAV_DIM)}
    act_a = torch.rand(4, 2, requires_grad=True)
    act_b = act_a.detach().clone().requires_grad_(True)
    q_full, _, _ = critic.ensemble_forward(obs, act_a)
    q_det, _, _ = critic.ensemble_forward(obs, act_b, detach_features=True)
    assert torch.allclose(q_full, q_det)
    critic.zero_grad(set_to_none=True)
    q_det.min(dim=1).values.sum().backward()
    q_full.min(dim=1).values.sum().backward()
    assert torch.allclose(act_a.grad, act_b.grad), "same dQ/d(action) for the actor loss"
    # only the full pass reached the extractor; a detached-only backward would leave it untouched
    critic.zero_grad(set_to_none=True)
    q_det2, _, _ = critic.ensemble_forward(obs, act_b.detach().requires_grad_(True), detach_features=True)
    q_det2.sum().backward()
    assert all(p.grad is None for p in critic.features_extractor.parameters())


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
