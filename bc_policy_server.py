#!/usr/bin/env python3
"""
LBM Eval — BC policy gRPC server (Phase 3)
==========================================
Standalone gRPC policy server that serves the Phase 2 BC checkpoint
(`bc_v1.pt`) over the lbm_eval benchmark's policy interface.

WHY A STANDALONE FILE (no Modal)
--------------------------------
Phase 2 ran the policy in-process: `evaluate_one(policy=BCPolicy(), use_rpc=
False)`. That proved the inference logic but bypassed the benchmark's intended
transport. The benchmark's real contract (README, "Evaluating using your own
policy wrapper") is a two-process design: a policy server speaking gRPC, and
the `evaluate` client. Phase 1's fan-out (`run_eval_shard` -> `evaluate_many`)
is ALREADY built as that gRPC *client* — it launches a server subprocess and
connects to it. Phase 1 used the bundled `wave_around_policy_server`; Phase 3
points the exact same machinery at THIS server instead.

Keeping the server standalone (only the `robot_gym` wheel + torch are needed,
per the README) means it is, at once: (a) the artifact goal 4 releases,
(b) the file v2's diffusion policy reuses verbatim — only the model swaps,
(c) already shaped to later run on its own GPU container (change one
`--server-uri`, no rewrite).

WHAT THIS FILE SUPPLIES
-----------------------
The benchmark's gRPC plumbing — servicer, proto marshalling, batching, port
binding — all lives behind `grpc_workspace.lbm_policy_server.run_policy_server`.
This file only supplies, exactly mirroring `wave_around_policy_server.py`:
  - BCPolicy       : the inference policy, lifted verbatim from Phase 2's
                     lbm_phase2_serve.py (its `step` works UNCHANGED over gRPC
                     — see the round-trip note below).
  - BCPolicyBatch  : the batch-interface wrapper run_policy_server requires.
                     run_policy_server calls reset_batch / step_batch /
                     get_policy_metadata on the policy — never plain
                     reset/step — so a WaveAroundBatch-shaped wrapper is
                     mandatory, not optional.
  - main()         : load checkpoint -> build model -> serve forever.

VERIFIED, NOT ASSUMED — the gRPC observation round-trip
-------------------------------------------------------
`wave_around` only reads `observation.robot.actual.poses`; it never touches
`observation.visuo`, so Phase 1 never exercised image transport. Checked
directly in grpc_workspace/lbm_policy_conversions.py:
  - `camera_image_set_map_to_grpc_msg` writes each `visuo` dict key into the
    proto `camera_serial` field; `grpc_msg_to_camera_image_set_map` reads it
    back as the dict key. The round-trip preserves the key verbatim, so
    `observation.visuo["scene_right_0"]` resolves server-side exactly as it
    did in-process.
  - it rebuilds `CameraImageSet(rgb=CameraRgbImage(array=...))` from a
    DTYPE_UINT8 Image, so `observation.visuo[...].rgb.array` is the same
    uint8 (480,640,3) array Phase 2 fed the net.
Conclusion: BCPolicy._extract_image / _extract_proprio need NO changes for the
gRPC path. As a cheap guard, BCPolicy logs the live `visuo` keys + image shape
once on the first step.

Run (normally launched as a subprocess by lbm_phase3.py):
  python3 bc_policy_server.py --checkpoint /data/phase2_models/bc_v1.pt \
                              --server-uri localhost:50051
"""
import argparse
import uuid

import numpy as np
import torch

from grpc_workspace.lbm_policy_server import (
    LbmPolicyServerConfig,
    run_policy_server,
)
from robot_gym.multiarm_spaces import PosesAndGrippers
from robot_gym.policy import Policy, PolicyMetadata
from pydrake.math import RigidTransform, RotationMatrix


# --- Interface constants — confirmed by the Phase 2 probes ------------------
# (see lbm_phase2_serve.py: probe_observation / probe_visuo)
SCENE_CAMERA_LIVE = "scene_right_0"   # = training camera serial 6CD146030E99
ARM_KEYS = ["right::panda", "left::panda"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_CHECKPOINT = "/data/phase2_models/bc_v1.pt"


# ===========================================================================
# MODEL + ROTATION HELPERS
# ===========================================================================
# Copied VERBATIM from lbm_phase2_serve.py so this server stays standalone
# (the README's design: a policy venv needs only the robot_gym wheel). The
# clean de-duplication is a shared `bc_model.py` imported by the trainer, the
# Phase 2 serve file, and this server — worth doing before the goal-4 release,
# but left as a follow-up so Phase 2's working files are not disturbed now.
# _build_model MUST stay structurally identical to the trainer, or
# load_state_dict() will reject bc_v1.pt.

def _rot6d_from_matrix(R):
    """3x3 rotation matrix -> 6D representation (first two columns, flattened).
    This is the Zhou et al. continuity representation the training data uses."""
    R = np.asarray(R, dtype=np.float32)
    return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)


def _matrix_from_rot6d(v):
    """6D representation -> valid 3x3 rotation matrix via Gram-Schmidt.
    Inverse of _rot6d_from_matrix; guarantees an orthonormal result even if
    the network output is slightly off-manifold."""
    v = np.asarray(v, dtype=np.float64).reshape(6)
    a1, a2 = v[:3], v[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2 = a2 - np.dot(b1, a2) * b1
    b2 = a2 / (np.linalg.norm(a2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)   # columns b1,b2,b3


def _build_model(proprio_dim, action_dim):
    """Rebuild the exact architecture from lbm_phase2_train.py so the saved
    state_dict loads. Kept byte-identical in structure to the trainer."""
    import torch
    import torch.nn as nn
    import torchvision

    class BCHead(nn.Module):
        def __init__(self, in_dim, action_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, action_dim),
            )

        def forward(self, feats):
            return self.net(feats)

    class BCPolicyNet(nn.Module):
        def __init__(self, proprio_dim, action_dim):
            super().__init__()
            backbone = torchvision.models.resnet18(weights=None)
            self.img_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.encoder = backbone
            self.proprio_mlp = nn.Sequential(
                nn.Linear(proprio_dim, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
            )
            self.head = BCHead(self.img_dim + 128, action_dim)

        def forward(self, img, proprio):
            feats = self.encoder(img)
            p = self.proprio_mlp(proprio)
            return self.head(torch.cat([feats, p], dim=1))

    return BCPolicyNet(proprio_dim, action_dim)


# ===========================================================================
# INFERENCE CONTEXT — model + normalization, loaded once, shared by all
# per-episode BCPolicy instances the batch wrapper creates.
# ===========================================================================
class _InferenceContext:
    """Holds the loaded model and normalization constants. Built once in
    main(), before the server binds its port; every BCPolicy created by
    BCPolicyBatch.reset_batch shares this — so constructing a fresh policy
    per episode is cheap (the ResNet is NOT rebuilt)."""

    def __init__(self, model, device, ckpt, checkpoint_path):
        self.model = model
        self.device = device
        self.checkpoint_path = checkpoint_path
        # Normalization constants (numpy, applied per-step).
        self.a_mean = np.asarray(ckpt["action_mean"], dtype=np.float32)
        self.a_std = np.asarray(ckpt["action_std"], dtype=np.float32)
        self.p_mean = np.asarray(ckpt["proprio_mean"], dtype=np.float32)
        self.p_std = np.asarray(ckpt["proprio_std"], dtype=np.float32)


# ===========================================================================
# BCPolicy — inference logic, lifted verbatim from lbm_phase2_serve.py
# ===========================================================================
class BCPolicy(Policy):
    """Serves the trained BC checkpoint against the lbm_eval harness.

    Logic is identical to Phase 2's in-process BCPolicy; the only change is
    that model + normalization come from a shared _InferenceContext rather
    than a closure, so the batch wrapper can cheaply make one per episode."""

    # Class-level: log the live observation structure exactly once across the
    # whole server lifetime, as a guard that the gRPC round-trip delivered
    # what we expect (Phase 1 never exercised the image path).
    _logged_obs = False

    def __init__(self, ctx: _InferenceContext):
        self._ctx = ctx
        self._step = 0

    def reset(self, seed=None, options=None):
        self._step = 0

    def get_policy_metadata(self):
        return PolicyMetadata(
            name="BC_v1", skill_type="pick_and_place_box",
            checkpoint_path=self._ctx.checkpoint_path, git_repo="lbm_eval",
            git_sha="phase3",
        )

    def _extract_proprio(self, observation):
        """Build the 18-dim proprio vector the SAME way prepare_data did:
        per arm, xyz(3) + rot6d(6), order right then left."""
        poses = observation.robot.actual.poses
        parts = []
        for arm in ARM_KEYS:
            X = poses[arm]
            xyz = np.asarray(X.translation(), dtype=np.float32)
            rot6d = _rot6d_from_matrix(X.rotation().matrix())
            parts.append(xyz)
            parts.append(rot6d)
        return np.concatenate(parts).astype(np.float32)   # (18,)

    def _extract_image(self, observation):
        """RGB from the probed-and-verified camera, preprocessed exactly as
        prepare_data did: 2x downscale 480x640->240x320, ImageNet norm, CHW."""
        rgb = observation.visuo[SCENE_CAMERA_LIVE].rgb.array  # (480,640,3)
        rgb = rgb[::2, ::2, :]                                # (240,320,3)
        img = rgb.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return np.transpose(img, (2, 0, 1)).astype(np.float32)  # (3,H,W)

    def step(self, observation):
        ctx = self._ctx

        # One-time guard log: confirm the gRPC-delivered observation has the
        # camera key and image shape Phase 2 verified in-process.
        if not BCPolicy._logged_obs:
            BCPolicy._logged_obs = True
            try:
                keys = list(observation.visuo.keys())
                rgb = observation.visuo[SCENE_CAMERA_LIVE].rgb.array
                print(f"[bc_policy_server] first observation OK — "
                      f"visuo keys={keys}, "
                      f"'{SCENE_CAMERA_LIVE}' rgb shape={rgb.shape} "
                      f"dtype={rgb.dtype}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[bc_policy_server] WARNING inspecting first "
                      f"observation: {type(e).__name__}: {e}", flush=True)

        # --- assemble network inputs ---
        proprio = self._extract_proprio(observation)
        img = self._extract_image(observation)
        # Clip normalized proprio to a sane range. INSURANCE against the
        # static-dimension blow-up diagnosed in Phase 3: the left arm is
        # parked throughout pick_and_place_box, so its proprio std was ~0.
        # With the original `+1e-6` std floor, a serve-time pose deviation of
        # even 0.002 became a normalized value of ~1e2-1e5 on those dims,
        # saturating the proprio MLP and flattening the output across all
        # scenes (the observed first-frame stutter). The real fix is the
        # std floor in lbm_phase2_train.prepare_data; once retrained, all
        # normalized values are O(1) and this clip never triggers (the moving
        # dims have real |z| ~2). Kept as a cheap guard regardless.
        proprio_n = np.clip((proprio - ctx.p_mean) / ctx.p_std, -5.0, 5.0)

        img_t = torch.from_numpy(img).unsqueeze(0).to(ctx.device)
        prop_t = torch.from_numpy(proprio_n).unsqueeze(0).to(ctx.device)

        # --- run policy ---
        with torch.no_grad():
            out = ctx.model(img_t, prop_t).cpu().numpy()[0]   # (20,) normalized

        # --- un-normalize to real action units ---
        action = out * ctx.a_std + ctx.a_mean                 # (20,)

        # --- unpack 20-dim: [R_xyz|R_rot6d|L_xyz|L_rot6d|RG|LG] ---
        r_xyz, r_rot6d = action[0:3], action[3:9]
        l_xyz, l_rot6d = action[9:12], action[12:18]
        r_grip, l_grip = float(action[18]), float(action[19])

        poses = {
            "right::panda": RigidTransform(
                RotationMatrix(_matrix_from_rot6d(r_rot6d)),
                r_xyz.astype(np.float64),
            ),
            "left::panda": RigidTransform(
                RotationMatrix(_matrix_from_rot6d(l_rot6d)),
                l_xyz.astype(np.float64),
            ),
        }
        grippers = {
            "right::panda_hand": r_grip,
            "left::panda_hand": l_grip,
        }
        self._step += 1
        return PosesAndGrippers(poses=poses, grippers=grippers)


# ===========================================================================
# BCPolicyBatch — the batch-interface wrapper run_policy_server requires.
# Structural copy of WaveAroundBatch (grpc_workspace/wave_around_policy_server
# .py): run_policy_server calls reset_batch / step_batch / get_policy_metadata,
# never plain reset/step.
# ===========================================================================
class BCPolicyBatch(Policy):
    """Adapts BCPolicy to the gRPC batch interface."""

    def __init__(self, ctx: _InferenceContext):
        self._ctx = ctx
        # Identifier used when the policy is called via the non-batch path.
        self._internal_uuid = uuid.uuid4()
        # One BCPolicy per client UUID (i.e. per concurrent episode).
        self._sub_policies: dict[uuid.UUID, BCPolicy] = {}

    def reset(self, seed=None, options=None):
        self.reset_batch({self._internal_uuid: seed}, options)

    def reset_batch(self, seeds, options=None):
        # BC inference is deterministic given the observation, so the per-scene
        # seed is intentionally ignored (the scene itself is already seeded by
        # scenario_index on the harness side). A fresh BCPolicy per episode is
        # cheap — the model lives in the shared _InferenceContext.
        for one_uuid in seeds:
            self._sub_policies[one_uuid] = BCPolicy(self._ctx)

    def get_policy_metadata(self):
        return BCPolicy(self._ctx).get_policy_metadata()

    def step(self, observation):
        actions = self.step_batch({self._internal_uuid: observation})
        return actions[self._internal_uuid]

    def step_batch(self, observations):
        return {
            one_uuid: self._sub_policies[one_uuid].step(obs)
            for one_uuid, obs in observations.items()
        }


def main():
    parser = argparse.ArgumentParser(
        description="BC policy gRPC server for the lbm_eval benchmark.")
    # Adds --server-uri, --batch-timeout-s, --batch-max-size, and the gRPC
    # message-size limits, all consumed by run_policy_server.
    LbmPolicyServerConfig.add_argparse_arguments(parser)
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT,
        help="Path to the BC checkpoint (.pt). Default: %(default)s")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[bc_policy_server] loading {args.checkpoint} (device={device}) ...",
          flush=True)
    # weights_only=False: the checkpoint also holds numpy normalization arrays,
    # which the PyTorch 2.6+ default loader refuses to unpickle. This file is
    # produced by our own lbm_phase2_train.py — trusted.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = _build_model(ckpt["proprio_dim"], ckpt["action_dim"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[bc_policy_server] model ready — proprio_dim={ckpt['proprio_dim']} "
          f"action_dim={ckpt['action_dim']} train_camera={ckpt['scene_camera']}",
          flush=True)

    ctx = _InferenceContext(model, device, ckpt, args.checkpoint)
    policy = BCPolicyBatch(ctx)

    # The model is fully loaded above BEFORE run_policy_server binds the port.
    # So "port open" == "server truly ready"; the evaluate client does its own
    # gRPC readiness wait on top. run_policy_server blocks forever.
    run_policy_server(policy, args)


if __name__ == "__main__":
    main()