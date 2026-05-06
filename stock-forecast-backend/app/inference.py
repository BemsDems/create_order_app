"""Pure-numpy inference for the multi-horizon TCN models (short/medium/long).

Avoids TensorFlow / sklearn / pandas at runtime so the container fits in a
shared-cpu-1x Fly.io machine (≤ 256 MiB).

Architecture (mirrors `build_tcn_model` from the training bundle):

    Input (T=30, F=28)
    LayerNormalization (eps=1e-3)
    Conv1D(32, k=3, causal, dilation=1) → ReLU → BatchNorm
    Conv1D(32, k=3, causal, dilation=2) → ReLU → BatchNorm
    Conv1D(16, k=3, causal, dilation=4) → ReLU → BatchNorm
    GlobalAveragePooling1D
    Dense(64, ReLU)
    Dense(32, ReLU)
    Dense(1, Sigmoid)

Three independent ensembles trained on different horizons:
    short  → trained on 5-day target  (used for horizons 5, 10)
    medium → trained on 30-day target (used for horizons 30, 60)
    long   → trained on 120-day target (used for horizons 120, 240, 365)

Each variant has its own RobustScaler fitted during training.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

SEQ_LEN = 30
N_FEATURES = 28

_ARTIFACTS_DIR = Path(__file__).resolve().parent / "models"

# RobustScaler center/scale baked in (extracted from scaler_{tag}.pkl).
_SCALERS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "short": (
        np.array([
            0.0, 0.0001671122999, 0.0004876859399, 0.001111552318, 0.002584751414,
            1.0, 1.0, 0.8969579373, 0.0, 51.34575569,
            0.0, 0.0, 0.5441640378, 0.01627555273, 0.04777246973,
            0.4219178082, 0.0, 3.766833036e-05, -6.751054855e-05, 0.007970890722,
            0.0, 0.0, 0.0004220992465, 0.002950230339, 0.01087952708,
            0.0, 0.0, 0.0,
        ], dtype=np.float32),
        np.array([
            0.01922836709, 0.02829015728, 0.03469559404, 0.04556108175, 0.06744915491,
            1.0, 1.0, 0.5544013198, 1.0, 24.64224535,
            1.0, 1.0, 0.5239335386, 0.008962552225, 0.08617290529,
            0.8, 1.0, 0.01049426459, 0.02315576631, 0.00565134997,
            1.0, 1.0, 0.01244144558, 0.03032079507, 0.05737323088,
            1.0, 1.0, 1.0,
        ], dtype=np.float32),
    ),
    "medium": (
        np.array([
            0.0, 0.0001542873892, 0.0004592976608, 0.001057082846, 0.002462911689,
            1.0, 1.0, 0.8966441952, 0.0, 51.27670441,
            0.0, 0.0, 0.5428132852, 0.01630629889, 0.04767694563,
            0.4191780822, 0.0, 0.0, -0.0001194053614, 0.007960866083,
            0.0, 0.0, 0.0004290445172, 0.002843311687, 0.01074913048,
            0.0, 0.0, 0.0,
        ], dtype=np.float32),
        np.array([
            0.0192954124, 0.02834859422, 0.0347654108, 0.04568789056, 0.06745753727,
            1.0, 1.0, 0.5529517751, 1.0, 24.64138259,
            1.0, 1.0, 0.5249324993, 0.008966755514, 0.08605703868,
            0.797260274, 1.0, 0.01049224542, 0.02320211661, 0.005692941712,
            1.0, 1.0, 0.012455368, 0.03024805966, 0.05757719511,
            1.0, 1.0, 1.0,
        ], dtype=np.float32),
    ),
    "long": (
        np.array([
            0.0, 0.0001331669276, 0.000429193272, 0.001029109226, 0.002412869847,
            1.0, 1.0, 0.8962627199, 0.0, 51.23915737,
            0.0, 0.0, 0.5423728813, 0.01639926253, 0.04723126602,
            0.4082191781, 0.0, -0.0001292156611, -0.0005710164247, 0.007966880303,
            0.0, 0.0, 0.0004983638379, 0.002899545723, 0.01128342439,
            0.0, 0.0, 0.0,
        ], dtype=np.float32),
        np.array([
            0.01949709199, 0.02861921989, 0.03509839746, 0.04594657429, 0.06781315905,
            1.0, 1.0, 0.5500310366, 1.0, 24.62070904,
            1.0, 1.0, 0.5252728514, 0.008981224389, 0.08558200674,
            0.7808219178, 1.0, 0.01035946661, 0.02290762263, 0.005593177726,
            1.0, 1.0, 0.01260572872, 0.03036286213, 0.05820836406,
            1.0, 1.0, 1.0,
        ], dtype=np.float32),
    ),
}

# Mapping from horizon (days) → which model+scaler to use, plus the absolute
# move threshold used for labelling during training (THR_MOVE per group).
HORIZON_GROUPS: dict[str, dict] = {
    "short":  {"horizons": (5, 10), "thr": 0.03, "trained_on": 5},
    "medium": {"horizons": (30, 60), "thr": 0.05, "trained_on": 30},
    "long":   {"horizons": (120, 240, 365), "thr": 0.12, "trained_on": 120},
}
HORIZON_TO_GROUP: dict[int, str] = {}
for _name, _info in HORIZON_GROUPS.items():
    for _h in _info["horizons"]:
        HORIZON_TO_GROUP[_h] = _name
ALLOWED_HORIZONS: tuple[int, ...] = tuple(sorted(HORIZON_TO_GROUP.keys()))


_lock = threading.Lock()
_weights: dict[str, dict[str, np.ndarray]] = {}


def horizon_to_group(horizon: int) -> str:
    if horizon not in HORIZON_TO_GROUP:
        raise ValueError(
            f"unsupported horizon {horizon}; allowed: {sorted(ALLOWED_HORIZONS)}"
        )
    return HORIZON_TO_GROUP[horizon]


def _load_group(group: str) -> dict[str, np.ndarray]:
    with _lock:
        if group not in _weights:
            path = _ARTIFACTS_DIR / f"tcn_{group}_weights.npz"
            npz = np.load(str(path))
            _weights[group] = {k: npz[k].astype(np.float32) for k in npz.files}
    return _weights[group]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def _causal_conv1d(
    x: np.ndarray,
    kernel: np.ndarray,
    bias: np.ndarray,
    dilation: int,
) -> np.ndarray:
    """Keras Conv1D with padding='causal'.

    x:      (T, F_in)
    kernel: (k, F_in, F_out)
    bias:   (F_out,)
    dilation: int
    """
    k, _, f_out = kernel.shape
    pad = (k - 1) * dilation
    xp = np.concatenate([np.zeros((pad, x.shape[1]), dtype=x.dtype), x], axis=0)
    out = bias.reshape(1, -1).repeat(x.shape[0], axis=0).astype(np.float32)
    for ki in range(k):
        # Window slice: positions ki*dilation..ki*dilation+T from xp
        offset = ki * dilation
        out += xp[offset:offset + x.shape[0]] @ kernel[ki]
    return out


def _batch_norm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    moving_mean: np.ndarray,
    moving_var: np.ndarray,
    eps: float = 1e-3,
) -> np.ndarray:
    return gamma * (x - moving_mean) / np.sqrt(moving_var + eps) + beta


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _robust_scale(features: np.ndarray, group: str) -> np.ndarray:
    center, scale = _SCALERS[group]
    return (features - center) / scale


def predict_proba(features: np.ndarray, horizon: int = 5) -> tuple[float, str, float]:
    """Run inference for the given horizon.

    Returns (prob_up, group_name, threshold).
        prob_up: probability that the close in `horizon` trading days is at
                 least `threshold` (e.g. 0.05 for medium) above today's close.
        group_name: 'short' | 'medium' | 'long' (for diagnostics).
        threshold: the move threshold used by the model during training.
    """
    if features.shape[0] < SEQ_LEN:
        raise ValueError(f"need at least {SEQ_LEN} rows, got {features.shape[0]}")
    if features.shape[1] != N_FEATURES:
        raise ValueError(
            f"feature mismatch: got {features.shape[1]}, expected {N_FEATURES}"
        )
    group = horizon_to_group(horizon)
    w = _load_group(group)

    scaled = _robust_scale(features[-SEQ_LEN:], group).astype(np.float32)

    x = _layer_norm(scaled, w["ln_g"], w["ln_b"])

    x = _causal_conv1d(x, w["conv0_k"], w["conv0_b"], dilation=1)
    x = _relu(x)
    x = _batch_norm(x, w["bn0_g"], w["bn0_b"], w["bn0_mm"], w["bn0_mv"])

    x = _causal_conv1d(x, w["conv1_k"], w["conv1_b"], dilation=2)
    x = _relu(x)
    x = _batch_norm(x, w["bn1_g"], w["bn1_b"], w["bn1_mm"], w["bn1_mv"])

    x = _causal_conv1d(x, w["conv2_k"], w["conv2_b"], dilation=4)
    x = _relu(x)
    x = _batch_norm(x, w["bn2_g"], w["bn2_b"], w["bn2_mm"], w["bn2_mv"])

    x = x.mean(axis=0)  # GlobalAveragePooling1D

    x = _relu(x @ w["d0_k"] + w["d0_b"])
    x = _relu(x @ w["d1_k"] + w["d1_b"])
    prob = float(_sigmoid(x @ w["d2_k"] + w["d2_b"]).ravel()[0])

    return prob, group, float(HORIZON_GROUPS[group]["thr"])


def warmup() -> None:
    dummy = np.zeros((SEQ_LEN, N_FEATURES), dtype=np.float32)
    for g in HORIZON_GROUPS:
        h = HORIZON_GROUPS[g]["horizons"][0]
        predict_proba(dummy, horizon=h)
