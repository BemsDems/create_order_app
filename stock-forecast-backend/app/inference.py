"""Pure-numpy inference for the SONNET v3 LSTM model.

Avoids TensorFlow, sklearn, and pandas at runtime to keep the container small
enough for a 256 MiB Fly.io machine.

Model architecture (mirrors `build_optimized_model()` from the training code):
    Input (T=30, F=15)
    LSTM(32, return_sequences=True)
    LayerNormalization
    LSTM(16)
    LayerNormalization
    Dense(12, relu)
    Dropout (no-op at inference)
    Dense(1, sigmoid)
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

SEQ_LEN = 30

_ARTIFACTS_DIR = Path(__file__).resolve().parent / "models"
_WEIGHTS_PATH = _ARTIFACTS_DIR / "sonnet_v3_weights.npz"

# RobustScaler parameters baked in — extracted from sonnet_v3_scaler.pkl
# (sklearn.preprocessing.RobustScaler fitted during training).
_SCALER_CENTER = np.array([
    0.00042015890266311473,
    0.0026083037668573894,
    0.006317784835943363,
    0.01206390036185534,
    0.00042007066049264526,
    0.005766927134013278,
    0.013948415898126898,
    1.0,
    54.07618699494307,
    -0.0073934983198239435,
    0.013980573041013585,
    0.5289117993810672,
    0.00055445255281239,
    0.574644707443371,
    0.17759520991899164,
], dtype=np.float32)

_SCALER_SCALE = np.array([
    0.013831051112786291,
    0.032255106735105565,
    0.04619836281489173,
    0.06863355211232045,
    0.013819528735036377,
    0.035535264145750894,
    0.06285961325417055,
    1.0,
    21.518270486887737,
    1.0942986008153788,
    0.0078000534764486075,
    1.8665077785336164,
    0.015622350262000984,
    0.4210686388089511,
    0.9685391559888291,
], dtype=np.float32)

N_FEATURES = _SCALER_CENTER.shape[0]

_lock = threading.Lock()
_weights: dict[str, np.ndarray] | None = None


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid.
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _lstm_forward(
    x: np.ndarray,
    kernel: np.ndarray,
    recurrent_kernel: np.ndarray,
    bias: np.ndarray,
    return_sequences: bool,
) -> np.ndarray:
    """Keras-compatible LSTM forward pass.

    x: (T, F). kernel: (F, 4U). recurrent_kernel: (U, 4U). bias: (4U,).
    Keras gate order inside the concatenated vector: [i, f, c_hat, o].
    """
    T, _ = x.shape
    U = recurrent_kernel.shape[0]
    h = np.zeros((U,), dtype=np.float32)
    c = np.zeros((U,), dtype=np.float32)

    xw = x @ kernel  # (T, 4U), precomputed

    outputs = np.empty((T, U), dtype=np.float32) if return_sequences else None
    for t in range(T):
        z = xw[t] + h @ recurrent_kernel + bias
        i = _sigmoid(z[:U])
        f = _sigmoid(z[U:2 * U])
        c_hat = np.tanh(z[2 * U:3 * U])
        o = _sigmoid(z[3 * U:])
        c = f * c + i * c_hat
        h = o * np.tanh(c)
        if return_sequences:
            outputs[t] = h
    if return_sequences:
        return outputs  # type: ignore[return-value]
    return h


def _layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Keras LayerNormalization (default eps=1e-3), last-axis normalization."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def _dense(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return x @ kernel + bias


def _load() -> dict[str, np.ndarray]:
    global _weights
    with _lock:
        if _weights is None:
            npz = np.load(str(_WEIGHTS_PATH))
            _weights = {
                "lstm0_k": npz["lstm__w0"].astype(np.float32),
                "lstm0_u": npz["lstm__w1"].astype(np.float32),
                "lstm0_b": npz["lstm__w2"].astype(np.float32),
                "ln0_g":   npz["layer_normalization__w0"].astype(np.float32),
                "ln0_b":   npz["layer_normalization__w1"].astype(np.float32),
                "lstm1_k": npz["lstm_1__w0"].astype(np.float32),
                "lstm1_u": npz["lstm_1__w1"].astype(np.float32),
                "lstm1_b": npz["lstm_1__w2"].astype(np.float32),
                "ln1_g":   npz["layer_normalization_1__w0"].astype(np.float32),
                "ln1_b":   npz["layer_normalization_1__w1"].astype(np.float32),
                "d0_k":    npz["dense__w0"].astype(np.float32),
                "d0_b":    npz["dense__w1"].astype(np.float32),
                "d1_k":    npz["dense_1__w0"].astype(np.float32),
                "d1_b":    npz["dense_1__w1"].astype(np.float32),
            }
    return _weights


def _robust_scale(features: np.ndarray) -> np.ndarray:
    """Mirrors sklearn RobustScaler.transform() using baked-in center/scale."""
    return (features - _SCALER_CENTER) / _SCALER_SCALE


def predict_proba(features: np.ndarray) -> float:
    """features: (N, 15) raw. Uses last SEQ_LEN rows. Returns P(up>=2% in 5d)."""
    w = _load()
    if features.shape[0] < SEQ_LEN:
        raise ValueError(f"need at least {SEQ_LEN} rows, got {features.shape[0]}")
    if features.shape[1] != N_FEATURES:
        raise ValueError(
            f"feature mismatch: got {features.shape[1]}, expected {N_FEATURES}"
        )
    scaled = _robust_scale(features[-SEQ_LEN:]).astype(np.float32)

    x = _lstm_forward(scaled, w["lstm0_k"], w["lstm0_u"], w["lstm0_b"], return_sequences=True)
    x = _layer_norm(x, w["ln0_g"], w["ln0_b"])
    x = _lstm_forward(x, w["lstm1_k"], w["lstm1_u"], w["lstm1_b"], return_sequences=False)
    x = _layer_norm(x, w["ln1_g"], w["ln1_b"])
    x = _dense(x, w["d0_k"], w["d0_b"])
    x = np.maximum(x, 0.0)  # relu
    x = _dense(x, w["d1_k"], w["d1_b"])
    return float(_sigmoid(x).ravel()[0])


def warmup() -> None:
    _load()
    dummy = np.zeros((SEQ_LEN, N_FEATURES), dtype=np.float32)
    predict_proba(dummy)
