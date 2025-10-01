"""Audio preprocessing utilities for voice cloning enhancement.

All functions are designed to be fast (< ~1-2s per several-second clip) and avoid heavy ML dependencies.
Some steps use optional libraries (pyloudnorm, scipy). If they are missing, fallbacks are used.

Main entry points:
- preprocess_reference(path, sr=24000, **kwargs) -> np.ndarray
- preprocess_source(path, reference_audio, source_sr=16000, **kwargs) -> np.ndarray

The goal is to (1) purify target/reference conditioning audio and (2) neutralize source audio
so the model more strongly latches onto target speaker identity.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import warnings
import numpy as np
import librosa

try:  # optional
    import pyloudnorm as pyln  # type: ignore
except ImportError:  # pragma: no cover
    pyln = None  # type: ignore

try:  # optional
    from scipy.signal import butter, sosfilt
except ImportError:  # pragma: no cover
    butter = None  # type: ignore
    sosfilt = None  # type: ignore

# ---------------- Basic Utilities ---------------- #

def load_and_trim(path: str, sr: int) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    yt, _ = librosa.effects.trim(y, top_db=25)
    return yt.astype(np.float32)


def loudness_normalize(y: np.ndarray, sr: int, target_lufs: float = -20.0) -> np.ndarray:
    if pyln is None:
        # Fallback: approximate by RMS gain toward a pseudo target
        rms = np.sqrt(np.mean(y ** 2) + 1e-9)
        target_rms = 10 ** (target_lufs / 20) * 0.1  # heuristic scaling
        if rms > 0:
            y = y * (target_rms / rms)
        peak = np.max(np.abs(y))
        if peak > 0.999:
            y = y / peak * 0.999
        return y
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y)
    gain = target_lufs - loudness
    y_out = y * (10 ** (gain / 20))
    peak = np.max(np.abs(y_out))
    if peak > 0.999:
        y_out /= peak / 0.999
    return y_out.astype(np.float32)


def spectral_gate(y: np.ndarray, sr: int, prop_decrease: float = 0.85) -> np.ndarray:
    """Very lightweight spectral gating denoise. Not a full NR system."""
    S = librosa.stft(y, n_fft=1024, hop_length=256)
    mag, phase = np.abs(S), np.angle(S)
    noise_profile = np.percentile(mag, 10, axis=1, keepdims=True)
    thresh = noise_profile * 1.3
    mask = (mag > thresh).astype(np.float32)
    mask = mask * (1 - prop_decrease) + (1 - prop_decrease)
    S_d = (mag * mask) * np.exp(1j * phase)
    y_d = librosa.istft(S_d, hop_length=256)
    if y_d is None:
        return y
    return y_d.astype(np.float32)


def pick_stable_window(y: np.ndarray, sr: int, win_sec: float = 4.0, step_sec: float = 0.5) -> np.ndarray:
    if len(y) < win_sec * sr:
        return y
    win = int(win_sec * sr)
    step = int(step_sec * sr)
    best = None
    best_score = 1e9
    for start in range(0, max(1, len(y) - win), step):
        seg = y[start:start + win]
        try:
            f0, _, _ = librosa.pyin(seg, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'), sr=sr)
            if f0 is None:
                continue
            valid = f0[np.isfinite(f0)]
            if valid.size == 0:
                continue
            f0v = np.var(valid)
            flat = librosa.feature.spectral_flatness(y=seg).mean()
            score = f0v + 0.5 * flat
            if score < best_score:
                best_score = score
                best = seg
        except Exception:
            continue
    return best.astype(np.float32) if best is not None else y


def spectral_tilt_normalize(y: np.ndarray, pre_emphasis: float = 0.97) -> np.ndarray:
    if y.size == 0:
        return y
    y_f = np.copy(y)
    y_f[1:] = y_f[1:] - pre_emphasis * y_f[:-1]
    # Normalize amplitude
    peak = np.max(np.abs(y_f)) + 1e-9
    y_f = y_f / peak * 0.99
    return y_f.astype(np.float32)

# ---------------- Source Neutralization ---------------- #

def smooth_formants(y: np.ndarray, sr: int, lifter: int = 30) -> np.ndarray:
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)) + 1e-9
    log_S = np.log(S)
    cep = np.fft.irfft(log_S, axis=0)
    if lifter < cep.shape[0]:
        cep[lifter:, :] = 0
    smooth_log_spec = np.fft.rfft(cep, axis=0).real
    S_smooth = np.exp(smooth_log_spec)
    phase = np.angle(librosa.stft(y, n_fft=1024, hop_length=256))
    Y = S_smooth * np.exp(1j * phase)
    y_out = librosa.istft(Y, hop_length=256)
    return (y_out if y_out is not None else y).astype(np.float32)


def match_rms(y: np.ndarray, ref: np.ndarray, target_db: float = -23.0) -> np.ndarray:
    def rms_db(x: np.ndarray) -> float:
        return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)
    # Align both roughly to target_db for consistency
    for arr in (y, ref):
        if np.max(np.abs(arr)) > 0.99:
            arr /= np.max(np.abs(arr)) + 1e-9
    # Compute gain
    y_db = rms_db(y)
    diff = target_db - y_db
    y_out = y * (10 ** (diff / 20))
    peak = np.max(np.abs(y_out))
    if peak > 0.999:
        y_out = y_out / peak * 0.999
    return y_out.astype(np.float32)


def neutralize_source(y: np.ndarray, ref: np.ndarray, sr: int) -> np.ndarray:
    y1 = smooth_formants(y, sr)
    # If reference shorter, just use overlapping region for energy reference
    ref_seg = ref[:min(len(ref), len(y1))]
    y2 = match_rms(y1, ref_seg)
    # Light normalization
    peak = np.max(np.abs(y2)) + 1e-9
    y2 = y2 / peak * 0.99
    return y2.astype(np.float32)

# ---------------- Band-limited Variant ---------------- #

def bandlimit(y: np.ndarray, sr: int, low: int = 300, high: int = 3400) -> Optional[np.ndarray]:
    if butter is None or sosfilt is None:
        return None
    try:
        sos = butter(6, [low, high], btype='band', fs=sr, output='sos')
        return sosfilt(sos, y).astype(np.float32)
    except Exception:
        return None

# ---------------- High-level Pipelines ---------------- #
@dataclass
class ReferencePreprocessConfig:
    sr: int = 24000
    target_lufs: float = -20.0
    apply_gate: bool = True
    stable_window_sec: float = 4.0
    use_stable_window: bool = True
    spectral_tilt: bool = True


def preprocess_reference(path: str, cfg: ReferencePreprocessConfig = ReferencePreprocessConfig()) -> Tuple[np.ndarray, dict]:
    y = load_and_trim(path, cfg.sr)
    y = loudness_normalize(y, cfg.sr, cfg.target_lufs)
    if cfg.apply_gate:
        y = spectral_gate(y, cfg.sr)
    if cfg.use_stable_window:
        y = pick_stable_window(y, cfg.sr, win_sec=cfg.stable_window_sec)
    if cfg.spectral_tilt:
        y = spectral_tilt_normalize(y)
    band = bandlimit(y, cfg.sr)
    info = {
        "length_samples": int(len(y)),
        "sample_rate": cfg.sr,
        "band_variant": band is not None,
    }
    return y, info


def preprocess_source(path: str, reference_audio: np.ndarray, source_sr: int = 16000) -> np.ndarray:
    y, _ = librosa.load(path, sr=source_sr, mono=True)
    return neutralize_source(y.astype(np.float32), reference_audio.astype(np.float32), source_sr)

# Convenience wrapper if both paths provided

def preprocess_pair(reference_path: str, source_path: str) -> Tuple[np.ndarray, np.ndarray, ReferencePreprocessConfig]:
    cfg = ReferencePreprocessConfig()
    ref_audio, _ = preprocess_reference(reference_path, cfg)
    # Downsample ref to 16k for source neutralization alignment segment
    ref_16 = librosa.resample(ref_audio, orig_sr=cfg.sr, target_sr=16000)
    src_audio = preprocess_source(source_path, ref_16, source_sr=16000)
    return ref_audio, src_audio, cfg

__all__ = [
    "preprocess_reference",
    "preprocess_source",
    "preprocess_pair",
    "ReferencePreprocessConfig",
]
