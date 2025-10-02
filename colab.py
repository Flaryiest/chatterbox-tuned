"""Colab Voice Conversion Demo with Identity Shift Metrics.

This script:
1. Loads a pretrained ChatterboxVC model.
2. Converts a source audio clip into a target speaker voice.
3. Saves and plays back the converted audio.
4. Computes objective speaker similarity metrics (source↔target, output↔source, output↔target) to quantify identity shift.

Adjust the AUDIO_PATH and TARGET_VOICE_PATH below. The last assignment to AUDIO_PATH is the one used.
"""

import torch
import soundfile as sf
import librosa
import numpy as np
from IPython.display import Audio, display
import time
import os
from datetime import datetime

# Optional: attempt lightweight dependency installs when running inside Colab runtime (fresh environment)
try:
    IN_COLAB = 'google.colab' in str(get_ipython())  # type: ignore  # noqa
except Exception:
    IN_COLAB = False

if IN_COLAB:
    # Heuristic: install missing optional deps only if not already present
    try:
        import pyloudnorm  # noqa: F401
    except ImportError:  # pragma: no cover
        !pip -q install pyloudnorm soundfile scipy >/dev/null 2>&1  # type: ignore  # noqa

from chatterbox import (
    preprocess_reference,
    preprocess_source,
    ReferencePreprocessConfig,
)

from chatterbox.vc import ChatterboxVC
from chatterbox.models.voice_encoder import VoiceEncoder

# ---------------- User Config ----------------
# (Only the final assignment to AUDIO_PATH is used; remove or comment out unused examples.)
SOURCE_AUDIO = "/content/TaylorSwiftShort.wav"  # Active source (unprocessed)
TARGET_VOICE_PATH = "/content/Barack Obama.mp3"  # Single target reference (unprocessed)

# Preprocessing toggles (do not alter original SOURCE_AUDIO / TARGET_VOICE_PATH paths)
ENABLE_PREPROCESS_REFERENCE = True      # Clean & stabilize target reference
ENABLE_PREPROCESS_SOURCE = True         # Neutralize source timbre prior to conversion
FAST_PREPROCESS = True                  # Use fast_mode (skips gate & stable window & full neutralization)
METRICS_USE_PREPROCESSED = True         # Whether metrics compare against preprocessed reference/source
POST_SMOOTH_RASPINESS = True            # Apply light spectral smoothing after generation
POST_SMOOTH_ALPHA = 0.15                # 0-1; higher = stronger smoothing
FORCE_SLOW_PREPROCESS = False           # Override to run full (slow) pipeline even if FAST_PREPROCESS True

# Output preprocessed artifact paths (kept separate to preserve original paths requested)
PREPROCESSED_REFERENCE_PATH = "/content/target_reference_preproc.wav"
PREPROCESSED_SOURCE_PATH = "/content/source_neutral_preproc.wav"

FLOW_CFG_RATE =  0.70       # Strong style guidance (try 0.82–0.88 first if artifacts)
SPEAKER_STRENGTH = 1.1     # Embedding scaling (1.15–1.30 typical)
PRUNE_TOKENS = 0            # 4–8 to reduce source leakage
ENABLE_PITCH_MATCH = True  # Use pitch matching hook
PITCH_TOLERANCE = 0.6      # Ignore tiny shifts (semitones)
MAX_PITCH_SHIFT = 2.0       # Clamp extreme shifts
RUN_VARIANT_SWEEP = False  # Set True to automatically evaluate a small grid
# Enable large grid sweep (set True to run after primary example). This supersedes RUN_VARIANT_SWEEP.
RUN_LARGE_GRID = False

# Large grid configuration (no prune tokens as requested)
GRID_FLOW_CFG_RATES_BASE = [0.0, 0.5, 0.8, 1.2, 1.6, 2.0, 2.5]
GRID_SPEAKER_STRENGTHS_BASE = [1.0, 1.2, 1.3, 1.4, 1.5]
# Subset for ramped refinement (picked from mid & upper region)
GRID_FLOW_CFG_RATES_RAMP = [0.8, 1.2, 1.6, 2.0]
GRID_SPEAKER_STRENGTHS_RAMP = [1.2, 1.3, 1.4]
GUIDANCE_RAMP_MIN_VALUES = [0.25, 0.4]

EXPORT_BASE_CSV = "/content/grid_base.csv"
EXPORT_RAMP_CSV = "/content/grid_ramp.csv"
EXPORT_JSON = "/content/grid_all.json"

MAX_BASE_RUNS = None  # set an int to early stop the base grid (debug)
MAX_RAMP_RUNS = None  # set an int to early stop the ramp grid

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ---------------- Optional Preprocessing Stage ----------------
def ts(msg: str):
    print(f"[{datetime.utcnow().isoformat()}] {msg}")

stage_times = {}
stage_t0 = time.time()

if ENABLE_PREPROCESS_REFERENCE or ENABLE_PREPROCESS_SOURCE:
    ts("[PRE] Starting preprocessing pipeline ...")
    ref_used_path = TARGET_VOICE_PATH
    src_used_path = SOURCE_AUDIO
    ref_audio_24k = None
    # 1. Reference preprocessing
    if ENABLE_PREPROCESS_REFERENCE:
        try:
            effective_fast = FAST_PREPROCESS and not FORCE_SLOW_PREPROCESS
            ref_cfg = ReferencePreprocessConfig(fast_mode=effective_fast, apply_gate=not effective_fast, use_stable_window=not effective_fast)
            ref_audio_24k, ref_info = preprocess_reference(TARGET_VOICE_PATH, ref_cfg, collect_timing=True)
            sf.write(PREPROCESSED_REFERENCE_PATH, ref_audio_24k, ref_info["sample_rate"])
            ref_used_path = PREPROCESSED_REFERENCE_PATH
            stage_times['ref_preprocess_sec'] = ref_info.get('total_sec', None)
            # Stats / hash
            import hashlib
            def audio_stats(arr, sr):
                return dict(duration_sec=len(arr)/sr, rms=float(np.sqrt(np.mean(arr**2)+1e-9)), peak=float(np.max(np.abs(arr))), centroid=float(librosa.feature.spectral_centroid(y=arr, sr=sr).mean()))
            raw_ref, _ = librosa.load(TARGET_VOICE_PATH, sr=ref_info['sample_rate'], mono=True)
            stats_raw = audio_stats(raw_ref, ref_info['sample_rate'])
            stats_proc = audio_stats(ref_audio_24k, ref_info['sample_rate'])
            md5_proc = hashlib.md5(ref_audio_24k.tobytes()).hexdigest()
            diff_mean = float(np.mean(np.abs(raw_ref[:len(ref_audio_24k)] - ref_audio_24k)))
            ts(f"[PRE][REF] Saved preprocessed reference -> {ref_used_path} total={ref_info.get('total_sec')}s fast={effective_fast} steps={ref_info.get('applied_steps')} timing={ref_info.get('timing', {})} md5={md5_proc} diff_mean={diff_mean:.6f} raw_stats={stats_raw} proc_stats={stats_proc}")
        except Exception as e:  # pragma: no cover
            ts(f"[PRE][REF][WARN] Failed preprocessing reference ({e}); falling back to raw file")
    # 2. Source preprocessing (needs reference downsampled)
    if ENABLE_PREPROCESS_SOURCE:
        try:
            if ref_audio_24k is None:
                # Load if not already done
                temp_ref, _ = librosa.load(TARGET_VOICE_PATH, sr=24000, mono=True)
            else:
                temp_ref = ref_audio_24k
            ref_16 = librosa.resample(temp_ref, orig_sr=24000, target_sr=16000)
            src_result = preprocess_source(SOURCE_AUDIO, ref_16, source_sr=16000, fast_mode=effective_fast, collect_timing=True)
            if isinstance(src_result, tuple):
                src_neutral, src_info = src_result
                stage_times['src_preprocess_sec'] = src_info.get('total_sec', None)
            else:
                src_neutral = src_result
            sf.write(PREPROCESSED_SOURCE_PATH, src_neutral, 16000)
            src_used_path = PREPROCESSED_SOURCE_PATH
            import hashlib
            raw_src, _ = librosa.load(SOURCE_AUDIO, sr=16000, mono=True)
            def src_stats(arr):
                return dict(duration_sec=len(arr)/16000, rms=float(np.sqrt(np.mean(arr**2)+1e-9)), peak=float(np.max(np.abs(arr))))
            stats_src_raw = src_stats(raw_src)
            stats_src_proc = src_stats(src_neutral)
            md5_src = hashlib.md5(src_neutral.tobytes()).hexdigest()
            diff_src_mean = float(np.mean(np.abs(raw_src[:len(src_neutral)] - src_neutral)))
            ts(f"[PRE][SRC] Saved preprocessed source -> {src_used_path} fast={effective_fast} info={src_info if 'src_info' in locals() else {}} md5={md5_src} diff_mean={diff_src_mean:.6f} raw_stats={stats_src_raw} proc_stats={stats_src_proc}")
        except Exception as e:  # pragma: no cover
            ts(f"[PRE][SRC][WARN] Failed preprocessing source ({e}); falling back to raw file")
else:
    ref_used_path = TARGET_VOICE_PATH
    src_used_path = SOURCE_AUDIO

stage_times['preprocess_total_sec'] = time.time() - stage_t0 if (ENABLE_PREPROCESS_REFERENCE or ENABLE_PREPROCESS_SOURCE) else 0.0

# ---------------- Load Model ----------------
load_t0 = time.time()
model = ChatterboxVC.from_pretrained(
    device,
    flow_cfg_rate=FLOW_CFG_RATE,
    speaker_strength=SPEAKER_STRENGTH,
    prune_tokens=PRUNE_TOKENS,
)
stage_times['model_load_sec'] = time.time() - load_t0

# Prepare target conditioning (single reference) — use processed if enabled
model.set_target_voice(ref_used_path)

def generate_with_introspection(model, audio_path, target_path, **override):
    """Generate audio while logging active parameters and token stats."""
    start = time.time()
    wav = model.generate(
        audio=audio_path,
        target_voice_path=target_path,
        **override,
    )
    dur = time.time() - start
    # Introspect parameters post-call with robust fallbacks
    try:
        cfg_used = model.s3gen.flow.decoder.cfm_params.inference_cfg_rate  # original expected path
    except Exception:
        cfg_used = getattr(model.s3gen, '_inference_cfg_rate', None)
    spk_strength_used = getattr(model.s3gen, 'speaker_strength', None)
    prune_used = override.get('prune_tokens', model.prune_tokens)
    print(f"[GEN] time={dur:.2f}s cfg_rate={cfg_used} speaker_strength={spk_strength_used} prune_tokens={prune_used}")
    return wav

# ---------------- Conversion (Primary) ----------------
gen_t0 = time.time()
wav = generate_with_introspection(
    model,
    src_used_path,
    ref_used_path,
    pitch_match=ENABLE_PITCH_MATCH,
    pitch_tolerance=PITCH_TOLERANCE,
    max_pitch_shift=MAX_PITCH_SHIFT,
)
stage_times['generation_sec'] = time.time() - gen_t0

# ---------------- Optional Post Smoothing (Raspiness Mitigation) ----------------
def post_spectral_smooth(wav_tensor, sr, alpha=0.15):
    if alpha <= 0: return wav_tensor
    y = wav_tensor.squeeze(0).cpu().numpy()
    S = librosa.stft(y, n_fft=1024, hop_length=256)
    mag, phase = np.abs(S), np.angle(S)
    # Simple spectral moving average across frequency bins
    k = 3
    mag_pad = np.pad(mag, ((0,0),(k,k)), mode='edge')
    smoothed = np.stack([mag_pad[:, i:i+mag.shape[1]] for i in range(2*k+1)], axis=0).mean(axis=0)
    mag_interp = (1-alpha)*mag + alpha*smoothed
    Y = mag_interp * np.exp(1j*phase)
    y_out = librosa.istft(Y, hop_length=256)
    if y_out is None:
        return wav_tensor
    return torch.from_numpy(y_out).float().unsqueeze(0)

post_t0 = time.time()
if POST_SMOOTH_RASPINESS:
    wav = post_spectral_smooth(wav, model.sr, alpha=POST_SMOOTH_ALPHA)
stage_times['post_smooth_sec'] = time.time() - post_t0 if POST_SMOOTH_RASPINESS else 0.0

out_path = "/content/output.wav"
sf.write(out_path, wav.squeeze(0).cpu().numpy(), model.sr)

display(Audio(filename=out_path, rate=model.sr))
print("Saved:", out_path)
print(f"Settings -> flow_cfg_rate={FLOW_CFG_RATE}, speaker_strength={SPEAKER_STRENGTH}, prune_tokens={PRUNE_TOKENS}, pitch_match={ENABLE_PITCH_MATCH}")
print("Timing Summary (seconds):", stage_times)

# ---------------- Identity Shift Evaluation ----------------
def load_embeds_utterance(path: str, ve: VoiceEncoder, sr_target: int = 16000):
    """Return both utterance-level (speaker) embedding and raw partial embeddings for richer metrics."""
    wav, _ = librosa.load(path, sr=sr_target)
    # as_spk=False returns utterance embeddings (partials) shape (U, E)
    partial_embeds = ve.embeds_from_wavs([wav], sample_rate=sr_target, as_spk=False)
    partial_embeds_t = torch.from_numpy(partial_embeds)  # (U, E)
    spk_embed = VoiceEncoder.utt_to_spk_embed(partial_embeds)  # aggregated
    spk_embed_t = torch.from_numpy(spk_embed)
    return spk_embed_t, partial_embeds_t

def cosine(a: torch.Tensor, b: torch.Tensor):
    return float(torch.dot(a, b) / (a.norm() * b.norm()))

def l2(a: torch.Tensor, b: torch.Tensor):
    return float(torch.norm(a - b))

voice_encoder = VoiceEncoder().to(device).eval()

# Decide which paths to use for metric baselines
metric_source_path = PREPROCESSED_SOURCE_PATH if (METRICS_USE_PREPROCESSED and ENABLE_PREPROCESS_SOURCE and os.path.exists(PREPROCESSED_SOURCE_PATH)) else SOURCE_AUDIO
metric_target_path = PREPROCESSED_REFERENCE_PATH if (METRICS_USE_PREPROCESSED and ENABLE_PREPROCESS_REFERENCE and os.path.exists(PREPROCESSED_REFERENCE_PATH)) else TARGET_VOICE_PATH

source_spk, source_partials = load_embeds_utterance(metric_source_path, voice_encoder)
target_spk, target_partials = load_embeds_utterance(metric_target_path, voice_encoder)
output_spk, output_partials = load_embeds_utterance(out_path, voice_encoder)

sim_source_target = cosine(source_spk, target_spk)
sim_output_source = cosine(output_spk, source_spk)
sim_output_target = cosine(output_spk, target_spk)
identity_gain = sim_output_target - sim_output_source

# Partial-level averaged metrics (can be more discriminative)
def mean_pairwise_cos(A: torch.Tensor, B: torch.Tensor):
    # A: (m,E), B: (n,E)
    return float((A @ B.T).mean() / (torch.norm(A, dim=1).mean() * torch.norm(B, dim=1).mean()))

partial_cos_out_target = mean_pairwise_cos(output_partials, target_partials)
partial_cos_out_source = mean_pairwise_cos(output_partials, source_partials)
partial_gain = partial_cos_out_target - partial_cos_out_source

spk_l2_source = l2(output_spk, source_spk)
spk_l2_target = l2(output_spk, target_spk)

print("\n[Identity Shift Metrics – Speaker Level]")
print(f"Cos(source, target): {sim_source_target:.4f}")
print(f"Cos(output, source): {sim_output_source:.4f}")
print(f"Cos(output, target): {sim_output_target:.4f}")
print(f"Identity gain (target - source): {identity_gain:.4f}")
print(f"L2(output, source): {spk_l2_source:.4f}")
print(f"L2(output, target): {spk_l2_target:.4f}")

print("\n[Identity Shift Metrics – Partial/Segment Level]")
print(f"Mean partial cos (out vs source): {partial_cos_out_source:.4f}")
print(f"Mean partial cos (out vs target): {partial_cos_out_target:.4f}")
print(f"Partial identity gain: {partial_gain:.4f}")

if (sim_output_target < sim_output_source) or (partial_cos_out_target < partial_cos_out_source):
    print("WARNING: Output closer to source than target on at least one metric -> adjust parameters (raise flow_cfg_rate/speaker_strength, enable prune_tokens).")
else:
    print("SUCCESS: Output closer to target across primary metrics.")

# Optional: quick variant comparison (uncomment to explore)
"""Variant sweep helper.
Set RUN_VARIANT_SWEEP=True at top to enable. Produces a table of metrics for different parameter combos.
"""
def run_variant(tag, flow_cfg_rate=None, speaker_strength=None, prune_tokens=None):
    wav_v = generate_with_introspection(
        model,
        src_used_path,
        ref_used_path,
        flow_cfg_rate=flow_cfg_rate,
        speaker_strength=speaker_strength,
        prune_tokens=prune_tokens,
        pitch_match=ENABLE_PITCH_MATCH,
        pitch_tolerance=PITCH_TOLERANCE,
        max_pitch_shift=MAX_PITCH_SHIFT,
    )
    p = f"/content/output_{tag}.wav"
    sf.write(p, wav_v.squeeze(0).cpu().numpy(), model.sr)
    out_spk_v, out_part_v = load_embeds_utterance(p, voice_encoder)
    return dict(
        tag=tag,
        cfg=flow_cfg_rate,
        strength=speaker_strength,
        prune=prune_tokens,
        cos_out_tgt=cosine(out_spk_v, target_spk),
        cos_out_src=cosine(out_spk_v, source_spk),
        l2_out_tgt=l2(out_spk_v, target_spk),
        l2_out_src=l2(out_spk_v, source_spk),
        partial_gain=(mean_pairwise_cos(out_part_v, target_partials) - mean_pairwise_cos(out_part_v, source_partials)),
    )

if RUN_VARIANT_SWEEP:
    variant_grid = [
        (0.82, 1.15, 0),
        (0.86, 1.20, 4),
        (0.90, 1.25, 8),
    ]
    rows = [run_variant(f"v{i}", cfg, strength, prune) for i, (cfg, strength, prune) in enumerate(variant_grid)]
    print("\n[Variant Sweep]")
    for r in rows:
        print(r)

# ---------------- Large Grid Sweep (No prune tokens) ----------------
"""Large grid exploration.

Two-stage approach:
1. Base grid without ramps.
2. Ramped refinement on a subset of (cfg, speaker_strength) pairs drawn from the higher-performing region.

Metrics captured per run:
 - cfg_rate, speaker_strength
 - guidance_ramp (bool), speaker_ramp (bool)
 - guidance_ramp_min (if used)
 - cos_out_tgt, cos_out_src, identity_gain
 - l2_out_tgt, l2_out_src, l2_advantage
 - partial_gain
 - runtime_sec
 - applied_pitch_shift (if pitch matching active)

Adjust the *_BASE / *_RAMP lists above to tune coverage. No prune token variation per request.
Set RUN_LARGE_GRID=True to execute.
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Iterable, Tuple

@dataclass
class RunResult:
    cfg_rate: float
    speaker_strength: float
    guidance_ramp: bool
    speaker_ramp: bool
    guidance_ramp_min: float | None
    cos_out_tgt: float
    cos_out_src: float
    identity_gain: float
    l2_out_tgt: float
    l2_out_src: float
    l2_advantage: float
    partial_gain: float
    runtime_sec: float
    applied_pitch_shift: float | None

def _eval_single(cfg_rate: float,
                 speaker_strength: float,
                 guidance_ramp: bool,
                 speaker_ramp: bool,
                 guidance_ramp_min: float | None,
                 tag: str = "") -> RunResult:
    start = time.time()
    wav_local = model.generate(
        audio=src_used_path,
        target_voice_path=ref_used_path,
        flow_cfg_rate=cfg_rate,
        speaker_strength=speaker_strength,
        prune_tokens=0,
        pitch_match=ENABLE_PITCH_MATCH,
        pitch_tolerance=PITCH_TOLERANCE,
        max_pitch_shift=MAX_PITCH_SHIFT,
        guidance_ramp=guidance_ramp,
        guidance_ramp_min=guidance_ramp_min if guidance_ramp else None,
        guidance_ramp_max=None,
        speaker_ramp=speaker_ramp,
        speaker_ramp_start=0.6,
        ramp_shape="sigmoid",
    )
    runtime = time.time() - start
    # Write temp file (optional listening) - can be skipped for speed
    out_tmp = f"/content/tmp_grid_{tag or 'x'}.wav"
    try:
        sf.write(out_tmp, wav_local.squeeze(0).cpu().numpy(), model.sr)
    except Exception:
        pass
    out_spk_v, out_part_v = load_embeds_utterance(out_tmp, voice_encoder)
    cos_tgt = cosine(out_spk_v, target_spk)
    cos_src = cosine(out_spk_v, source_spk)
    id_gain = cos_tgt - cos_src
    l2_tgt = l2(out_spk_v, target_spk)
    l2_src = l2(out_spk_v, source_spk)
    p_gain = (mean_pairwise_cos(out_part_v, target_partials) -
              mean_pairwise_cos(out_part_v, source_partials))
    l2_adv = l2_src - l2_tgt
    pitch_shift = model.get_last_pitch_shift() if hasattr(model, 'get_last_pitch_shift') else None
    return RunResult(
        cfg_rate=cfg_rate,
        speaker_strength=speaker_strength,
        guidance_ramp=guidance_ramp,
        speaker_ramp=speaker_ramp,
        guidance_ramp_min=guidance_ramp_min if guidance_ramp else None,
        cos_out_tgt=cos_tgt,
        cos_out_src=cos_src,
        identity_gain=id_gain,
        l2_out_tgt=l2_tgt,
        l2_out_src=l2_src,
        l2_advantage=l2_adv,
        partial_gain=p_gain,
        runtime_sec=runtime,
        applied_pitch_shift=pitch_shift,
    )

def _build_base_pairs():
    for cfg in GRID_FLOW_CFG_RATES_BASE:
        for strength in GRID_SPEAKER_STRENGTHS_BASE:
            yield cfg, strength

def _build_ramp_pairs():
    for cfg in GRID_FLOW_CFG_RATES_RAMP:
        for strength in GRID_SPEAKER_STRENGTHS_RAMP:
            for ramp_min in GUIDANCE_RAMP_MIN_VALUES:
                # Evaluate combinations: ramp only, ramp+speaker_ramp, and (optionally) speaker_ramp only skipped to limit size
                yield cfg, strength, ramp_min, True, False   # guidance ramp only
                yield cfg, strength, ramp_min, True, True    # both ramps

def _select_promising(base_results: List[RunResult], top_k: int = 10) -> List[Tuple[float, float]]:
    # Score by identity_gain primary, l2_out_tgt secondary
    scored = sorted(base_results, key=lambda r: (r.identity_gain, -r.l2_out_tgt), reverse=True)
    picked = []
    seen = set()
    for r in scored:
        key = (r.cfg_rate, r.speaker_strength)
        if key in seen:
            continue
        picked.append(key)
        seen.add(key)
        if len(picked) >= top_k:
            break
    return picked

def _filter_ramp_pairs(promising_keys: List[Tuple[float, float]]):
    prom_set = set(promising_keys)
    for cfg, strength, ramp_min, g_ramp, s_ramp in _build_ramp_pairs():
        if (cfg, strength) in prom_set:
            yield cfg, strength, ramp_min, g_ramp, s_ramp

def _export_csv(path: str, rows: List[RunResult]):
    import csv
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    print(f"[GRID] Wrote {len(rows)} rows -> {path}")

def _export_json(path: str, base_rows: List[RunResult], ramp_rows: List[RunResult]):
    payload = {
        'base': [asdict(r) for r in base_rows],
        'ramp': [asdict(r) for r in ramp_rows],
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"[GRID] Wrote JSON -> {path}")

if RUN_LARGE_GRID:
    print("[GRID] Starting base grid sweep (no ramps)...")
    base_results: List[RunResult] = []
    for i, (cfg, strength) in enumerate(_build_base_pairs()):
        if MAX_BASE_RUNS is not None and i >= MAX_BASE_RUNS:
            print("[GRID] Early stop base grid due to MAX_BASE_RUNS")
            break
        res = _eval_single(cfg, strength, guidance_ramp=False, speaker_ramp=False, guidance_ramp_min=None, tag=f"b{i}")
        base_results.append(res)
        if (i + 1) % 10 == 0:
            print(f"[GRID][BASE] Completed {i+1} runs")
    if base_results:
        _export_csv(EXPORT_BASE_CSV, base_results)
    else:
        print("[GRID] No base results produced (empty grid?)")

    print("[GRID] Selecting promising pairs for ramp refinement...")
    promising = _select_promising(base_results, top_k=10)
    print(f"[GRID] Promising pairs: {promising}")

    print("[GRID] Starting ramp refinement sweep...")
    ramp_results: List[RunResult] = []
    for j, (cfg, strength, ramp_min, g_ramp, s_ramp) in enumerate(_filter_ramp_pairs(promising)):
        if MAX_RAMP_RUNS is not None and j >= MAX_RAMP_RUNS:
            print("[GRID] Early stop ramp grid due to MAX_RAMP_RUNS")
            break
        res = _eval_single(cfg, strength, guidance_ramp=g_ramp, speaker_ramp=s_ramp, guidance_ramp_min=ramp_min, tag=f"r{j}")
        ramp_results.append(res)
        if (j + 1) % 10 == 0:
            print(f"[GRID][RAMP] Completed {j+1} runs")
    if ramp_results:
        _export_csv(EXPORT_RAMP_CSV, ramp_results)
    else:
        print("[GRID] No ramp results produced (empty refinement set?)")

    if base_results or ramp_results:
        _export_json(EXPORT_JSON, base_results, ramp_results)

    # Print top 5 combined by identity_gain
    combined = base_results + ramp_results
    combined_sorted = sorted(combined, key=lambda r: (r.identity_gain, -r.l2_out_tgt), reverse=True)
    print("\n[GRID] Top 5 configurations:")
    for rr in combined_sorted[:5]:
        print(asdict(rr))

