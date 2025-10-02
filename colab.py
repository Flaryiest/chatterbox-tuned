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

from chatterbox.vc import ChatterboxVC
from chatterbox.models.voice_encoder import VoiceEncoder

# ---------------- User Config ----------------
# (Only the final assignment to AUDIO_PATH is used; remove or comment out unused examples.)
SOURCE_AUDIO = "/content/TaylorSwiftShort.wav"  # Active source
TARGET_VOICE_PATH = "/content/Barack Obama.mp3"  # Single target reference

FLOW_CFG_RATE =  0.65       # Content/style balance (0.6-0.7 recommended)
SPEAKER_STRENGTH = 1.8      # 🔥 KEY PARAMETER: Higher = more target voice (1.5-2.5 range)
PRUNE_TOKENS = 8            # Slight quality improvement
ENABLE_PITCH_MATCH = True   # Use pitch matching hook
PITCH_TOLERANCE = 0.6       # Ignore tiny shifts (semitones)
MAX_PITCH_SHIFT = 2.0       # Clamp extreme shifts

# ========== PROCESSING OPTIONS (FIXED) ==========
# Pre-processing DISABLED: Caused word loss even at low strengths
# Post-processing now SAFE: Only gentle RMS matching, no artifacts
ENABLE_PREPROCESSING = False     # ❌ DISABLED: Too destructive
PREPROCESSING_STRENGTH = 0.0     # Not used when disabled
ENABLE_POSTPROCESSING = True     # ✅ SAFE: Gentle energy matching only
POSTPROCESSING_STRENGTH = 0.4    # 0.3-0.5 recommended (safe range)
# ================================================

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

# ---------------- Load Model ----------------
model = ChatterboxVC.from_pretrained(
    device,
    flow_cfg_rate=FLOW_CFG_RATE,
    speaker_strength=SPEAKER_STRENGTH,
    prune_tokens=PRUNE_TOKENS,
    enable_preprocessing=ENABLE_PREPROCESSING,
    enable_postprocessing=ENABLE_POSTPROCESSING,
    preprocessing_strength=PREPROCESSING_STRENGTH,
    postprocessing_strength=POSTPROCESSING_STRENGTH,
)

# Prepare target conditioning (single reference)
model.set_target_voice(TARGET_VOICE_PATH)

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
wav = generate_with_introspection(
    model,
    SOURCE_AUDIO,
    TARGET_VOICE_PATH,
    pitch_match=ENABLE_PITCH_MATCH,
    pitch_tolerance=PITCH_TOLERANCE,
    max_pitch_shift=MAX_PITCH_SHIFT,
)

out_path = "/content/output.wav"
sf.write(out_path, wav.squeeze(0).cpu().numpy(), model.sr)

display(Audio(filename=out_path, rate=model.sr))
print("Saved:", out_path)
print(f"Settings -> flow_cfg_rate={FLOW_CFG_RATE}, speaker_strength={SPEAKER_STRENGTH}, prune_tokens={PRUNE_TOKENS}, pitch_match={ENABLE_PITCH_MATCH}")

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

source_spk, source_partials = load_embeds_utterance(SOURCE_AUDIO, voice_encoder)
target_spk, target_partials = load_embeds_utterance(TARGET_VOICE_PATH, voice_encoder)
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
        SOURCE_AUDIO,
        TARGET_VOICE_PATH,
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
        audio=SOURCE_AUDIO,
        target_voice_path=TARGET_VOICE_PATH,
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

# ---------------- Parameter Verification Utility ----------------
"""Manual verification helper: run a miniature sweep over cfg_rate and speaker_strength
and confirm that:
 1. The internal flow trace reflects the requested cfg_rate (or schedule values).
 2. Increasing cfg_rate or speaker_strength increases early-step diff_norm (up to saturation).

Set RUN_VERIFY_PARAMS=True to execute. Adjust VERIFY_CFGS / VERIFY_STRENGTHS below.
"""

RUN_VERIFY_PARAMS = False
VERIFY_CFGS = [0.0, 0.8, 1.6, 2.5]
VERIFY_STRENGTHS = [1.0, 1.2, 1.4]
VERIFY_SOURCE = SOURCE_AUDIO
VERIFY_TARGET = TARGET_VOICE_PATH

def verify_parameters(model, cfg_values, strength_values):
    print("[VERIFY] Enabling trace...")
    model.s3gen.enable_param_trace(True)
    rows = []
    for cfg in cfg_values:
        model.s3gen.set_inference_cfg_rate(cfg)
        for strength in strength_values:
            model.s3gen.set_speaker_strength(strength)
            wav_v = model.generate(
                audio=VERIFY_SOURCE,
                target_voice_path=VERIFY_TARGET,
                flow_cfg_rate=cfg,
                speaker_strength=strength,
                prune_tokens=0,
                pitch_match=ENABLE_PITCH_MATCH,
                pitch_tolerance=PITCH_TOLERANCE,
                max_pitch_shift=MAX_PITCH_SHIFT,
            )
            trace = model.s3gen.get_last_flow_trace() or []
            if trace:
                first = trace[0]
                avg_diff = sum(t['diff_norm'] for t in trace) / len(trace)
                rows.append({
                    'cfg_rate': cfg,
                    'speaker_strength': strength,
                    'first_step_cfg_rate': first['cfg_rate'],
                    'first_step_diff_norm': first['diff_norm'],
                    'avg_diff_norm': avg_diff,
                })
            else:
                rows.append({
                    'cfg_rate': cfg,
                    'speaker_strength': strength,
                    'first_step_cfg_rate': None,
                    'first_step_diff_norm': None,
                    'avg_diff_norm': None,
                })
    # Basic monotonicity checks
    print("\n[VERIFY] Results:")
    for r in rows:
        print(r)
    # Group by speaker strength to see diff_norm vs cfg
    print("\n[VERIFY] Monotonicity by speaker strength:")
    from collections import defaultdict
    by_strength = defaultdict(list)
    for r in rows:
        by_strength[r['speaker_strength']].append(r)
    for strength, lst in by_strength.items():
        lst_sorted = sorted(lst, key=lambda x: x['cfg_rate'])
        diffs = [x['first_step_diff_norm'] for x in lst_sorted if x['first_step_diff_norm'] is not None]
        trend = 'increasing' if all(diffs[i] <= diffs[i+1] for i in range(len(diffs)-1)) else 'non-monotonic'
        print(f" speaker_strength={strength}: first_step_diff_norm sequence={diffs} -> {trend}")
    model.s3gen.enable_param_trace(False)
    return rows

if RUN_VERIFY_PARAMS:
    verify_parameters(model, VERIFY_CFGS, VERIFY_STRENGTHS)

