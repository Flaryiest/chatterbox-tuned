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
AUDIO_PATH = "/content/matrix.mp3"
AUDIO_PATH = "/content/TaylorSwiftShort.wav"  # Active source

TARGET_VOICE_PATH = "/content/Barack Obama.mp3"  # Target speaker reference

FLOW_CFG_RATE = 0.90        # Strong style guidance (try 0.82–0.88 first if artifacts)
SPEAKER_STRENGTH = 1.25     # Embedding scaling (1.15–1.30 typical)
PRUNE_TOKENS = 0            # Try 4–8 to reduce source leakage
RUN_VARIANT_SWEEP = False   # Set True to automatically evaluate a small grid

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ---------------- Load Model ----------------
model = ChatterboxVC.from_pretrained(
    device,
    flow_cfg_rate=FLOW_CFG_RATE,
    speaker_strength=SPEAKER_STRENGTH,
    prune_tokens=PRUNE_TOKENS,
)

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
    AUDIO_PATH,
    TARGET_VOICE_PATH,
)

out_path = "/content/output.wav"
sf.write(out_path, wav.squeeze(0).cpu().numpy(), model.sr)

display(Audio(filename=out_path, rate=model.sr))
print("Saved:", out_path)
print(f"Settings -> flow_cfg_rate={FLOW_CFG_RATE}, speaker_strength={SPEAKER_STRENGTH}, prune_tokens={PRUNE_TOKENS}")

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

source_spk, source_partials = load_embeds_utterance(AUDIO_PATH, voice_encoder)
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
        AUDIO_PATH,
        TARGET_VOICE_PATH,
        flow_cfg_rate=flow_cfg_rate,
        speaker_strength=speaker_strength,
        prune_tokens=prune_tokens,
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
