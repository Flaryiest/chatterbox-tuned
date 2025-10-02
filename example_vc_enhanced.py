"""
Enhanced Voice Conversion Example with Advanced Pre/Post-Processing

This demonstrates the new audio processing features for improved target speaker similarity.
"""

import torch
import torchaudio as ta
from chatterbox.vc import ChatterboxVC

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# =============================================================================
# Configuration
# =============================================================================

SOURCE_AUDIO = "YOUR_SOURCE_FILE.wav"
TARGET_VOICE = "YOUR_TARGET_FILE.wav"

# Core VC parameters
FLOW_CFG_RATE = 0.8      # Guidance strength (0.7-1.2 typical)
SPEAKER_STRENGTH = 1.0    # Speaker embedding scaling (1.0-1.5 typical)
PRUNE_TOKENS = 0         # Token pruning (0 = disabled)

# Pre-processing parameters (NEW!)
ENABLE_PREPROCESSING = True
PREPROCESSING_STRENGTH = 0.7  # 0.0-1.0, how aggressively to neutralize source

# Post-processing parameters (NEW!)
ENABLE_POSTPROCESSING = True
POSTPROCESSING_STRENGTH = 0.8  # 0.0-1.0, how strongly to push toward target

# =============================================================================
# Load Model
# =============================================================================

print("Loading ChatterboxVC model...")
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

print("Model loaded successfully!")

# =============================================================================
# Generate Voice Conversion
# =============================================================================

print(f"\n{'='*60}")
print("Generating voice conversion...")
print(f"Source: {SOURCE_AUDIO}")
print(f"Target: {TARGET_VOICE}")
print(f"Pre-processing: {'ENABLED' if ENABLE_PREPROCESSING else 'DISABLED'} (strength={PREPROCESSING_STRENGTH})")
print(f"Post-processing: {'ENABLED' if ENABLE_POSTPROCESSING else 'DISABLED'} (strength={POSTPROCESSING_STRENGTH})")
print(f"{'='*60}\n")

wav = model.generate(
    audio=SOURCE_AUDIO,
    target_voice_path=TARGET_VOICE,
)

# Save output
OUTPUT_PATH = "output_enhanced.wav"
ta.save(OUTPUT_PATH, wav, model.sr)

print(f"\n✓ Voice conversion complete!")
print(f"✓ Saved to: {OUTPUT_PATH}")

# =============================================================================
# Comparison: Generate without processing
# =============================================================================

print(f"\n{'='*60}")
print("Generating baseline (no processing) for comparison...")
print(f"{'='*60}\n")

wav_baseline = model.generate(
    audio=SOURCE_AUDIO,
    target_voice_path=TARGET_VOICE,
    enable_preprocessing=False,
    enable_postprocessing=False,
)

BASELINE_PATH = "output_baseline.wav"
ta.save(BASELINE_PATH, wav_baseline, model.sr)

print(f"✓ Baseline saved to: {BASELINE_PATH}")

# =============================================================================
# Experimentation Guide
# =============================================================================

print(f"\n{'='*60}")
print("EXPERIMENTATION GUIDE")
print(f"{'='*60}")
print("""
The new processing pipeline has been applied! Here's what each component does:

PRE-PROCESSING (applied to source audio):
  1. Spectral preemphasis - Reduces low-frequency speaker cues
  2. Pitch normalization - Shifts source pitch toward target
  3. Spectral whitening - Reduces timbre fingerprints
  4. Dynamics normalization - Equalizes energy patterns
  5. High-pass filtering - Removes deep voice characteristics

POST-PROCESSING (applied to output audio):
  1. Prosody alignment - Adjusts F0 contour to match target
  2. Formant shifting - Modifies vocal tract resonances
  3. Spectral envelope matching - Fine-tunes overall timbre

TUNING RECOMMENDATIONS:

If output still sounds too much like source:
  ✓ Increase preprocessing_strength (try 0.8-0.9)
  ✓ Increase postprocessing_strength (try 0.9-1.0)
  ✓ Increase speaker_strength (try 1.2-1.5)
  ✓ Increase flow_cfg_rate (try 1.0-1.2)

If output loses clarity or has artifacts:
  ✓ Decrease preprocessing_strength (try 0.5-0.6)
  ✓ Decrease postprocessing_strength (try 0.6-0.7)
  ✓ Decrease flow_cfg_rate (try 0.6-0.7)

If linguistic content is distorted:
  ✓ Disable pre-processing (enable_preprocessing=False)
  ✓ Keep only post-processing active
  ✓ Lower preprocessing_strength to 0.3-0.5

ADVANCED USAGE:

# Per-call overrides (doesn't change model defaults)
wav = model.generate(
    audio=SOURCE_AUDIO,
    target_voice_path=TARGET_VOICE,
    preprocessing_strength=0.9,      # Override for this call
    postprocessing_strength=0.95,    # Override for this call
    flow_cfg_rate=1.0,               # Override for this call
)

# Try different combinations
combinations = [
    {"preprocessing_strength": 0.5, "postprocessing_strength": 0.7},
    {"preprocessing_strength": 0.7, "postprocessing_strength": 0.8},
    {"preprocessing_strength": 0.9, "postprocessing_strength": 0.9},
]

for i, params in enumerate(combinations):
    wav = model.generate(
        audio=SOURCE_AUDIO,
        target_voice_path=TARGET_VOICE,
        **params
    )
    ta.save(f"output_combo_{i}.wav", wav, model.sr)
    print(f"Generated combo {i}: {params}")
""")

print(f"\n{'='*60}")
print("Done! Compare output_enhanced.wav with output_baseline.wav")
print(f"{'='*60}\n")
