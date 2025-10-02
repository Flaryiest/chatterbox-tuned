"""
ACTUAL SOLUTION to 50/50 voice blending issue.

The DSP pre/post-processing was causing artifacts. The REAL solution is:
1. Increase speaker_strength (controls how much target voice is used)
2. Adjust flow_cfg_rate (controls content vs style tradeoff)
3. Use longer/better target reference audio
4. Optionally: gentle post-processing for energy matching only

NO aggressive DSP needed!
"""

from src.chatterbox.vc import ChatterboxVC
import soundfile as sf

# ========================================
# SOLUTION 1: Increase speaker_strength
# ========================================
# This is the MAIN parameter that controls target voice similarity
# Default was 1.0, but 1.5-2.0 gives much stronger target voice

model = ChatterboxVC.from_pretrained(
    device="cuda",  # or "cpu"
    speaker_strength=1.8,  # 🔥 KEY PARAMETER: Higher = more like target voice
    flow_cfg_rate=0.7,     # Keep at 0.7 (good content preservation)
    prune_tokens=8,        # Optional: slight quality improvement
)

# Set your target voice
model.set_target_voice("audio/Barack Obama.mp3")

# Generate - should now be MUCH closer to target!
output_wav = model.generate(
    audio="input.wav",  # Your source audio
    speaker_strength=1.8,  # Can override per-call
)

sf.write("output_better.wav", output_wav.squeeze().cpu().numpy(), 24000)
print("✓ Generated with speaker_strength=1.8")

# ========================================
# SOLUTION 2: Try even higher values
# ========================================
# If still not similar enough, try 2.0-3.0
# (may lose some emotional nuance but much better identity)

output_stronger = model.generate(
    audio="input.wav",
    speaker_strength=2.5,  # Very strong target influence
)

sf.write("output_strongest.wav", output_stronger.squeeze().cpu().numpy(), 24000)
print("✓ Generated with speaker_strength=2.5")

# ========================================
# SOLUTION 3: Adjust cfg_rate if needed
# ========================================
# If losing too much content clarity:
# - Lower flow_cfg_rate (e.g., 0.5-0.6) = more content, less style
# If still not similar enough:
# - Keep flow_cfg_rate at 0.7-0.8 with high speaker_strength

model_balanced = ChatterboxVC.from_pretrained(
    device="cuda",
    speaker_strength=2.0,
    flow_cfg_rate=0.6,  # Slightly more content-focused
)

model_balanced.set_target_voice("audio/Barack Obama.mp3")
output_balanced = model_balanced.generate(audio="input.wav")
sf.write("output_balanced.wav", output_balanced.squeeze().cpu().numpy(), 24000)
print("✓ Generated with balanced settings")

# ========================================
# OPTIONAL: Gentle post-processing
# ========================================
# The new safe post-processing only does gentle RMS matching
# No artifacts, safe to enable

model_with_post = ChatterboxVC.from_pretrained(
    device="cuda",
    speaker_strength=1.8,
    enable_postprocessing=True,  # Now safe!
    postprocessing_strength=0.5,  # Gentle energy matching
)

model_with_post.set_target_voice("audio/Barack Obama.mp3")
output_with_post = model_with_post.generate(audio="input.wav")
sf.write("output_with_gentle_post.wav", output_with_post.squeeze().cpu().numpy(), 24000)
print("✓ Generated with gentle post-processing")

# ========================================
# COMPARISON TEST
# ========================================
print("\n" + "="*60)
print("RECOMMENDED SETTINGS FOR BEST TARGET SIMILARITY:")
print("="*60)
print("speaker_strength: 1.5-2.5 (start with 1.8)")
print("flow_cfg_rate: 0.6-0.7")
print("prune_tokens: 8-12")
print("enable_preprocessing: False (causes word loss)")
print("enable_postprocessing: True with strength 0.3-0.5 (safe now)")
print("\nTry these combinations:")
print("  • Conservative: speaker_strength=1.5, flow_cfg_rate=0.7")
print("  • Balanced: speaker_strength=1.8, flow_cfg_rate=0.65")
print("  • Aggressive: speaker_strength=2.5, flow_cfg_rate=0.6")
print("="*60)
