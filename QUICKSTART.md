# Quick Start Guide: Enhanced Voice Conversion

## 🚀 Getting Started in 5 Minutes

### Step 1: Update Your Import
No changes needed! Your existing code still works:
```python
from chatterbox.vc import ChatterboxVC
```

### Step 2: Create Model with Processing Enabled

**Replace this:**
```python
model = ChatterboxVC.from_pretrained(device="cuda")
```

**With this:**
```python
model = ChatterboxVC.from_pretrained(
    device="cuda",
    enable_preprocessing=True,       # NEW: Neutralize source identity
    enable_postprocessing=True,      # NEW: Enhance target identity
    preprocessing_strength=0.7,      # NEW: How much to remove source (0-1)
    postprocessing_strength=0.8,     # NEW: How much to add target (0-1)
)
```

### Step 3: Generate Voice Conversion
Use exactly as before - processing happens automatically:
```python
wav = model.generate(
    audio="source.wav",
    target_voice_path="target.wav",
)
```

**That's it!** The output should now be much closer to the target speaker.

---

## 🎯 If Output is Still 50/50 Mixed

### Try Configuration 1: Aggressive Processing
```python
model = ChatterboxVC.from_pretrained(
    device="cuda",
    preprocessing_strength=0.9,      # Strong source removal
    postprocessing_strength=0.95,    # Strong target enhancement
)
```

### Try Configuration 2: Combined with Model Params
```python
model = ChatterboxVC.from_pretrained(
    device="cuda",
    # Processing
    preprocessing_strength=0.85,
    postprocessing_strength=0.90,
    # Model parameters
    speaker_strength=1.4,            # Scale speaker embedding
    flow_cfg_rate=1.0,               # Stronger guidance
)
```

### Try Configuration 3: Post-Processing Only
```python
model = ChatterboxVC.from_pretrained(
    device="cuda",
    enable_preprocessing=False,      # Skip source neutralization
    enable_postprocessing=True,      # Only enhance target
    postprocessing_strength=0.85,
)
```

---

## 📊 Compare Results

Generate both processed and baseline versions:

```python
import torchaudio as ta

model = ChatterboxVC.from_pretrained(device="cuda")

# With processing
wav_enhanced = model.generate(
    audio="source.wav",
    target_voice_path="target.wav",
    preprocessing_strength=0.8,
    postprocessing_strength=0.9,
)
ta.save("enhanced.wav", wav_enhanced, model.sr)

# Without processing
wav_baseline = model.generate(
    audio="source.wav",
    target_voice_path="target.wav",
    enable_preprocessing=False,
    enable_postprocessing=False,
)
ta.save("baseline.wav", wav_baseline, model.sr)

# Listen and compare!
```

---

## 🔧 Tuning Guide

### Problem → Solution

| Issue | Solution |
|-------|----------|
| Still sounds like source | Increase both strengths to 0.9+ |
| Has artifacts / robotic | Decrease both strengths to 0.5-0.6 |
| Lost word clarity | Disable preprocessing, keep postprocessing |
| Target voice too subtle | Increase postprocessing_strength to 0.95 |
| Processing too slow | Disable preprocessing (post has bigger impact) |

### Strength Values Reference

| Strength | Effect | Best For |
|----------|--------|----------|
| 0.0-0.4 | Minimal | High quality audio, subtle changes |
| 0.5-0.7 | Moderate | General use, balanced results |
| 0.8-0.9 | Strong | Difficult conversions, very different speakers |
| 0.95-1.0 | Maximum | Desperate cases (may have artifacts) |

---

## 🧪 Systematic Testing

Test multiple configurations automatically:

```python
import torchaudio as ta

model = ChatterboxVC.from_pretrained(device="cuda")

# Test grid
configs = [
    {"pre": 0.5, "post": 0.7},   # Conservative
    {"pre": 0.7, "post": 0.8},   # Balanced (default)
    {"pre": 0.9, "post": 0.95},  # Aggressive
]

for i, cfg in enumerate(configs):
    wav = model.generate(
        audio="source.wav",
        target_voice_path="target.wav",
        preprocessing_strength=cfg["pre"],
        postprocessing_strength=cfg["post"],
    )
    ta.save(f"test_{i}_pre{cfg['pre']}_post{cfg['post']}.wav", wav, model.sr)
    print(f"✓ Generated test_{i}")

print("\nListen to all test_*.wav files and pick the best!")
```

---

## 📝 What Changed Under the Hood

### Before (Original Pipeline)
```
Source Audio → Tokenize → S3Gen → Output
```

### After (Enhanced Pipeline)
```
Source Audio 
  ↓
[PRE-PROCESS: Remove speaker identity]
  ↓
Tokenize → S3Gen
  ↓
[POST-PROCESS: Add target speaker]
  ↓
Output
```

### Pre-Processing Does:
1. Flattens spectral envelope (removes timbre)
2. Shifts pitch toward target
3. Whitens spectrum (reduces resonances)
4. Normalizes dynamics
5. Filters out low frequencies

### Post-Processing Does:
1. Aligns prosody to target
2. Shifts formants (vocal tract shaping)
3. Matches spectral envelope to target

---

## 💡 Pro Tips

1. **Start with defaults** (0.7 pre, 0.8 post) - they work well for most cases

2. **Use high strengths** (0.9+) when:
   - Source and target are very different (male↔female, different ages)
   - Baseline result is 50/50 mixed
   - You prioritize target similarity over quality

3. **Use low strengths** (0.5-0.6) when:
   - Audio quality is critical
   - Source and target are similar
   - You want subtle conversion

4. **Disable pre-processing if:**
   - Words become unclear
   - Losing important linguistic content
   - Post-processing alone gives good results

5. **Processing time**: Adds ~2-3 seconds per generation
   - Pre: ~0.5-1s
   - Post: ~1-2s
   - Skip pre-processing if speed critical

---

## 📚 More Information

- **Full documentation**: See `VOICE_PROCESSING_README.md`
- **Working example**: Run `example_vc_enhanced.py`
- **Technical details**: See `IMPLEMENTATION_SUMMARY.md`
- **Original colab**: Check `colab.py` for integration example

---

## ✅ Checklist

- [ ] Updated model creation to include processing parameters
- [ ] Generated test outputs with processing enabled
- [ ] Compared enhanced vs baseline results
- [ ] Tuned strengths for your specific use case
- [ ] Documented your optimal configuration

---

## 🆘 Troubleshooting

**Q: No improvement over baseline?**
A: Increase strengths to 0.9+ or try post-only processing

**Q: Audio sounds distorted?**
A: Lower strengths to 0.5-0.6

**Q: Import errors?**
A: No new dependencies needed! Uses existing librosa, scipy, torch

**Q: Too slow?**
A: Disable preprocessing, keep only postprocessing

**Q: Code changes broke my workflow?**
A: Fully backward compatible - existing code works unchanged

---

**Remember**: The goal is target speaker similarity. If processing helps, great! If not, you can disable it anytime. Experiment and find what works best for your specific audio!
