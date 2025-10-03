# Aggressive Mode Guide - For Maximum Identity Shift

## Problem Diagnosis
If you're seeing metrics like:
```
Cos(source, target): 0.9995  ← Nearly identical embeddings!
Identity gain: 0.0002        ← Tiny improvement
```

This means the **voice encoder is saturated** - it sees the source and target as basically the same speaker. Standard preprocessing won't help because it operates in a space where everything already looks similar.

## Solution: Aggressive Mode

### Quick Start
```python
# In colab.py, set these at the top:
PREPROCESSING_STRATEGY = "aggressive"
USE_AGGRESSIVE_VC_PARAMS = True
```

That's it! Run the script and it will automatically use all aggressive techniques.

---

## What Aggressive Mode Does Differently

### Preprocessing (5 Steps Instead of 3)

#### 1. **Extreme Spectral Flattening** (NEW)
- **Standard:** Gaussian smoothing of spectral envelope
- **Aggressive:** Cepstral liftering - keeps only first 20 quefrency components
- **Effect:** Removes ALL formant structure, not just some
- **Time:** ~0.5s
- **Why:** Forces complete timbre neutralization

#### 2. **Aggressive Pitch Normalization** (NEW)
- **Standard:** Optional pitch matching
- **Aggressive:** FORCES pitch to match target median F0
- **Effect:** Changes perceived gender/age
- **Time:** ~1-2s
- **Why:** Pitch is a primary identity cue - we normalize it completely

#### 3. **Formant Shifting** (NEW)
- **Standard:** Not applied
- **Aggressive:** Time-stretch + resample trick to shift formants
- **Effect:** Changes vocal tract length perception
- **Time:** ~0.5s
- **Parameters:**
  - Male target (F0 < 140Hz): shift formants DOWN (factor=0.88)
  - Female target (F0 > 180Hz): shift formants UP (factor=1.12)
  - Neutral otherwise

#### 4. **Heavy Dynamic Compression**
- **Standard:** Ratio 4:1
- **Aggressive:** Ratio 6:1
- **Effect:** Flattens emotional expression more aggressively
- **Time:** <0.1s

#### 5. **Energy Envelope Transfer**
- Same as standard mode
- Imposes target's energy patterns

**Total Preprocessing Time:** ~3-5 seconds

---

### Voice Conversion Parameters

#### Standard Parameters:
```python
speaker_strength = 1.1
prune_tokens = 0
flow_cfg_rate = 0.7
```

#### Aggressive Parameters:
```python
speaker_strength = 1.5      # +36% stronger
prune_tokens = 12           # Remove first 480ms of tokens
flow_cfg_rate = 1.8         # +157% stronger guidance
guidance_ramp = True        # Start at 0.3, ramp to 1.8
speaker_ramp = True         # Start at 0.5, ramp to 1.5
```

**Why These Work:**
- **Higher speaker_strength:** Amplifies target embedding magnitude in the decoder
- **More token pruning:** Removes source identity from beginning where prosody cues are strongest
- **Much higher CFG:** Pushes generation strongly toward target conditioning
- **Ramping:** Progressive increase prevents early-step artifacts while achieving strong final bias

---

### Postprocessing

#### Standard:
```python
alpha = 0.6  # 60% morphing toward target
```

#### Aggressive:
```python
alpha = 0.85  # 85% morphing toward target
```

More direct spectral manipulation to force output closer to target statistics.

---

## Expected Results

### Standard Mode (for similar voices):
```
Identity gain: 0.05 - 0.15
Quality: High naturalness
Use case: Same gender/age, slight accent change
```

### Aggressive Mode (for very different voices):
```
Identity gain: 0.15 - 0.40
Quality: Good (some artifacts possible)
Use case: Gender change, major age difference, cross-language
```

---

## When to Use Each Mode

### Use STANDARD Mode When:
- ✅ Source and target are same gender/age range
- ✅ Embeddings show source/target similarity < 0.99
- ✅ You get identity gain > 0.08 with standard preprocessing
- ✅ Naturalness is top priority

### Use AGGRESSIVE Mode When:
- ✅ Male → Female or Female → Male conversion
- ✅ Adult → Child or Child → Adult conversion
- ✅ Embeddings show source/target similarity > 0.995
- ✅ Standard preprocessing gives identity gain < 0.05
- ✅ Maximum identity shift is more important than perfect naturalness

---

## Parameter Tuning Guide

### If Output Still Too Close to Source:

#### Increase Preprocessing Aggressiveness:
```python
# In extreme_spectral_flattening():
lifter = 15  # Even more aggressive (was 20)

# In aggressive_pitch_normalization():
# Already at maximum - can't push further without artifacts

# In formant_shift_aggressive():
shift_factor = 1.20  # More extreme (was 1.15 for females)
shift_factor = 0.85  # More extreme (was 0.88 for males)
```

#### Increase Model Parameters:
```python
AGGRESSIVE_SPEAKER_STRENGTH = 1.8  # (was 1.5)
AGGRESSIVE_PRUNE_TOKENS = 16       # (was 12)
AGGRESSIVE_CFG_RATE = 2.2          # (was 1.8)
```

⚠️ **Warning:** Values above these may cause:
- Metallic/robotic artifacts
- Choppy speech
- Unnatural prosody
- Model instability

### If Output Has Too Many Artifacts:

#### Reduce Preprocessing Aggressiveness:
```python
# In extreme_spectral_flattening():
lifter = 25  # Less aggressive (was 20)

# In compress_dynamics():
ratio = 5.0  # Less compression (was 6.0)
```

#### Reduce Model Parameters:
```python
AGGRESSIVE_SPEAKER_STRENGTH = 1.3  # (was 1.5)
AGGRESSIVE_CFG_RATE = 1.4          # (was 1.8)
```

#### Reduce Postprocessing:
```python
AGGRESSIVE_POSTPROCESS_ALPHA = 0.7  # (was 0.85)
```

### If Speech Rate Changes Too Much:

Token pruning can affect timing. Try:
```python
AGGRESSIVE_PRUNE_TOKENS = 8  # Less pruning (was 12)
```

---

## Understanding the Metrics

### Cosine Similarity (Higher = More Similar)
- `> 0.999`: Nearly identical (preprocessing may not help much)
- `0.99 - 0.999`: Very similar (aggressive mode recommended)
- `0.95 - 0.99`: Similar (standard mode may work)
- `< 0.95`: Different (standard mode should work well)

### Identity Gain (Target - Source similarity)
- `> 0.25`: Excellent identity shift ⭐⭐⭐
- `0.15 - 0.25`: Strong identity shift ⭐⭐
- `0.08 - 0.15`: Good identity shift ⭐
- `0.05 - 0.08`: Moderate (try tuning parameters)
- `< 0.05`: Poor (try aggressive mode or different approach)

### L2 Distance (Lower = More Similar)
- Complementary to cosine similarity
- Good for detecting subtle differences
- Target: `L2(output, target) < L2(output, source)`

---

## Technical Deep Dive

### Why Extreme Spectral Flattening?

Standard spectral whitening (Gaussian smoothing) only reduces formant peaks. Cepstral liftering REMOVES the entire fine spectral structure:

```python
# Cepstrum = inverse fourier of log magnitude spectrum
cepstrum = DCT(log(magnitude))

# Low quefrency = overall spectral shape (pitch, energy)
# High quefrency = fine detail (formants, timbre)

# By zeroing high quefrency coefficients, we keep only gross shape
cepstrum[lifter:] = 0  # Remove fine detail

# Inverse transform gives ultra-flat spectrum
flattened_spectrum = exp(IDCT(cepstrum))
```

This is MUCH more aggressive than Gaussian smoothing.

### Why Formant Shifting Works

Vocal tract length determines formant frequencies:
- **Longer tract** (males) = lower formants
- **Shorter tract** (females/children) = higher formants

Time-stretching changes the temporal rate but preserves frequency content. Resampling then shifts all frequencies proportionally, changing the implied vocal tract length without affecting pitch.

### Why Token Pruning Helps

The S3 tokenizer runs at 25 tokens/sec. The first ~12 tokens (480ms) contain:
- Prosodic onset patterns (speaker-specific)
- Initial pitch/energy contours (identity cues)
- Attack characteristics (voice quality markers)

Removing these forces the model to "start fresh" with less source identity information.

### Why High CFG Works

Classifier-Free Guidance interpolates between conditional and unconditional predictions:
```python
output = unconditional + cfg_rate * (conditional - unconditional)
```

With `cfg_rate=1.8`, the model is pushed 180% toward the target-conditioned prediction, effectively "overshooting" the natural balance. This strong bias compensates for the preprocessed audio being more neutral.

---

## Troubleshooting

### "No valid pitch detected"
- Source audio is too short or too noisy
- Pitch normalization will be skipped automatically
- Try with cleaner/longer audio

### Artifacts in aggressive mode
- Normal - aggressive preprocessing trades naturalness for identity shift
- Try reducing `lifter` value (25-30 instead of 20)
- Reduce `AGGRESSIVE_CFG_RATE` to 1.5-1.6

### Still too close to source
- Check if target reference is clean and representative
- Try using multiple target references: `model.set_target_voices([path1, path2, path3])`
- Increase `AGGRESSIVE_SPEAKER_STRENGTH` to 1.7-1.8
- Consider that some phonetic content may be fundamentally preserved

### Output too robotic
- Reduce `AGGRESSIVE_CFG_RATE` to 1.2-1.4
- Increase `lifter` to 30-35 (less aggressive flattening)
- Disable formant shifting (set `formant_factor = 1.0`)

---

## Examples

### Example 1: Female → Male
```python
SOURCE_AUDIO = "/content/female_speaker.wav"
TARGET_VOICE_PATH = "/content/male_speaker.wav"
PREPROCESSING_STRATEGY = "aggressive"
USE_AGGRESSIVE_VC_PARAMS = True

# Expected results:
# - Pitch lowered from ~220Hz to ~120Hz
# - Formants shifted down (factor=0.88)
# - Identity gain: 0.20-0.35
```

### Example 2: Child → Adult
```python
SOURCE_AUDIO = "/content/child_speaker.wav"
TARGET_VOICE_PATH = "/content/adult_speaker.wav"
PREPROCESSING_STRATEGY = "aggressive"
USE_AGGRESSIVE_VC_PARAMS = True

# Expected results:
# - Pitch normalized to adult range
# - Formants shifted based on target gender
# - Identity gain: 0.15-0.30
```

### Example 3: Same Gender, Different Accent
```python
SOURCE_AUDIO = "/content/american_speaker.wav"
TARGET_VOICE_PATH = "/content/british_speaker.wav"
PREPROCESSING_STRATEGY = "standard"  # Aggressive not needed
USE_AGGRESSIVE_VC_PARAMS = False

# Expected results:
# - Accent features transferred
# - Identity gain: 0.08-0.15
# - Higher naturalness than aggressive mode
```

---

## Performance

### Time Breakdown (Aggressive Mode on GPU):
- Extreme spectral flattening: ~0.5s
- Pitch normalization: ~1-2s
- Formant shifting: ~0.5s
- Dynamic compression: <0.1s
- Energy transfer: ~0.5s
- **Preprocessing total:** ~3-5s

- Voice conversion: ~6-10s
- Postprocessing: ~0.1s
- Evaluation: ~0.2s

**Overall:** ~10-15 seconds (vs ~8-12s for standard mode)

---

## Future Improvements

Potential enhancements not yet implemented:
1. **Adaptive liftering:** Auto-adjust based on source/target distance
2. **Prosody transplantation:** Transfer target prosody curves directly
3. **Multi-reference synthesis:** Blend multiple target speakers
4. **Guided diffusion:** Inject target features at specific diffusion steps
5. **Adversarial refinement:** Post-hoc GAN to push toward target distribution

---

## Summary

**Aggressive mode is designed for extreme identity shifts** where standard preprocessing fails. It works by:
1. **Completely neutralizing** source characteristics (vs. just reducing them)
2. **Actively shifting** toward target characteristics (pitch, formants)
3. **Maximizing model bias** through extreme parameters
4. **Forcing output** toward target through aggressive postprocessing

Trade-off: Some loss of naturalness, but much stronger identity transfer.

**Rule of thumb:** If your identity gain < 0.05 or source/target similarity > 0.995, switch to aggressive mode.
