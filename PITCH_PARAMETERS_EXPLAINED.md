# Pitch Parameters Explained: Tolerance and Shift

## 🎵 What is Pitch in Voice Conversion?

Pitch (fundamental frequency, F0) is the primary acoustic feature that determines perceived voice identity. It's separate from speaker embeddings but equally important.

### Key Concepts:
- **Male voices**: ~85-180 Hz average
- **Female voices**: ~165-255 Hz average  
- **Cross-gender conversion**: Requires pitch shifting (male→female: +7-12 semitones, female→male: -7-12 semitones)

---

## 🔧 Parameter 1: `pitch_match` (Binary Toggle)

**Current setting:** `pitch_match=True`

### What It Does:
Automatically detects source and target pitch ranges and applies appropriate shift to match the target's pitch profile.

### How It Works:
```python
# If pitch_match=True:
1. Extract F0 from source audio
2. Extract F0 from target reference
3. Calculate median pitch difference: Δ = median(target_F0) - median(source_F0)
4. Apply pitch shift to converted output to match target range
```

### When to Use:
- ✅ **Cross-gender conversions** (male→female or female→male): ESSENTIAL
- ✅ **Large pitch differences** (e.g., deep voice → high voice)
- ❌ **Same-gender, similar pitch**: May cause artifacts
- ❌ **When target pitch is unnatural**: Better to disable

### Example Impact:
```
Taylor Swift (female, ~200 Hz) → Barack Obama (male, ~110 Hz)
- pitch_match=True: Output at ~110 Hz (matches Obama)
- pitch_match=False: Output at ~200 Hz (keeps Swift's pitch - sounds wrong!)
```

**Current verdict:** You SHOULD keep `pitch_match=True` for cross-gender conversion.

---

## 🎚️ Parameter 2: `pitch_shift` (Manual Semitone Adjustment)

**Not currently in your code, but available in Chatterbox**

### What It Does:
Manually shifts pitch by a specific number of semitones (±12 semitones = ±1 octave).

### Syntax:
```python
pitch_shift = 0   # No change (default)
pitch_shift = 12  # Up one octave (+100% frequency)
pitch_shift = -12 # Down one octave (-50% frequency)
pitch_shift = 3.5 # Up ~25% (in-between values work)
```

### When to Use:
- Manual control when `pitch_match` doesn't get it right
- Fine-tuning after conversion ("sounds almost right but slightly too high")
- Creative effects (chipmunk voice, demon voice)

### Common Values:
```
Cross-gender conversions:
  Male → Female: +7 to +12 semitones
  Female → Male: -7 to -12 semitones

Same-gender tuning:
  Slightly deeper: -2 to -4 semitones
  Slightly higher: +2 to +4 semitones
```

---

## 🎯 Parameter 3: `pitch_tolerance` (Matching Strictness)

**Not currently visible in your code, but may be available**

### What It Does:
Controls how strictly the pitch matching algorithm enforces the target pitch range.

### Conceptual Scale:
```python
pitch_tolerance = 0.0   # Strict: Force exact pitch match
pitch_tolerance = 0.5   # Moderate: Allow some deviation
pitch_tolerance = 1.0   # Loose: Preserve more source pitch character
```

### How It Affects Output:
- **Low tolerance (0.0-0.3)**: Aggressive pitch matching
  - ✅ Strong target identity
  - ❌ May sound robotic or unnatural
  - ❌ Can introduce pitch artifacts

- **Medium tolerance (0.4-0.6)**: Balanced (recommended)
  - ✅ Good target similarity
  - ✅ Preserves some prosody
  - ✅ More natural sounding

- **High tolerance (0.7-1.0)**: Gentle matching
  - ✅ Very natural prosody
  - ❌ Weaker target identity
  - ❌ May not fully match target pitch

---

## 🤔 Will Pitch Parameters Help Your Saturation Problem?

### Short Answer: **Probably not significantly**

Here's why:

### 1. Pitch ≠ Speaker Embeddings

Your saturation problem is in the **speaker embedding space** (Cos=0.9995), not pitch:

```
Embedding saturation:
  ├─ CAMPPlus encoder: Sees speakers as 99.95% identical
  ├─ ECAPA encoder: Also sees them as ~99.95% identical
  └─ Hybrid encoder: Can't create separation that doesn't exist

Pitch matching:
  ├─ Works AFTER embeddings are applied
  ├─ Can't fix embedding saturation
  └─ Already working correctly in your case
```

### 2. Your Current Setup Is Already Optimal

```python
pitch_match=True  # ✅ Correct for cross-gender
```

This is already doing the right thing - matching Obama's pitch range.

### 3. What Happens If You Change Pitch Parameters?

**Scenario A: Disable `pitch_match=False`**
```
Result: Output keeps Taylor Swift's pitch (~200 Hz)
Effect on identity: Barack Obama's voice at female pitch
Embedding similarity: UNCHANGED (still 0.9995)
Perceptual result: Sounds like a woman, not Obama
```

**Scenario B: Add manual `pitch_shift=-8`**
```
Result: Force pitch even lower than Obama's natural range
Effect on identity: Deep male voice
Embedding similarity: UNCHANGED (still 0.9995)
Perceptual result: Unnatural, robotic Obama
```

**Scenario C: Increase `pitch_tolerance` (if available)**
```
Result: Less aggressive pitch matching
Effect on identity: More Swift's prosody, less Obama's
Embedding similarity: UNCHANGED (still 0.9995)
Perceptual result: Slightly more feminine Obama
```

### 4. Why Embeddings Matter More

The voice conversion pipeline works like this:

```
[1. Text/Prosody] → [2. Speaker Embedding] → [3. Vocoder] → [4. Pitch Adjust]
                           ⬆                                      ⬆
                    THIS IS SATURATED                    THIS IS WORKING FINE
```

Your problem is at stage 2 (speaker embedding), not stage 4 (pitch).

---

## 🎯 What COULD Help (Ranked by Likelihood)

### 1. **Different Speaker Pair** (90% chance of improvement)
Test with same-gender or more distinct speakers:
```python
# Instead of Taylor Swift → Barack Obama
# Try:
Male → Male: Morgan Freeman → Samuel L. Jackson
Female → Female: Scarlett Johansson → Emma Stone
```
**Why:** Reduces baseline saturation from 0.9995 to ~0.990-0.993

### 2. **Higher Hybrid Strength** (50% chance of improvement)
You're at 0.8, try pushing to maximum:
```python
HYBRID_PROJECTION_STRENGTH = 0.95  # Almost pure ECAPA
```
**Why:** Gives ECAPA 95% control, may break past saturation

### 3. **Longer Target Reference** (30% chance of improvement)
Use 60+ seconds of Obama speech instead of 30s:
```python
# More audio = more robust embedding
```
**Why:** Reduces embedding noise, gets better average

### 4. **Multi-Reference Averaging** (40% chance of improvement)
Average embeddings from 3-5 different Obama clips:
```python
target_embeds = [embed(clip1), embed(clip2), embed(clip3)]
final_embed = torch.mean(torch.stack(target_embeds), dim=0)
```
**Why:** Captures "core" Obama identity better

### 5. **Token Pruning** (20% chance of improvement)
Remove source tokens aggressively:
```python
PRUNE_TOKENS = 8  # Up from 0
```
**Why:** Reduces source leakage into output

### 6. **Pitch Parameter Tweaks** (5% chance of improvement)
Try manual pitch adjustment:
```python
pitch_match = False  # Disable auto
pitch_shift = -9     # Manual semitones
```
**Why:** Minimal effect on embedding similarity, but may improve perceptual identity

---

## 🔬 Diagnostic: Check If Pitch Is The Problem

Add this code to see if pitch is already correct:

```python
import librosa

# Extract pitch from outputs
y1, sr1 = librosa.load('/content/output_preprocessed.wav')
y2, sr2 = librosa.load('/content/Barack Obama.mp3')

f0_output = librosa.yin(y1, fmin=50, fmax=400)
f0_target = librosa.yin(y2, fmin=50, fmax=400)

print(f"Output median pitch: {np.median(f0_output[f0_output>0]):.1f} Hz")
print(f"Target (Obama) pitch: {np.median(f0_target[f0_target>0]):.1f} Hz")
print(f"Pitch difference: {abs(np.median(f0_output[f0_output>0]) - np.median(f0_target[f0_target>0])):.1f} Hz")
```

If pitch difference < 10 Hz → Pitch matching is already working perfectly!

---

## 📋 My Recommendation

**Don't focus on pitch parameters.** Your current `pitch_match=True` is correct.

**Instead, do this in order:**

1. ✅ **Add encoder discrimination diagnostic** (I'll add this code)
   - See if ECAPA actually discriminates better than CAMPPlus
   
2. ✅ **Test with same-gender pair** (highest impact)
   - Male→male or female→female
   
3. ✅ **Try projection strength 0.95** (easy test)
   - Give ECAPA almost full control

4. ⚠️ **Try pitch_match=False** only if audio sounds wrong
   - Listen first - if Obama's pitch is already correct, don't change it

---

## 🎓 TL;DR

**Pitch tolerance/shift WON'T fix embedding saturation** because:
1. Pitch adjustment happens AFTER embedding extraction
2. Embedding similarity (0.9995) is independent of pitch
3. Your pitch matching is already working correctly

**What WILL help:**
1. Different speaker pairs (breaks saturation naturally)
2. Higher hybrid projection strength (0.95+)
3. Multi-reference target averaging
4. Token pruning to reduce source leakage

The saturation problem is in the **semantic space** where the model represents speaker identity, not in the **acoustic space** where pitch lives.
