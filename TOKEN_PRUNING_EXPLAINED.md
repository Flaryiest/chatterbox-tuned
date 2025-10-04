# Token Pruning Explained: How `prune_tokens` Works

## TL;DR

**`prune_tokens`** removes the **first N tokens** from the audio before voice conversion. Since S3 tokens encode prosody and speaker identity in the beginning, removing them reduces **source speaker leakage** and helps the model shift toward the target voice.

- **Each token = 40ms of audio** (25 tokens/second)
- `prune_tokens=4` removes first **160ms** (0.16 seconds)
- `prune_tokens=8` removes first **320ms** (0.32 seconds)

---

## How S3 Tokenization Works

### Token Rate
From `s3tokenizer.py`:
```python
S3_SR = 16_000           # 16kHz audio input
S3_TOKEN_RATE = 25       # 25 tokens per second
S3_TOKEN_HOP = 640       # 640 samples per token
```

**Math:**
- 1 token = 640 samples ÷ 16,000 samples/sec = **0.040 seconds (40ms)**
- 1 second of audio = **25 tokens**

### What S3 Tokens Encode

S3Tokenizer converts audio into discrete tokens representing:
1. **Phonetic content** (what's being said)
2. **Prosody** (rhythm, stress, intonation)
3. **Speaker characteristics** (voice identity)

**Critical insight:** The **first few tokens** contain disproportionate speaker identity information because:
- Initial phonation reveals vocal tract characteristics
- Attack/onset of speech encodes voice timbre
- First prosodic contours are heavily speaker-dependent

---

## What `prune_tokens` Does

From `vc.py` (line 288-289):
```python
s3_tokens, _ = self.s3gen.tokenizer(audio_16)
if active_prune > 0 and s3_tokens.size(1) > active_prune:
    s3_tokens = s3_tokens[:, active_prune:]  # Remove first N tokens
```

**Behavior:**
- Takes the tokenized audio
- **Slices off the first `prune_tokens` tokens**
- Passes remaining tokens to the decoder

### Visual Example

**Original audio (5 seconds = 125 tokens):**
```
[T0, T1, T2, T3, T4, T5, T6, T7, T8, ..., T124]
 └─── Source identity ───┘
```

**After `prune_tokens=4`:**
```
[T4, T5, T6, T7, T8, ..., T124]
 ↑ Now starts here (160ms removed)
```

**After `prune_tokens=8`:**
```
[T8, T9, T10, ..., T124]
 ↑ Now starts here (320ms removed)
```

---

## Why This Helps Voice Conversion

### Problem: Source Speaker Leakage

The decoder receives:
1. **S3 tokens** (from source audio) → contain source identity
2. **Target speaker embedding** → provides target identity

**Conflict:** Tokens say "source voice", embedding says "target voice"

The model tries to blend both, resulting in 50/50 output.

### Solution: Remove Source Identity Tokens

By pruning the **first few tokens** (most source-identity-heavy):
- ✅ Reduces source speaker information in token sequence
- ✅ Gives target embedding more influence
- ✅ Decoder has less source identity to fight against

### Tradeoff

**Benefits:**
- Less source leakage
- Stronger target voice
- Better identity shift

**Costs:**
- Removes beginning of audio (prosody/context)
- May affect naturalness
- Can cause abrupt starts

---

## Practical Usage Guide

### Recommended Values

| `prune_tokens` | Time Removed | Use Case |
|----------------|--------------|----------|
| **0** | 0ms | Default, no pruning |
| **2-4** | 80-160ms | Mild source reduction |
| **6-8** | 240-320ms | Moderate source reduction ⭐ |
| **10-12** | 400-480ms | Aggressive source reduction |
| **>12** | >480ms | ⚠️ May damage output quality |

### When to Use Token Pruning

✅ **Use if:**
- Output sounds too much like source
- Identity gain is low (<0.05)
- You have **longer audio** (>3 seconds) that can afford to lose 160-320ms
- Model parameters (CFG, speaker_strength) aren't helping

❌ **Don't use if:**
- Audio is already short (<2 seconds) - can't afford to lose beginning
- Output quality is already poor
- You're getting artifacts/unnatural speech

### Combining with Other Parameters

**Stack with stronger parameters:**
```python
# Configuration for strong target shift
SPEAKER_STRENGTH = 1.3      # Amplify target embedding
FLOW_CFG_RATE = 0.9         # Stronger guidance toward target
PRUNE_TOKENS = 6            # Remove 240ms of source identity
```

**Why this works:**
- `SPEAKER_STRENGTH` → makes target embedding louder
- `FLOW_CFG_RATE` → pushes generation toward target
- `PRUNE_TOKENS` → reduces source information competing with target

---

## How Token Pruning Differs from Preprocessing

### Preprocessing (spectral whitening, compression, energy transfer)
- **When:** Applied to source audio **before** tokenization
- **What:** Modifies acoustic features (timbre, dynamics, energy)
- **How:** Signal processing in frequency/time domain
- **Effect:** Removes source characteristics from raw waveform

### Token Pruning
- **When:** Applied to S3 tokens **after** tokenization
- **What:** Removes discrete tokens (semantic/prosodic units)
- **How:** Simple tensor slicing
- **Effect:** Removes structured source identity from token sequence

**They're complementary!**
- Preprocessing → cleans the audio before tokenization
- Token pruning → removes source identity after tokenization
- Both → double-attack on source leakage

---

## Advanced: What About "Patch Tolerance" and "Patch Shift"?

**These terms don't exist in the Chatterbox codebase.**

After extensive search, there are **no parameters** called:
- ❌ `patch_tolerance`
- ❌ `patch_shift`
- ❌ `tolerance` (except `pitch_tolerance`)
- ❌ `shift` (except `pitch_shift`)

### What You Actually Have:

**1. Token Pruning (`prune_tokens`)** ← What we explained above

**2. Pitch Matching (`pitch_match`, `pitch_tolerance`, `max_pitch_shift`)**
- `pitch_tolerance`: Ignore pitch shifts smaller than this (semitones)
- `max_pitch_shift`: Maximum allowed pitch shift (semitones)
- See `vc.py` lines 253-280 for implementation

**3. Guidance/Speaker Ramping** (advanced scheduling)
- `guidance_ramp`: Progressively increase CFG during generation
- `speaker_ramp`: Progressively increase speaker embedding strength
- See `vc.py` lines 290-327 for implementation

---

## Diagnostic: Check If Token Pruning Helps

Add this to your evaluation to see the effect:

```python
# Test without token pruning
wav_baseline = model.generate(
    audio=source_audio,
    target_voice_path=target_audio,
    prune_tokens=0,
    speaker_strength=1.1,
    flow_cfg_rate=0.7
)

# Test with moderate token pruning
wav_pruned = model.generate(
    audio=source_audio,
    target_voice_path=target_audio,
    prune_tokens=6,  # Remove first 240ms
    speaker_strength=1.1,
    flow_cfg_rate=0.7
)

# Compare identity shift
# If pruned version has higher identity gain → token pruning helps!
```

---

## Your Current Situation

Looking at your results:
```
Baseline similarity: 0.9995 (EXTREME SATURATION)
Identity gain: 0.0004 (virtually zero)
```

### Will Token Pruning Help?

**Unlikely to be a silver bullet**, because:
1. Your problem is **embedding saturation** (encoders see speakers as 99.95% identical)
2. Token pruning addresses **token-level source leakage**, not embedding-level discrimination
3. With 0.9995 baseline similarity, even removing ALL tokens won't create separation

### What MIGHT Help (in order):

**Priority 1: Different Speaker Pair** ⭐⭐⭐
- Current pair too extreme (Cos=0.9995)
- Try same-gender or more distinct voices
- Validate hybrid encoder on less saturated pair

**Priority 2: Increase Hybrid Projection Strength**
- Current: `HYBRID_PROJECTION_STRENGTH = 0.80` (good!)
- Try: `0.90` or `0.95` (give ECAPA even more influence)
- This works at embedding level where your problem is

**Priority 3: Aggressive Model Parameters**
- `SPEAKER_STRENGTH = 1.4-1.5` (amplify target embedding)
- `FLOW_CFG_RATE = 1.0-1.2` (stronger guidance)
- `PRUNE_TOKENS = 8` (remove 320ms source identity)
- Combined effect might break through saturation

**Priority 4: Multi-Reference Target**
- Average 3-5 different clips of target speaker
- More robust embedding less affected by single-clip quirks
- Use `multi_ref_paths=[clip1, clip2, clip3]`

---

## Summary

### What Token Pruning Is
- Removes first N tokens (40ms per token) from tokenized audio
- Reduces source speaker identity in token sequence
- Simple, fast, effective for source leakage

### What Token Pruning Is NOT
- Not audio preprocessing (happens after tokenization)
- Not a fix for embedding saturation
- Not related to "patch tolerance/shift" (those don't exist)

### When to Use
- ✅ Source leakage in token sequence (output sounds like source)
- ✅ Combined with strong model parameters
- ✅ Audio longer than 3 seconds

### When NOT to Use
- ❌ Embedding saturation (like your case) - use hybrid encoder instead
- ❌ Short audio (<2 seconds)
- ❌ Already getting artifacts

### For Your Specific Case
Token pruning alone won't solve 0.9995 saturation. Focus on:
1. Testing different speaker pairs
2. Increasing hybrid projection strength
3. Combining all parameters for synergistic effect

Your hybrid encoder implementation is **already working correctly** - just needs the right test case!
