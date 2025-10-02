# Advanced Voice Conversion Processing

This module adds sophisticated pre and post-processing to improve target speaker similarity in voice conversion tasks.

## Overview

The voice conversion pipeline now includes:

```
Source Audio → [PRE-PROCESSING] → Tokenization → S3Gen Inference → [POST-PROCESSING] → Output Audio
```

### Pre-Processing (Applied to Source)
Neutralizes speaker-specific characteristics from source audio:
1. **Spectral Preemphasis** - Flattens spectral envelope to reduce timbre cues
2. **Pitch Normalization** - Shifts F0 toward target speaker's median
3. **Spectral Whitening** - Reduces speaker-specific resonance patterns
4. **Dynamics Normalization** - Equalizes energy distribution
5. **High-pass Filtering** - Removes low-frequency voice characteristics

### Post-Processing (Applied to Output)
Enhances target speaker characteristics in generated audio:
1. **Prosody Alignment** - Adjusts F0 contour to match target patterns
2. **Formant Shifting** - Modifies vocal tract resonances (VTLN-like)
3. **Spectral Envelope Matching** - Fine-tunes overall timbre profile

## Quick Start

### Basic Usage

```python
from chatterbox.vc import ChatterboxVC
import torchaudio as ta

model = ChatterboxVC.from_pretrained(
    device="cuda",
    enable_preprocessing=True,      # Enable pre-processing
    enable_postprocessing=True,     # Enable post-processing
    preprocessing_strength=0.7,     # How aggressively to neutralize source
    postprocessing_strength=0.8,    # How strongly to enhance target
)

wav = model.generate(
    audio="source.wav",
    target_voice_path="target.wav",
)

ta.save("output.wav", wav, model.sr)
```

### Per-Call Overrides

```python
# Override processing for specific generations
wav = model.generate(
    audio="source.wav",
    target_voice_path="target.wav",
    preprocessing_strength=0.9,      # Stronger source removal
    postprocessing_strength=0.95,    # Stronger target enhancement
)

# Disable processing for one call
wav_baseline = model.generate(
    audio="source.wav",
    target_voice_path="target.wav",
    enable_preprocessing=False,
    enable_postprocessing=False,
)
```

## Parameter Guide

### Preprocessing Strength (0.0 - 1.0)

- **0.0**: No pre-processing (preserve all source characteristics)
- **0.3-0.5**: Subtle neutralization (minimal source identity removal)
- **0.6-0.7**: Balanced approach (recommended starting point)
- **0.8-0.9**: Aggressive neutralization (strong source removal)
- **1.0**: Maximum neutralization (may affect clarity)

**When to adjust:**
- Source voice very dominant → Increase (0.8-0.9)
- Losing linguistic clarity → Decrease (0.4-0.6)
- Source and target very different → Increase (0.7-0.9)

### Postprocessing Strength (0.0 - 1.0)

- **0.0**: No post-processing (accept model output as-is)
- **0.4-0.6**: Gentle enhancement (subtle target push)
- **0.7-0.8**: Standard enhancement (recommended)
- **0.9-1.0**: Aggressive enhancement (strong target forcing)

**When to adjust:**
- Output still sounds like source → Increase (0.9-1.0)
- Artifacts or unnatural sound → Decrease (0.5-0.7)
- Target has very distinctive voice → Increase (0.8-0.9)

## Tuning Workflow

### Problem: Output is 50/50 source and target

**Solution 1: Increase both strengths**
```python
model = ChatterboxVC.from_pretrained(
    device="cuda",
    preprocessing_strength=0.85,
    postprocessing_strength=0.90,
)
```

**Solution 2: Combine with higher speaker_strength**
```python
model = ChatterboxVC.from_pretrained(
    device="cuda",
    preprocessing_strength=0.8,
    postprocessing_strength=0.85,
    speaker_strength=1.3,        # Scale speaker embedding
    flow_cfg_rate=1.0,           # Stronger guidance
)
```

### Problem: Output has artifacts or sounds robotic

**Solution: Reduce processing strength**
```python
model = ChatterboxVC.from_pretrained(
    device="cuda",
    preprocessing_strength=0.5,
    postprocessing_strength=0.6,
)
```

### Problem: Lost clarity or words are distorted

**Solution: Disable pre-processing, keep post-processing**
```python
model = ChatterboxVC.from_pretrained(
    device="cuda",
    enable_preprocessing=False,      # Turn off source neutralization
    enable_postprocessing=True,      # Keep target enhancement
    postprocessing_strength=0.75,
)
```

## Advanced Techniques

### Systematic Grid Search

```python
import itertools
import torchaudio as ta

# Define parameter grid
pre_strengths = [0.5, 0.7, 0.9]
post_strengths = [0.6, 0.8, 0.95]

model = ChatterboxVC.from_pretrained(device="cuda")

# Test all combinations
for i, (pre, post) in enumerate(itertools.product(pre_strengths, post_strengths)):
    print(f"Testing: pre={pre}, post={post}")
    
    wav = model.generate(
        audio="source.wav",
        target_voice_path="target.wav",
        preprocessing_strength=pre,
        postprocessing_strength=post,
    )
    
    ta.save(f"output_pre{pre}_post{post}.wav", wav, model.sr)
```

### Combined with Other Parameters

```python
# Maximum target similarity configuration
wav = model.generate(
    audio="source.wav",
    target_voice_path="target.wav",
    # Processing
    preprocessing_strength=0.85,
    postprocessing_strength=0.90,
    # Core VC parameters
    speaker_strength=1.4,
    flow_cfg_rate=1.1,
    # Optional pitch matching
    pitch_match=True,
    pitch_tolerance=0.3,
)
```

### Debugging Processing Effects

```python
# Generate with and without processing
wav_processed = model.generate(
    audio="source.wav",
    target_voice_path="target.wav",
    enable_preprocessing=True,
    enable_postprocessing=True,
)

wav_baseline = model.generate(
    audio="source.wav",
    target_voice_path="target.wav",
    enable_preprocessing=False,
    enable_postprocessing=False,
)

# Compare results
ta.save("processed.wav", wav_processed, model.sr)
ta.save("baseline.wav", wav_baseline, model.sr)
```

## Technical Details

### Pre-Processing Pipeline

1. **Spectral Preemphasis**
   - First-order high-pass filter with coefficient α = 0.95 × strength
   - Reduces spectral tilt and low-frequency emphasis

2. **Pitch Normalization**
   - Extracts F0 using PYIN algorithm
   - Computes semitone shift to target median F0
   - Applies librosa pitch shifting with kaiser_best resampling

3. **Spectral Whitening**
   - STFT-based magnitude normalization
   - Divides by smoothed spectral envelope
   - Preserves overall energy while reducing timbre

4. **High-pass Filter**
   - 4th-order Butterworth filter at 80 Hz
   - Removes speaker-specific chest resonance

### Post-Processing Pipeline

1. **Prosody Alignment**
   - Compares median F0 of output vs. target
   - Applies pitch shift to align prosodic centers
   - Strength parameter controls shift magnitude

2. **Formant Shifting (VTLN-like)**
   - Estimates vocal tract characteristics via LPC
   - Computes frequency warping factor
   - Applies bilinear frequency transformation

3. **Spectral Envelope Matching**
   - Computes mel-spectrogram envelopes
   - Calculates spectral difference
   - Applies correction via magnitude modification

## Performance Considerations

- **Pre-processing**: ~0.5-1.0 seconds overhead per generation
- **Post-processing**: ~1.0-2.0 seconds overhead per generation
- **Memory**: Minimal additional memory usage (~100MB)
- **Quality**: Best results with high-quality, clean audio inputs

## Examples

See `example_vc_enhanced.py` for a complete working example with comparison outputs.

## Troubleshooting

**Q: Pre-processing makes audio sound muffled**
A: Reduce preprocessing_strength to 0.5 or lower

**Q: Post-processing creates artifacts**
A: Reduce postprocessing_strength to 0.6 or disable formant shifting by setting strength to 0.5

**Q: No improvement over baseline**
A: Try increasing both strengths to 0.9+ and combining with higher speaker_strength

**Q: Output loses word clarity**
A: Disable pre-processing, use only post-processing

**Q: Works great but too slow**
A: Disable pre-processing if it's not helping, post-processing has bigger impact

## API Reference

### ChatterboxVC Constructor Parameters

- `enable_preprocessing` (bool): Enable pre-processing pipeline (default: True)
- `enable_postprocessing` (bool): Enable post-processing pipeline (default: True)
- `preprocessing_strength` (float): Pre-processing aggressiveness, 0.0-1.0 (default: 0.7)
- `postprocessing_strength` (float): Post-processing aggressiveness, 0.0-1.0 (default: 0.8)

### ChatterboxVC.generate() Override Parameters

All constructor parameters can be overridden per generation:

```python
wav = model.generate(
    audio="...",
    target_voice_path="...",
    enable_preprocessing=True/False,
    enable_postprocessing=True/False,
    preprocessing_strength=0.0-1.0,
    postprocessing_strength=0.0-1.0,
)
```

## Citation

If you use this processing pipeline in your research, please cite:

```
@software{chatterbox_enhanced_vc,
  title={Advanced Pre and Post-Processing for Voice Conversion},
  author={Resemble AI},
  year={2025},
  url={https://github.com/resemble-ai/chatterbox}
}
```
