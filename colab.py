"""Colab Voice Conversion Demo with Identity Shift Metrics and Advanced Preprocessing.

This script:
1. Loads a pretrained ChatterboxVC model.
2. Applies PREPROCESSING to remove source speaker characteristics:
   
   STANDARD STRATEGY (for similar voices):
   - Spectral Whitening: Removes source timbre/formant character
   - Dynamic Range Compression: Flattens emotional dynamics
   - Energy Envelope Transfer: Imposes target speaker's energy patterns
   
   AGGRESSIVE STRATEGY (for very different voices) - NOW BALANCED:
   - Moderate Spectral Smoothing: Gentle formant structure reduction via cepstral liftering
   - Gentle Pitch Adaptation: Subtle pitch shift toward target (max ±2 semitones)
   - Subtle Formant Shifting: Very gentle gender/age characteristic adjustment
   - Moderate Dynamic Compression: Balanced emotional dynamics reduction
   - Energy Envelope Transfer: Imposes target speaker's energy patterns
   
3. Converts the preprocessed audio into target speaker voice with configurable parameters.
4. Applies POSTPROCESSING to enhance target similarity:
   - Spectral Morphing: Morphs output spectrum toward target characteristics
5. Computes objective speaker similarity metrics with comparative analysis.

WHEN TO USE AGGRESSIVE MODE:
- Source and target are very different (different gender/age/language)
- Standard preprocessing gives identity gain < 0.05
- Voice encoder embeddings show source/target similarity > 0.995
- You need maximum identity shift at the cost of some naturalness

CONFIGURATION:
Set PREPROCESSING_STRATEGY = "aggressive" and USE_AGGRESSIVE_VC_PARAMS = True
This enables BALANCED aggressive mode:
- Moderate preprocessing (gentle source characteristic reduction)
- Higher speaker_strength (1.25 vs 1.1)
- Some token pruning (6 vs 0)
- Stronger CFG guidance (1.0 vs 0.7)
- Guidance/speaker ramping for progressive conditioning
- Moderate postprocessing (0.65 vs 0.6 alpha)

NOTE: If audio becomes metallic/unrecognizable, use PREPROCESSING_STRATEGY="standard" instead.

DEPENDENCIES: scipy (for signal processing)
Install with: !pip install scipy
"""

import torch
import soundfile as sf
import librosa
import numpy as np
from IPython.display import Audio, display
import time
from datetime import datetime

# Preprocessing dependencies (install with: pip install scipy)
try:
    import scipy.ndimage
    from scipy.signal import medfilt
    PREPROCESSING_AVAILABLE = True
except ImportError:
    print("WARNING: scipy not installed. Preprocessing will be disabled.")
    print("Install with: pip install scipy")
    PREPROCESSING_AVAILABLE = False

from chatterbox.vc import ChatterboxVC
from chatterbox.models.voice_encoder import VoiceEncoder

# Try to import hybrid encoder (requires speechbrain)
try:
    from chatterbox.models.hybrid_voice_encoder import HybridVoiceEncoder
    HYBRID_ENCODER_AVAILABLE = True
except ImportError:
    HYBRID_ENCODER_AVAILABLE = False
    HybridVoiceEncoder = None  # Define as None for isinstance checks

# ---------------- Installation (Run first in Colab) ----------------
# !pip install scipy

# ---------------- User Config ----------------
# (Only the final assignment to SOURCE_AUDIO is used; remove or comment out unused examples.)
SOURCE_AUDIO = "/content/TaylorSwiftShort.wav"  # Active source
TARGET_VOICE_PATH = "/content/Barack Obama.mp3"  # Single target reference

# PREPROCESSING STRATEGY
# "none" = No preprocessing, just use model parameters
# "standard" = Original spectral methods (good for similar voices)
# "aggressive" = Balanced preprocessing with pitch/formant adaptation (for different voices)
PREPROCESSING_STRATEGY = "standard"  # Changed from "aggressive" - try this first

# VOICE CONVERSION PARAMETERS (will be overridden by aggressive mode if enabled)
FLOW_CFG_RATE =  0.70       # Strong style guidance (try 0.82–0.88 first if artifacts)
SPEAKER_STRENGTH = 1.1     # Embedding scaling (1.15–1.30 typical)
PRUNE_TOKENS = 0            # 4–8 to reduce source leakage
ENABLE_PITCH_MATCH = True  # Use pitch matching hook
PITCH_TOLERANCE = 0.6      # Ignore tiny shifts (semitones)
MAX_PITCH_SHIFT = 2.0       # Clamp extreme shifts

# AGGRESSIVE MODE OVERRIDES (used when PREPROCESSING_STRATEGY="aggressive")
# These are now more BALANCED to prevent audio degradation
USE_AGGRESSIVE_VC_PARAMS = True  # Override with stronger (but not extreme) parameters
AGGRESSIVE_SPEAKER_STRENGTH = 1.25  # Reduced from 1.5
AGGRESSIVE_PRUNE_TOKENS = 6         # Reduced from 12
AGGRESSIVE_CFG_RATE = 1.0           # Reduced from 1.8
AGGRESSIVE_POSTPROCESS_ALPHA = 0.65 # Reduced from 0.85

# ITERATIVE VOICE CONVERSION (Multiple Passes for Stronger Identity Shift)
USE_ITERATIVE_VC = True  # Enable multi-pass voice conversion
ITERATIVE_VC_PASSES = 3  # Number of conversion passes (2-4 recommended, 3 is sweet spot)
ITERATIVE_STRENGTH_RAMP = True  # Gradually increase conversion strength each pass

# HYBRID VOICE ENCODER (Fixes Embedding Saturation)
USE_HYBRID_ENCODER = True  # Use ECAPA-TDNN + LSTM hybrid encoder
HYBRID_PROJECTION_STRENGTH = 0.4  # How much ECAPA guidance to apply (0.0-1.0, recommend 0.3-0.5)

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

# ============================================================================
# ITERATIVE VOICE CONVERSION
# ============================================================================

def iterative_voice_conversion(
    model,
    source_audio,
    target_voice,
    num_iterations=3,
    base_params=None,
    enable_strength_ramp=True,
    save_intermediates=False
):
    """
    Apply voice conversion multiple times to compound the effect.
    
    Each pass removes more source identity and adds more target identity.
    Mathematically: After N passes with α=0.4 per pass:
      - 1 pass: 60% source, 40% target
      - 2 passes: 36% source, 64% target
      - 3 passes: 21.6% source, 78.4% target
      - 4 passes: 13% source, 87% target
    
    Args:
        model: ChatterboxVC model instance
        source_audio: Path to source audio OR numpy array
        target_voice: Path to target voice reference
        num_iterations: Number of conversion passes (2-4 recommended)
        base_params: Dict of generation parameters (optional)
        enable_strength_ramp: Gradually increase conversion strength each pass
        save_intermediates: Save intermediate outputs for inspection
    
    Returns:
        Final converted audio (torch.Tensor), list of intermediate outputs
    """
    import tempfile
    import os
    
    log_step("\n" + "="*80)
    log_step(f"ITERATIVE VOICE CONVERSION ({num_iterations} passes)")
    log_step("="*80)
    
    # Default parameters
    default_params = {
        'speaker_strength': 1.1,
        'flow_cfg_rate': 0.7,
        'prune_tokens': 0,
        'pitch_match': ENABLE_PITCH_MATCH,
        'pitch_tolerance': PITCH_TOLERANCE,
        'max_pitch_shift': MAX_PITCH_SHIFT,
    }
    if base_params:
        default_params.update(base_params)
    
    current_audio = source_audio
    intermediate_outputs = []
    
    for i in range(num_iterations):
        log_step(f"\n--- Pass {i+1}/{num_iterations} ---")
        
        # Gradually increase conversion strength each pass
        if enable_strength_ramp:
            # Progressive ramping: more aggressive as we go
            strength_multiplier = 1.0 + (i * 0.08)  # 1.0, 1.08, 1.16, 1.24
            cfg_multiplier = 1.0 + (i * 0.15)       # 1.0, 1.15, 1.30, 1.45
            prune_increase = min(i * 3, 10)         # 0, 3, 6, 9
            
            log_step(f"Strength multipliers: speaker={strength_multiplier:.2f}x, cfg={cfg_multiplier:.2f}x, prune=+{prune_increase}")
        else:
            strength_multiplier = 1.0
            cfg_multiplier = 1.0
            prune_increase = 0
        
        # Calculate parameters for this pass
        current_speaker_strength = default_params['speaker_strength'] * strength_multiplier
        current_cfg_rate = default_params['flow_cfg_rate'] * cfg_multiplier
        current_prune_tokens = default_params['prune_tokens'] + prune_increase
        
        log_step(f"Parameters: speaker_strength={current_speaker_strength:.2f}, "
                f"cfg_rate={current_cfg_rate:.2f}, prune_tokens={current_prune_tokens}")
        
        # Generate with adjusted parameters
        pass_start = time.time()
        wav = model.generate(
            audio=current_audio,
            target_voice_path=target_voice,
            speaker_strength=current_speaker_strength,
            flow_cfg_rate=current_cfg_rate,
            prune_tokens=current_prune_tokens,
            pitch_match=default_params['pitch_match'],
            pitch_tolerance=default_params['pitch_tolerance'],
            max_pitch_shift=default_params['max_pitch_shift'],
        )
        pass_elapsed = time.time() - pass_start
        log_step(f"Pass {i+1} completed in {pass_elapsed:.2f}s")
        
        # Store intermediate result
        intermediate_outputs.append(wav.clone())
        
        # Save intermediate result for next iteration
        if i < num_iterations - 1:  # Don't save on last iteration
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name
                sf.write(tmp_path, wav.squeeze(0).cpu().numpy(), model.sr)
                current_audio = tmp_path
                
                if save_intermediates:
                    # Also save with descriptive name
                    output_path = f"/content/iterative_pass_{i+1}.wav"
                    sf.write(output_path, wav.squeeze(0).cpu().numpy(), model.sr)
                    log_step(f"Intermediate saved: {output_path}")
                else:
                    log_step(f"Intermediate for next pass: {tmp_path}")
    
    log_step("\n" + "="*80)
    log_step(f"ITERATIVE VC COMPLETE - All {num_iterations} passes finished")
    log_step("="*80)
    
    return wav, intermediate_outputs

# ============================================================================
# PREPROCESSING FUNCTIONS WITH LOGGING
# ============================================================================

def log_step(message):
    """Log with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")

def spectral_whitening(audio, sr, alpha=0.7):
    """Remove spectral coloration from source speaker"""
    if not PREPROCESSING_AVAILABLE:
        log_step("Spectral whitening skipped (scipy not available)")
        return audio
    log_step("Starting spectral whitening...")
    start = time.time()
    
    # STFT
    stft = librosa.stft(audio, n_fft=2048, hop_length=512)
    mag, phase = np.abs(stft), np.angle(stft)
    
    # Compute spectral envelope (smoothed magnitude)
    envelope = scipy.ndimage.gaussian_filter1d(mag, sigma=5, axis=0)
    
    # Whiten (reduce envelope influence)
    whitened_mag = mag / (envelope ** alpha + 1e-8)
    
    # Reconstruct
    whitened_stft = whitened_mag * np.exp(1j * phase)
    result = librosa.istft(whitened_stft, hop_length=512)
    
    elapsed = time.time() - start
    log_step(f"Spectral whitening complete in {elapsed:.3f}s")
    return result

def compress_dynamics(audio, sr, threshold_db=-20, ratio=4.0):
    """Apply compression to flatten volume dynamics"""
    log_step("Starting dynamic range compression...")
    start = time.time()
    
    # Simple RMS-based normalization
    rms = np.sqrt(np.mean(audio**2))
    normalized = audio / (rms + 1e-8) * 0.1
    
    # Apply simple compression
    audio_db = 20 * np.log10(np.abs(normalized) + 1e-8)
    compressed = np.where(
        audio_db > threshold_db,
        np.sign(normalized) * np.power(10, (threshold_db + (audio_db - threshold_db) / ratio) / 20),
        normalized
    )
    
    # Normalize back
    compressed = compressed / (np.max(np.abs(compressed)) + 1e-8) * 0.95
    
    elapsed = time.time() - start
    log_step(f"Dynamic compression complete in {elapsed:.3f}s")
    return compressed

def transfer_energy_envelope(source_audio, target_audio, sr):
    """Replace source energy contour with target's"""
    if not PREPROCESSING_AVAILABLE:
        log_step("Energy envelope transfer skipped (scipy not available)")
        return source_audio
    log_step("Starting energy envelope transfer...")
    start = time.time()
    
    # Extract energy envelopes
    src_rms = librosa.feature.rms(y=source_audio, frame_length=2048, hop_length=512)[0]
    tgt_rms = librosa.feature.rms(y=target_audio, frame_length=2048, hop_length=512)[0]
    
    # Smooth target envelope
    tgt_envelope_smooth = medfilt(tgt_rms, kernel_size=21)
    
    # Match lengths
    if len(tgt_envelope_smooth) != len(src_rms):
        tgt_envelope_smooth = np.interp(
            np.linspace(0, 1, len(src_rms)),
            np.linspace(0, 1, len(tgt_envelope_smooth)),
            tgt_envelope_smooth
        )
    
    # Apply target envelope to source
    src_stft = librosa.stft(source_audio, hop_length=512)
    src_mag, src_phase = np.abs(src_stft), np.angle(src_stft)
    
    # Scale magnitude by envelope ratio
    envelope_ratio = tgt_envelope_smooth / (src_rms + 1e-8)
    scaled_mag = src_mag * envelope_ratio[None, :]
    
    modified_stft = scaled_mag * np.exp(1j * src_phase)
    result = librosa.istft(modified_stft, hop_length=512)
    
    elapsed = time.time() - start
    log_step(f"Energy envelope transfer complete in {elapsed:.3f}s")
    return result

def gentle_pitch_adaptation(audio, sr, target_pitch=None, max_shift=3.0):
    """Gently adapt pitch toward target (clamped to prevent artifacts)"""
    log_step("Starting gentle pitch adaptation...")
    start = time.time()
    
    try:
        # Extract pitch
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        # Get median pitch
        valid_f0 = f0[~np.isnan(f0)]
        if len(valid_f0) == 0 or target_pitch is None:
            log_step("Skipping pitch adaptation (no valid pitch or target)")
            elapsed = time.time() - start
            log_step(f"Pitch adaptation skipped in {elapsed:.3f}s")
            return audio
        
        source_median = np.median(valid_f0)
        
        # Calculate shift in semitones
        shift_semitones = 12 * np.log2(target_pitch / source_median)
        
        # Clamp to prevent extreme artifacts
        shift_semitones = np.clip(shift_semitones, -max_shift, max_shift)
        
        # Skip if shift is very small
        if abs(shift_semitones) < 0.5:
            log_step(f"Pitch shift too small ({shift_semitones:.2f} semitones), skipping")
            elapsed = time.time() - start
            log_step(f"Pitch adaptation skipped in {elapsed:.3f}s")
            return audio
        
        log_step(f"Shifting pitch by {shift_semitones:.2f} semitones (from {source_median:.1f}Hz to {target_pitch:.1f}Hz)")
        
        # Apply pitch shift
        shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=shift_semitones)
        
        elapsed = time.time() - start
        log_step(f"Gentle pitch adaptation complete in {elapsed:.3f}s")
        return shifted
    except Exception as e:
        log_step(f"Pitch adaptation failed: {e}")
        return audio

def subtle_formant_shift(audio, sr, shift_factor=1.05):
    """Subtle formant shifting (only if shift_factor differs from 1.0)"""
    if abs(shift_factor - 1.0) < 0.01:
        log_step("Skipping formant shift (factor too close to 1.0)")
        return audio
        
    log_step(f"Starting subtle formant shifting (factor={shift_factor})...")
    start = time.time()
    
    try:
        # Use time stretching + resampling trick for formant shifting
        # Stretch time
        stretched = librosa.effects.time_stretch(audio, rate=1.0/shift_factor)
        
        # Resample back to original length (changes formants)
        target_length = len(audio)
        if len(stretched) != target_length:
            formant_shifted = librosa.resample(
                stretched,
                orig_sr=sr/shift_factor,
                target_sr=sr
            )[:target_length]
        else:
            formant_shifted = stretched
        
        # Ensure same length
        if len(formant_shifted) < target_length:
            formant_shifted = np.pad(formant_shifted, (0, target_length - len(formant_shifted)))
        elif len(formant_shifted) > target_length:
            formant_shifted = formant_shifted[:target_length]
        
        elapsed = time.time() - start
        log_step(f"Subtle formant shifting complete in {elapsed:.3f}s")
        return formant_shifted
    except Exception as e:
        log_step(f"Formant shifting failed: {e}")
        return audio

def moderate_spectral_smoothing(audio, sr, alpha=0.5):
    """Moderate spectral envelope smoothing to reduce speaker characteristics"""
    log_step("Starting moderate spectral smoothing...")
    start = time.time()
    
    if not PREPROCESSING_AVAILABLE:
        log_step("Spectral smoothing skipped (scipy not available)")
        return audio
    
    # STFT
    stft = librosa.stft(audio, n_fft=2048, hop_length=512)
    mag, phase = np.abs(stft), np.angle(stft)
    
    # Get spectral envelope through cepstral smoothing (but less aggressive)
    log_mag = np.log(mag + 1e-8)
    
    try:
        # Apply DCT to get cepstrum
        from scipy.fftpack import dct, idct
        cepstrum = dct(log_mag, axis=0, norm='ortho')
        
        # Keep more quefrency components (less aggressive than before)
        lifter = 50  # Much less aggressive (was 20)
        cepstrum_smoothed = cepstrum.copy()
        cepstrum_smoothed[lifter:, :] = 0
        
        # Reconstruct smoothed envelope
        smoothed_log_mag = idct(cepstrum_smoothed, axis=0, norm='ortho')
        smoothed_mag = np.exp(smoothed_log_mag)
        
        # Blend with original (partial application)
        blended_mag = mag ** (1 - alpha) * smoothed_mag ** alpha
        
        # Normalize to preserve energy
        original_energy = np.sum(mag**2)
        blended_energy = np.sum(blended_mag**2)
        blended_mag = blended_mag * np.sqrt(original_energy / (blended_energy + 1e-8))
        
        # Reconstruct
        result_stft = blended_mag * np.exp(1j * phase)
        result = librosa.istft(result_stft, hop_length=512)
    except Exception as e:
        log_step(f"Cepstral smoothing failed ({e}), using simple whitening")
        # Fallback to simple spectral whitening
        envelope = scipy.ndimage.gaussian_filter1d(mag, sigma=5, axis=0)
        whitened_mag = mag / (envelope ** alpha + 1e-8)
        result_stft = whitened_mag * np.exp(1j * phase)
        result = librosa.istft(result_stft, hop_length=512)
    
    elapsed = time.time() - start
    log_step(f"Moderate spectral smoothing complete in {elapsed:.3f}s")
    return result

def spectral_morphing_postprocess(output_audio, target_audio, sr, alpha=0.6):
    """Morph output spectrum toward target's spectral characteristics"""
    if not PREPROCESSING_AVAILABLE:
        log_step("Spectral morphing skipped (scipy not available)")
        return output_audio
    log_step("Starting spectral morphing postprocess...")
    start = time.time()
    
    # Get spectral envelopes
    out_stft = librosa.stft(output_audio)
    
    # Match target length to output
    if len(target_audio) < len(output_audio):
        target_audio = np.pad(target_audio, (0, len(output_audio) - len(target_audio)), mode='reflect')
    else:
        target_audio = target_audio[:len(output_audio)]
    
    tgt_stft = librosa.stft(target_audio)
    
    out_mag = np.abs(out_stft)
    tgt_mag = np.abs(tgt_stft)
    out_phase = np.angle(out_stft)
    
    # Match time dimensions
    min_frames = min(out_mag.shape[1], tgt_mag.shape[1])
    out_mag = out_mag[:, :min_frames]
    tgt_mag = tgt_mag[:, :min_frames]
    out_phase = out_phase[:, :min_frames]
    
    # Get smoothed spectral envelopes
    out_env = scipy.ndimage.gaussian_filter1d(out_mag, sigma=3, axis=0)
    tgt_env = scipy.ndimage.gaussian_filter1d(tgt_mag, sigma=3, axis=0)
    
    # Morph envelope
    morphed_env = out_env ** (1 - alpha) * tgt_env ** alpha
    
    # Apply to output
    morphed_mag = out_mag * (morphed_env / (out_env + 1e-8))
    morphed_stft = morphed_mag * np.exp(1j * out_phase)
    
    result = librosa.istft(morphed_stft)
    
    elapsed = time.time() - start
    log_step(f"Spectral morphing complete in {elapsed:.3f}s")
    return result

def preprocess_audio_pipeline(audio_path, target_path, sr=16000, enable_all=True, strategy="aggressive"):
    """Apply full preprocessing pipeline to improve target similarity
    
    Args:
        audio_path: Path to source audio file
        target_path: Path to target voice audio file
        sr: Sample rate for processing (default 16000)
        enable_all: If False, only loads audio without preprocessing
        strategy: "standard" (original) or "aggressive" (new alternative approach)
    
    Returns:
        Preprocessed audio as numpy array
    """
    log_step("="*60)
    log_step(f"STARTING PREPROCESSING PIPELINE (strategy={strategy})")
    log_step("="*60)
    
    if not PREPROCESSING_AVAILABLE:
        log_step("WARNING: Preprocessing disabled (scipy not installed)")
        enable_all = False
    
    pipeline_start = time.time()
    
    # Load audio
    log_step(f"Loading source audio: {audio_path}")
    audio, _ = librosa.load(audio_path, sr=sr)
    log_step(f"Source audio length: {len(audio)/sr:.2f}s")
    
    log_step(f"Loading target audio: {target_path}")
    target, _ = librosa.load(target_path, sr=sr)
    log_step(f"Target audio length: {len(target)/sr:.2f}s")
    
    if enable_all:
        if strategy == "none":
            log_step("No preprocessing applied (strategy=none)")
            # Return audio as-is
            pass
            
        elif strategy == "aggressive":
            log_step("Using BALANCED preprocessing strategy (reduced aggressiveness)")
            
            # Get target median pitch for optional adaptation
            try:
                target_f0, _, _ = librosa.pyin(
                    target[:sr*10],  # Use first 10s
                    fmin=librosa.note_to_hz('C2'),
                    fmax=librosa.note_to_hz('C7'),
                    sr=sr
                )
                valid_target_f0 = target_f0[~np.isnan(target_f0)]
                target_median_pitch = np.median(valid_target_f0) if len(valid_target_f0) > 0 else None
                if target_median_pitch:
                    log_step(f"Target median pitch: {target_median_pitch:.1f}Hz")
            except:
                target_median_pitch = None
            
            # 1. Moderate spectral smoothing (gentle formant reduction)
            audio = moderate_spectral_smoothing(audio, sr, alpha=0.4)  # Gentle (0.4 vs 0.7)
            
            # 2. Gentle pitch adaptation (only if significant difference)
            audio = gentle_pitch_adaptation(audio, sr, target_pitch=target_median_pitch, max_shift=2.0)
            
            # 3. Very subtle formant shifting (if needed)
            if target_median_pitch:
                if target_median_pitch < 130:  # Male
                    formant_factor = 0.97  # Very subtle shift down (was 0.88)
                elif target_median_pitch > 190:  # Female
                    formant_factor = 1.03  # Very subtle shift up (was 1.12)
                else:
                    formant_factor = 1.0  # Neutral
                
                audio = subtle_formant_shift(audio, sr, shift_factor=formant_factor)
            
            # 4. Moderate dynamic compression (reduce emotional dynamics)
            audio = compress_dynamics(audio, sr, threshold_db=-20, ratio=4.5)  # More moderate (was 6.0)
            
            # 5. Energy envelope transfer (impose target characteristics)
            audio = transfer_energy_envelope(audio, target, sr)
            
        else:  # Standard strategy
            log_step("Using STANDARD preprocessing strategy")
            # 1. Spectral whitening (remove source timbre)
            audio = spectral_whitening(audio, sr, alpha=0.7)
            
            # 2. Dynamic range compression (flatten dynamics)
            audio = compress_dynamics(audio, sr, threshold_db=-20, ratio=4.0)
            
            # 3. Energy envelope transfer (impose target characteristics)
            audio = transfer_energy_envelope(audio, target, sr)
    
    pipeline_elapsed = time.time() - pipeline_start
    log_step("="*60)
    log_step(f"PREPROCESSING COMPLETE - Total time: {pipeline_elapsed:.3f}s")
    log_step("="*60)
    
    return audio

# ---------------- Load Model ----------------
log_step("Loading Chatterbox voice conversion model...")
model = ChatterboxVC.from_pretrained(
    device,
    flow_cfg_rate=FLOW_CFG_RATE,
    speaker_strength=SPEAKER_STRENGTH,
    prune_tokens=PRUNE_TOKENS,
)

# Inject hybrid encoder into model if enabled
if USE_HYBRID_ENCODER and HYBRID_ENCODER_AVAILABLE and HybridVoiceEncoder is not None:
    try:
        log_step("\n🔧 Injecting hybrid encoder into voice conversion model...")
        # Create hybrid encoder for model's use
        model_hybrid_encoder = HybridVoiceEncoder(
            lstm_encoder=VoiceEncoder().to(device).eval(),
            device=device,
            projection_strength=HYBRID_PROJECTION_STRENGTH,
        ).to(device).eval()
        # Replace model's voice encoder with hybrid version
        model.voice_encoder = model_hybrid_encoder
        log_step("✅ Model now using hybrid encoder for target voice embedding")
    except Exception as e:
        log_step(f"⚠️  Failed to inject hybrid encoder into model: {e}")
        log_step("   Model will use standard LSTM encoder")

log_step("Model loaded successfully")

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

# ---------------- Preprocessing Phase ----------------
log_step("\n" + "="*80)
log_step("PHASE 1: PREPROCESSING (AGGRESSIVE MODE)")
log_step("="*80)

# Apply preprocessing pipeline with selected strategy
preprocessed_audio = preprocess_audio_pipeline(
    SOURCE_AUDIO,
    TARGET_VOICE_PATH,
    sr=16000,
    enable_all=True,
    strategy=PREPROCESSING_STRATEGY
)

# Save preprocessed audio for inspection
preprocessed_path = "/content/preprocessed.wav"
sf.write(preprocessed_path, preprocessed_audio, 16000)
log_step(f"Preprocessed audio saved to: {preprocessed_path}")

# ---------------- Conversion (Primary) ----------------
log_step("\n" + "="*80)
log_step("PHASE 2: VOICE CONVERSION")
log_step("="*80)

conversion_start = time.time()

# Determine parameters based on preprocessing strategy
if PREPROCESSING_STRATEGY == "aggressive" and USE_AGGRESSIVE_VC_PARAMS:
    base_params = {
        'speaker_strength': AGGRESSIVE_SPEAKER_STRENGTH,
        'prune_tokens': AGGRESSIVE_PRUNE_TOKENS,
        'flow_cfg_rate': AGGRESSIVE_CFG_RATE,
    }
    log_step("Using AGGRESSIVE base parameters for VC")
    log_step(f"  - speaker_strength: {AGGRESSIVE_SPEAKER_STRENGTH} (baseline: {SPEAKER_STRENGTH})")
    log_step(f"  - prune_tokens: {AGGRESSIVE_PRUNE_TOKENS} (baseline: {PRUNE_TOKENS})")
    log_step(f"  - flow_cfg_rate: {AGGRESSIVE_CFG_RATE} (baseline: {FLOW_CFG_RATE})")
else:
    base_params = {
        'speaker_strength': SPEAKER_STRENGTH,
        'prune_tokens': PRUNE_TOKENS,
        'flow_cfg_rate': FLOW_CFG_RATE,
    }
    log_step("Using STANDARD base parameters for VC")
    log_step(f"  - speaker_strength: {SPEAKER_STRENGTH}")
    log_step(f"  - prune_tokens: {PRUNE_TOKENS}")
    log_step(f"  - flow_cfg_rate: {FLOW_CFG_RATE}")

# Apply iterative VC or single-pass VC
if USE_ITERATIVE_VC and ITERATIVE_VC_PASSES > 1:
    log_step(f"\n🔄 ITERATIVE MODE ENABLED: {ITERATIVE_VC_PASSES} passes")
    wav, intermediate_outputs = iterative_voice_conversion(
        model=model,
        source_audio=preprocessed_path,
        target_voice=TARGET_VOICE_PATH,
        num_iterations=ITERATIVE_VC_PASSES,
        base_params=base_params,
        enable_strength_ramp=ITERATIVE_STRENGTH_RAMP,
        save_intermediates=True  # Save intermediate passes for inspection
    )
else:
    log_step("\n⚡ SINGLE-PASS MODE")
    wav = model.generate(
        audio=preprocessed_path,
        target_voice_path=TARGET_VOICE_PATH,
        **base_params,
        pitch_match=ENABLE_PITCH_MATCH,
        pitch_tolerance=PITCH_TOLERANCE,
        max_pitch_shift=MAX_PITCH_SHIFT,
    )
    intermediate_outputs = []

conversion_elapsed = time.time() - conversion_start
log_step(f"\nVoice conversion completed in {conversion_elapsed:.3f}s")

out_path = "/content/output_preprocessed.wav"
sf.write(out_path, wav.squeeze(0).cpu().numpy(), model.sr)

# ---------------- Postprocessing Phase ----------------
log_step("\n" + "="*80)
log_step("PHASE 3: POSTPROCESSING")
log_step("="*80)

# Apply spectral morphing postprocess
output_audio = wav.squeeze(0).cpu().numpy()
target_audio, _ = librosa.load(TARGET_VOICE_PATH, sr=model.sr)

# Apply postprocessing with appropriate aggressiveness
if PREPROCESSING_STRATEGY == "aggressive" and USE_AGGRESSIVE_VC_PARAMS:
    alpha_value = AGGRESSIVE_POSTPROCESS_ALPHA
    log_step(f"Using AGGRESSIVE postprocessing (alpha={alpha_value})")
else:
    alpha_value = 0.6
    log_step(f"Using STANDARD postprocessing (alpha={alpha_value})")

postprocessed_audio = spectral_morphing_postprocess(
    output_audio,
    target_audio,
    model.sr,
    alpha=alpha_value
)

postprocessed_path = "/content/output_postprocessed.wav"
sf.write(postprocessed_path, postprocessed_audio, model.sr)
log_step(f"Postprocessed audio saved to: {postprocessed_path}")

log_step("\n" + "="*80)
log_step("ALL PHASES COMPLETE")
log_step("="*80)
log_step(f"Original output: {out_path}")
log_step(f"Postprocessed output: {postprocessed_path}")

# Show iterative progression if enabled
if USE_ITERATIVE_VC and ITERATIVE_VC_PASSES > 1:
    log_step(f"\n📊 ITERATIVE VC PROGRESSION ({ITERATIVE_VC_PASSES} passes):")
    for i in range(ITERATIVE_VC_PASSES):
        iter_path = f"/content/iterative_pass_{i+1}.wav"
        if i < ITERATIVE_VC_PASSES - 1:  # Intermediate passes
            log_step(f"  Pass {i+1}: {iter_path}")
        else:  # Final pass
            log_step(f"  Pass {i+1} (final): {out_path}")

display(Audio(filename=out_path, rate=model.sr))
log_step("\n[Playing: Preprocessed only (no postprocess)]")
display(Audio(filename=postprocessed_path, rate=model.sr))
log_step("[Playing: Preprocessed + Postprocessed]")

# Play intermediate iterations if available
if USE_ITERATIVE_VC and ITERATIVE_VC_PASSES > 1:
    log_step("\n[Iterative VC Progression - Listen to improvement across passes:]")
    for i in range(ITERATIVE_VC_PASSES - 1):  # Don't replay final (already played above)
        iter_path = f"/content/iterative_pass_{i+1}.wav"
        try:
            display(Audio(filename=iter_path, rate=model.sr))
            log_step(f"  ↑ Pass {i+1} of {ITERATIVE_VC_PASSES}")
        except:
            pass

print(f"\nSettings -> flow_cfg_rate={FLOW_CFG_RATE}, speaker_strength={SPEAKER_STRENGTH}, prune_tokens={PRUNE_TOKENS}, pitch_match={ENABLE_PITCH_MATCH}")

# ---------------- Identity Shift Evaluation ----------------
log_step("\n" + "="*80)
log_step("PHASE 4: IDENTITY SHIFT EVALUATION")
log_step("="*80)

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

# Load base LSTM encoder
log_step("Loading voice encoder for evaluation...")
lstm_encoder = VoiceEncoder().to(device).eval()
log_step(f"LSTM encoder loaded: {lstm_encoder.embedding_size}-dim embeddings")

# Optionally wrap with hybrid encoder
if USE_HYBRID_ENCODER and HYBRID_ENCODER_AVAILABLE:
    try:
        log_step("\n🔄 Initializing HYBRID ENCODER (ECAPA-TDNN + LSTM)...")
        voice_encoder = HybridVoiceEncoder(
            lstm_encoder=lstm_encoder,
            device=device,
            projection_strength=HYBRID_PROJECTION_STRENGTH,
        ).to(device).eval()
        log_step(f"✅ Hybrid encoder ready (projection strength: {HYBRID_PROJECTION_STRENGTH:.2f})")
        log_step("   This should break past embedding saturation!")
    except Exception as e:
        log_step(f"⚠️  Failed to load hybrid encoder: {e}")
        log_step("   Falling back to standard LSTM encoder")
        voice_encoder = lstm_encoder
else:
    if USE_HYBRID_ENCODER and not HYBRID_ENCODER_AVAILABLE:
        log_step("⚠️  Hybrid encoder requested but not available")
        log_step("   Install: pip install speechbrain")
    voice_encoder = lstm_encoder
    log_step("Using standard LSTM encoder")

log_step("Computing embeddings for source, target, and outputs...")
embed_start = time.time()

source_spk, source_partials = load_embeds_utterance(SOURCE_AUDIO, voice_encoder)
target_spk, target_partials = load_embeds_utterance(TARGET_VOICE_PATH, voice_encoder)
output_preprocessed_spk, output_preprocessed_partials = load_embeds_utterance(out_path, voice_encoder)
output_postprocessed_spk, output_postprocessed_partials = load_embeds_utterance(postprocessed_path, voice_encoder)

embed_elapsed = time.time() - embed_start
log_step(f"Embedding computation completed in {embed_elapsed:.3f}s")

# Partial-level averaged metrics (can be more discriminative)
def mean_pairwise_cos(A: torch.Tensor, B: torch.Tensor):
    # A: (m,E), B: (n,E)
    return float((A @ B.T).mean() / (torch.norm(A, dim=1).mean() * torch.norm(B, dim=1).mean()))

# Baseline metrics
sim_source_target = cosine(source_spk, target_spk)

# Preprocessed output metrics
sim_preproc_source = cosine(output_preprocessed_spk, source_spk)
sim_preproc_target = cosine(output_preprocessed_spk, target_spk)
identity_gain_preproc = sim_preproc_target - sim_preproc_source

partial_cos_preproc_target = mean_pairwise_cos(output_preprocessed_partials, target_partials)
partial_cos_preproc_source = mean_pairwise_cos(output_preprocessed_partials, source_partials)
partial_gain_preproc = partial_cos_preproc_target - partial_cos_preproc_source

spk_l2_preproc_source = l2(output_preprocessed_spk, source_spk)
spk_l2_preproc_target = l2(output_preprocessed_spk, target_spk)

# Postprocessed output metrics
sim_postproc_source = cosine(output_postprocessed_spk, source_spk)
sim_postproc_target = cosine(output_postprocessed_spk, target_spk)
identity_gain_postproc = sim_postproc_target - sim_postproc_source

partial_cos_postproc_target = mean_pairwise_cos(output_postprocessed_partials, target_partials)
partial_cos_postproc_source = mean_pairwise_cos(output_postprocessed_partials, source_partials)
partial_gain_postproc = partial_cos_postproc_target - partial_cos_postproc_source

spk_l2_postproc_source = l2(output_postprocessed_spk, source_spk)
spk_l2_postproc_target = l2(output_postprocessed_spk, target_spk)

print("\n" + "="*80)
print("IDENTITY SHIFT METRICS COMPARISON")
print("="*80)

print("\n[Baseline]")
print(f"Cos(source, target): {sim_source_target:.4f}")

print("\n[Preprocessed Only – Speaker Level]")
print(f"Cos(output, source): {sim_preproc_source:.4f}")
print(f"Cos(output, target): {sim_preproc_target:.4f}")
print(f"Identity gain (target - source): {identity_gain_preproc:.4f}")
print(f"L2(output, source): {spk_l2_preproc_source:.4f}")
print(f"L2(output, target): {spk_l2_preproc_target:.4f}")

print("\n[Preprocessed Only – Partial/Segment Level]")
print(f"Mean partial cos (out vs source): {partial_cos_preproc_source:.4f}")
print(f"Mean partial cos (out vs target): {partial_cos_preproc_target:.4f}")
print(f"Partial identity gain: {partial_gain_preproc:.4f}")

print("\n[Preprocessed + Postprocessed – Speaker Level]")
print(f"Cos(output, source): {sim_postproc_source:.4f}")
print(f"Cos(output, target): {sim_postproc_target:.4f}")
print(f"Identity gain (target - source): {identity_gain_postproc:.4f} [Δ{identity_gain_postproc - identity_gain_preproc:+.4f}]")
print(f"L2(output, source): {spk_l2_postproc_source:.4f}")
print(f"L2(output, target): {spk_l2_postproc_target:.4f}")

print("\n[Preprocessed + Postprocessed – Partial/Segment Level]")
print(f"Mean partial cos (out vs source): {partial_cos_postproc_source:.4f}")
print(f"Mean partial cos (out vs target): {partial_cos_postproc_target:.4f}")
print(f"Partial identity gain: {partial_gain_postproc:.4f} [Δ{partial_gain_postproc - partial_gain_preproc:+.4f}]")

print("\n[IMPROVEMENT SUMMARY]")
print(f"Preprocessing gain: {identity_gain_preproc:.4f}")
print(f"Preprocessing + Postprocessing gain: {identity_gain_postproc:.4f}")
print(f"Additional postprocessing benefit: {identity_gain_postproc - identity_gain_preproc:+.4f}")

if (sim_postproc_target < sim_postproc_source) or (partial_cos_postproc_target < partial_cos_postproc_source):
    print("\n⚠️  WARNING: Output still closer to source than target on at least one metric")
else:
    print("\n✅ SUCCESS: Output closer to target across primary metrics")
    if identity_gain_postproc > 0.15:
        print("   Strong identity shift achieved!")
    elif identity_gain_postproc > 0.08:
        print("   Good identity shift achieved.")
    else:
        print("   Moderate identity shift. Consider tuning preprocessing parameters.")

# Add diagnosis for embedding saturation
if sim_source_target > 0.999:
    print("\n⚠️  EMBEDDING SATURATION DETECTED")
    print(f"   Source/target similarity: {sim_source_target:.4f} (>0.999)")
    print("   The voice encoder sees these speakers as nearly identical.")
    print("   RECOMMENDATION: Preprocessing has limited effect in this case.")
    print("   - Try using a different source or target voice")
    print("   - Focus on model parameters (speaker_strength, prune_tokens, cfg_rate)")
    print("   - The small identity gain may be as good as this model can achieve")
    if USE_HYBRID_ENCODER and HYBRID_ENCODER_AVAILABLE:
        if isinstance(voice_encoder, HybridVoiceEncoder):
            print(f"   ✅ Hybrid encoder is ACTIVE (projection strength: {HYBRID_PROJECTION_STRENGTH:.2f})")
            print(f"      Using ECAPA-TDNN to break past saturation ceiling!")
        else:
            print(f"   💡 TIP: Hybrid encoder failed to load - check speechbrain installation")
    else:
        print("   💡 TIP: Enable USE_HYBRID_ENCODER=True and install speechbrain")
    if USE_ITERATIVE_VC:
        print(f"   ✅ Iterative VC is ENABLED ({ITERATIVE_VC_PASSES} passes) - compounds the effect")
    else:
        print("   💡 TIP: Enable USE_ITERATIVE_VC=True for 30-40% better results")

# Analyze iterative progression if enabled
if USE_ITERATIVE_VC and ITERATIVE_VC_PASSES > 1 and len(intermediate_outputs) > 0:
    print("\n" + "="*80)
    print("ITERATIVE VC PROGRESSION ANALYSIS")
    print("="*80)
    
    for i, iter_output in enumerate(intermediate_outputs):
        iter_audio = iter_output.squeeze(0).cpu().numpy()
        
        # Determine correct path for this iteration
        if i < ITERATIVE_VC_PASSES - 1:
            # Intermediate passes saved separately
            iter_path = f"/content/iterative_pass_{i+1}.wav"
        else:
            # Final pass is the main output
            iter_path = out_path
        
        # Compute embeddings for this iteration
        try:
            iter_spk, iter_partials = load_embeds_utterance(iter_path, voice_encoder)
            
            iter_cos_source = cosine(iter_spk, source_spk)
            iter_cos_target = cosine(iter_spk, target_spk)
            iter_gain = iter_cos_target - iter_cos_source
            
            print(f"\n[Pass {i+1}/{ITERATIVE_VC_PASSES}]")
            print(f"  Cos(output, source): {iter_cos_source:.4f}")
            print(f"  Cos(output, target): {iter_cos_target:.4f}")
            print(f"  Identity gain: {iter_gain:.4f}")
            
            if i > 0:
                # Compare to previous pass
                if i == 1:
                    prev_path = f"/content/iterative_pass_{i}.wav"
                else:
                    prev_path = f"/content/iterative_pass_{i}.wav" if i < ITERATIVE_VC_PASSES - 1 else f"/content/iterative_pass_{i}.wav"
                
                try:
                    prev_spk, _ = load_embeds_utterance(prev_path, voice_encoder)
                    prev_gain = cosine(prev_spk, target_spk) - cosine(prev_spk, source_spk)
                    improvement = iter_gain - prev_gain
                    print(f"  Improvement from Pass {i}: {improvement:+.4f}")
                except:
                    pass  # Can't compute improvement if previous pass unavailable
        except Exception as e:
            print(f"\n[Pass {i+1}] Could not compute metrics: {e}")
    
    print("\n" + "="*80)

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

