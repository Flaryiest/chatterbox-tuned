"""Colab Voice Conversion Demo with Hybrid Encoder.

This script:
1. Loads a pretrained ChatterboxVC model.
2. Injects a Hybrid Encoder (ECAPA-TDNN + CAMPPlus) to fix embedding saturation.
3. Applies optional preprocessing to remove source speaker characteristics:
   - Spectral Whitening: Removes source timbre/formant character
   - Dynamic Range Compression: Flattens emotional dynamics
   - Energy Envelope Transfer: Imposes target speaker's energy patterns
4. Converts audio into target speaker voice with configurable parameters.
5. Applies optional postprocessing to enhance target similarity:
   - Spectral Morphing: Morphs output spectrum toward target characteristics
6. Computes objective speaker similarity metrics.

DEPENDENCIES: 
- scipy (for signal processing): pip install scipy
- speechbrain (for hybrid encoder): pip install speechbrain
"""

import torch
import soundfile as sf
import librosa
import numpy as np
from IPython.display import Audio, display
import time
from datetime import datetime

# Preprocessing dependencies (install with: pip install scipy pyworld)
try:
    import scipy.ndimage
    from scipy.signal import medfilt
    from scipy.interpolate import interp1d
    PREPROCESSING_AVAILABLE = True
except ImportError:
    print("WARNING: scipy not installed. Legacy preprocessing will be disabled.")
    print("Install with: pip install scipy")
    PREPROCESSING_AVAILABLE = False

# Advanced preprocessing dependencies
try:
    import pyworld as pw
    PYWORLD_AVAILABLE = True
    print("✅ PyWorld available - Advanced formant-based processing enabled")
except ImportError:
    PYWORLD_AVAILABLE = False
    print("⚠️  PyWorld not installed. Advanced preprocessing disabled.")
    print("Install with: pip install pyworld")
    print("Falling back to legacy preprocessing (less effective for cross-gender)")

from chatterbox.vc import ChatterboxVC
from chatterbox.models.voice_encoder import VoiceEncoder

# Try to import hybrid encoder (requires speechbrain)
try:
    from chatterbox.models.hybrid_voice_encoder import HybridVoiceEncoder, HybridCAMPPlusEncoder
    HYBRID_ENCODER_AVAILABLE = True
except ImportError:
    HYBRID_ENCODER_AVAILABLE = False
    HybridVoiceEncoder = None  # Define as None for isinstance checks
    HybridCAMPPlusEncoder = None

# ---------------- Installation (Run first in Colab) ----------------
# !pip install scipy

# ---------------- User Config ----------------
SOURCE_AUDIO = "/content/TaylorSwiftShort.wav"
TARGET_VOICE_PATH = "/content/Barack Obama.mp3"

# PREPROCESSING (Choose strategy)
ENABLE_PREPROCESSING = True  # Enable advanced formant-based preprocessing
PREPROCESSING_STRATEGY = "formant_shift"  # Options: "formant_shift", "source_filter", "combined", "legacy"
GENDER_SHIFT = "female_to_male"  # Options: "female_to_male", "male_to_female"
FORMANT_STRENGTH = 0.78  # 0.80-0.90 (lower=more aggressive for f→m) - 22% shift for stronger effect
NEUTRALIZE_VOCAL_TRACT = False  # Disable - may cause nasal quality

# POSTPROCESSING (Choose strategy)
ENABLE_POSTPROCESSING = False  # DISABLED: Degrades CAMPPlus gain from 0.296 → 0.051 (Δ-0.245!)
POSTPROCESSING_STRATEGY = "spectral_transfer"  # Options: "spectral_transfer", "formant_shift", "combined", "legacy"
TIMBRE_STRENGTH = 0.4  # 0.5-0.9 (how much target timbre to apply) - GENTLE to avoid over-correction
POST_FORMANT_SHIFT = None  # Options: None, "neutral_to_male", "neutral_to_female"
POST_FORMANT_STRENGTH = 0.90  # If using post formant shift

# EXTERNAL PITCH SHIFT (Applied AFTER VC to avoid quality loss)
ENABLE_EXTERNAL_PITCH_SHIFT = True  # Use PyWorld to shift pitch after VC
TARGET_PITCH_HZ = 111  # Target F0 in Hz (Barack Obama's pitch)
# Set to None to calculate from target audio automatically

# VOICE CONVERSION PARAMETERS (MINIMAL INTERVENTION - prioritize quality)
FLOW_CFG_RATE = 0.85       # Strong CFG for maximum target influence (90% Obama)
SPEAKER_STRENGTH = 1.6     # Very high to force target characteristics
PRUNE_TOKENS = 0           # NO pruning - preserve all content
ENABLE_PITCH_MATCH = False # DISABLED - Causes quality/word loss for extreme shifts
PITCH_TOLERANCE = 0.5      # (not used)
MAX_PITCH_SHIFT = 5.0      # (not used)

# HYBRID VOICE ENCODER (Fixes Embedding Saturation)
USE_HYBRID_ENCODER = True  # Use ECAPA-TDNN + CAMPPlus hybrid encoder
HYBRID_PROJECTION_STRENGTH = 0.85  # Higher strength for stronger target push (was 0.70)

# MULTI-REFERENCE TARGET (Improves embedding robustness)
USE_MULTI_REFERENCE = False  # Average embeddings from multiple target clips
MULTI_REFERENCE_PATHS = [
    "/content/Barack Obama.mp3",
    # Add more paths here if available
    # "/content/Obama2.mp3",
    # "/content/Obama3.mp3",
]

# EMBEDDING EXTRAPOLATION (Push beyond target)
USE_EMBEDDING_EXTRAPOLATION = False  # Experimental: extrapolate past target
EXTRAPOLATION_STRENGTH = 1.3  # How far to push (1.0=no extrapolation, 1.5=push 50% beyond)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ============================================================================
# PREPROCESSING FUNCTIONS WITH LOGGING
# ============================================================================

def log_step(message):
    """Log with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")

# ============================================================================
# ADVANCED PREPROCESSING FUNCTIONS (Formant-Based Processing)
# ============================================================================

def warp_spectral_envelope(sp, sr, alpha):
    """
    Frequency warping of spectral envelope (formant shifting).
    
    Args:
        sp: Spectral envelope (T, freq_bins)
        sr: Sample rate
        alpha: Warping factor (<1: compress, >1: expand)
    
    Returns:
        Warped spectral envelope
    """
    n_freqs = sp.shape[1]
    freqs = np.linspace(0, sr/2, n_freqs)
    warped_freqs = freqs * alpha
    
    # Clip to valid range
    warped_freqs = np.clip(warped_freqs, 0, sr/2)
    
    # Interpolate for each time frame
    sp_warped = np.zeros_like(sp)
    for t in range(sp.shape[0]):
        interpolator = interp1d(freqs, sp[t], kind='linear', 
                                fill_value='extrapolate')
        sp_warped[t] = interpolator(warped_freqs)
    
    return sp_warped


def formant_shift_preprocessing(audio, sr, gender_shift='female_to_male', 
                                  shift_strength=0.85):
    """
    Shift formants to neutralize or pre-adapt to target gender.
    
    This is THE most effective technique for cross-gender timbre issues.
    
    Args:
        audio: Source audio (numpy array)
        sr: Sample rate
        gender_shift: 'female_to_male' or 'male_to_female'
        shift_strength: Warping factor (0.8-0.9 for f→m, 1.1-1.2 for m→f)
    
    Returns:
        Modified audio with shifted formants
    """
    if not PYWORLD_AVAILABLE:
        log_step("⚠️  Formant shifting requires pyworld. Returning original audio.")
        return audio
    
    log_step(f"🔧 Applying formant shifting ({gender_shift})...")
    
    # Convert to float64 for pyworld
    audio = audio.astype(np.float64)
    
    # Extract F0, spectral envelope, and aperiodicity
    f0, timeaxis = pw.dio(audio, sr)
    f0 = pw.stonemask(audio, f0, timeaxis, sr)
    sp = pw.cheaptrick(audio, f0, timeaxis, sr)
    ap = pw.d4c(audio, f0, timeaxis, sr)
    
    # Determine warping factor
    if gender_shift == 'female_to_male':
        alpha = shift_strength  # Default 0.85 (compress formants down)
        log_step(f"   Shifting female formants DOWN by {(1-alpha)*100:.1f}%")
    elif gender_shift == 'male_to_female':
        alpha = 1.0 / shift_strength  # e.g., 1/0.85 = 1.176 (expand up)
        log_step(f"   Shifting male formants UP by {(alpha-1)*100:.1f}%")
    else:
        alpha = 1.0  # No shift
    
    # Warp spectral envelope (formant shifting)
    sp_warped = warp_spectral_envelope(sp, sr, alpha)
    
    # Resynthesize with original F0 but warped formants
    modified = pw.synthesize(f0, sp_warped, ap, sr)
    
    log_step(f"✅ Formant shifting complete")
    return modified.astype(np.float32)


def source_filter_neutralization(audio, sr, smoothing_sigma=8):
    """
    Decompose into source (glottal) and filter (vocal tract).
    Smooth the filter to neutralize source speaker timbre.
    
    Args:
        audio: Source audio
        sr: Sample rate
        smoothing_sigma: Gaussian smoothing strength (higher = more neutral)
    
    Returns:
        Audio with neutralized vocal tract characteristics
    """
    if not PYWORLD_AVAILABLE:
        log_step("⚠️  Source-filter decomposition requires pyworld.")
        return audio
    
    log_step(f"🔧 Neutralizing source vocal tract (sigma={smoothing_sigma})...")
    
    audio_f64 = audio.astype(np.float64)
    
    # Extract components
    f0, timeaxis = pw.dio(audio_f64, sr)
    f0 = pw.stonemask(audio_f64, f0, timeaxis, sr)
    sp = pw.cheaptrick(audio_f64, f0, timeaxis, sr)
    ap = pw.d4c(audio_f64, f0, timeaxis, sr)
    
    # Smooth spectral envelope to reduce speaker-specific peaks
    from scipy.ndimage import gaussian_filter1d
    sp_smoothed = gaussian_filter1d(sp, sigma=smoothing_sigma, axis=1)
    
    # Resynthesize
    neutral_audio = pw.synthesize(f0, sp_smoothed, ap, sr)
    
    log_step(f"✅ Vocal tract neutralization complete")
    return neutral_audio.astype(np.float32)


def combined_preprocessing_pipeline(audio, sr, gender_shift='female_to_male',
                                     formant_strength=0.85, neutralize_vocal_tract=True):
    """
    Apply multiple preprocessing techniques in sequence.
    
    Recommended for difficult cross-gender conversions.
    """
    log_step("🔧 COMBINED PREPROCESSING PIPELINE")
    
    # Step 1: Formant shifting (most important)
    audio = formant_shift_preprocessing(audio, sr, gender_shift, formant_strength)
    
    # Step 2: Vocal tract neutralization (optional, more aggressive)
    if neutralize_vocal_tract:
        audio = source_filter_neutralization(audio, sr, smoothing_sigma=6)
    
    return audio


def adaptive_spectral_transfer(output_audio, target_audio, sr, 
                                timbre_strength=0.7, preserve_dynamics=True):
    """
    Transfer target's spectral envelope to output while preserving phonetic content.
    
    Much better than simple spectral morphing.
    
    Args:
        output_audio: Converted audio
        target_audio: Target speaker reference
        sr: Sample rate
        timbre_strength: How much target timbre to apply (0.0-1.0)
        preserve_dynamics: Keep output's dynamics, only transfer timbre
    
    Returns:
        Audio with target-like timbre
    """
    if not PYWORLD_AVAILABLE:
        log_step("⚠️  Spectral transfer requires pyworld.")
        return output_audio
    
    log_step(f"🔧 Applying adaptive spectral transfer (strength={timbre_strength})...")
    
    # Match lengths
    min_len = min(len(output_audio), len(target_audio))
    output_audio = output_audio[:min_len]
    target_audio = target_audio[:min_len]
    
    # Extract features
    out_f0, out_time = pw.dio(output_audio.astype(np.float64), sr)
    out_f0 = pw.stonemask(output_audio.astype(np.float64), out_f0, out_time, sr)
    out_sp = pw.cheaptrick(output_audio.astype(np.float64), out_f0, out_time, sr)
    out_ap = pw.d4c(output_audio.astype(np.float64), out_f0, out_time, sr)
    
    tgt_f0, tgt_time = pw.dio(target_audio.astype(np.float64), sr)
    tgt_f0 = pw.stonemask(target_audio.astype(np.float64), tgt_f0, tgt_time, sr)
    tgt_sp = pw.cheaptrick(target_audio.astype(np.float64), tgt_f0, tgt_time, sr)
    
    # Compute target's average spectral envelope (timbre template)
    tgt_sp_avg = np.median(tgt_sp, axis=0, keepdims=True)
    
    # Transfer timbre frame-by-frame
    out_sp_transferred = np.zeros_like(out_sp)
    for t in range(out_sp.shape[0]):
        if preserve_dynamics:
            # Keep output's spectral fine structure, apply target's envelope
            out_magnitude = np.sqrt(np.mean(out_sp[t]**2))
            tgt_magnitude = np.sqrt(np.mean(tgt_sp_avg[0]**2))
            
            # Normalize both to unit energy
            out_normalized = out_sp[t] / (out_magnitude + 1e-8)
            tgt_normalized = tgt_sp_avg[0] / (tgt_magnitude + 1e-8)
            
            # Blend normalized envelopes
            blended_normalized = (
                (1 - timbre_strength) * out_normalized + 
                timbre_strength * tgt_normalized
            )
            
            # Restore output's magnitude
            transferred = blended_normalized * out_magnitude
        else:
            # Direct blending (may lose some content)
            transferred = (
                (1 - timbre_strength) * out_sp[t] + 
                timbre_strength * tgt_sp_avg[0]
            )
        
        out_sp_transferred[t] = transferred
    
    # Resynthesize
    result = pw.synthesize(out_f0, out_sp_transferred, out_ap, sr)
    
    log_step(f"✅ Spectral transfer complete")
    return result.astype(np.float32)


def formant_shift_postprocessing(output_audio, sr, gender_shift='neutral_to_male',
                                   shift_strength=0.90):
    """
    Apply formant shifting to OUTPUT to enhance target gender.
    
    Use this if output still sounds too much like source gender.
    
    Args:
        output_audio: Converted audio
        sr: Sample rate
        gender_shift: 'neutral_to_male' or 'neutral_to_female'
        shift_strength: How much to shift (0.85-0.95 typical)
    """
    if not PYWORLD_AVAILABLE:
        return output_audio
    
    log_step(f"🔧 Post-processing formant shift ({gender_shift})...")
    
    if gender_shift == 'neutral_to_male':
        alpha = shift_strength  # Compress down
    else:
        alpha = 1.0 / shift_strength  # Expand up
    
    audio_f64 = output_audio.astype(np.float64)
    f0, timeaxis = pw.dio(audio_f64, sr)
    f0 = pw.stonemask(audio_f64, f0, timeaxis, sr)
    sp = pw.cheaptrick(audio_f64, f0, timeaxis, sr)
    ap = pw.d4c(audio_f64, f0, timeaxis, sr)
    
    sp_warped = warp_spectral_envelope(sp, sr, alpha)
    result = pw.synthesize(f0, sp_warped, ap, sr)
    
    log_step(f"✅ Post-processing formant shift complete")
    return result.astype(np.float32)


def combined_postprocessing_pipeline(output_audio, target_audio, sr,
                                      spectral_transfer_strength=0.7,
                                      formant_shift=None,
                                      formant_shift_strength=0.90):
    """
    Apply multiple postprocessing techniques.
    
    Args:
        output_audio: Converted audio
        target_audio: Target reference
        sr: Sample rate
        spectral_transfer_strength: Timbre blending (0.0-1.0)
        formant_shift: None, 'neutral_to_male', or 'neutral_to_female'
        formant_shift_strength: Formant warping amount
    """
    log_step("🔧 COMBINED POSTPROCESSING PIPELINE")
    
    # Step 1: Spectral transfer
    output_audio = adaptive_spectral_transfer(
        output_audio, target_audio, sr,
        timbre_strength=spectral_transfer_strength
    )
    
    # Step 2: Optional formant shift to enhance target gender
    if formant_shift:
        output_audio = formant_shift_postprocessing(
            output_audio, sr, formant_shift, formant_shift_strength
        )
    
    return output_audio


def analyze_spectral_characteristics(audio, sr, label="Audio"):
    """
    Print spectral analysis to diagnose timbre issues.
    """
    if not PYWORLD_AVAILABLE:
        return
    
    audio_f64 = audio.astype(np.float64)
    f0, timeaxis = pw.dio(audio_f64, sr)
    f0 = pw.stonemask(audio_f64, f0, timeaxis, sr)
    sp = pw.cheaptrick(audio_f64, f0, timeaxis, sr)
    
    # Compute statistics
    f0_valid = f0[f0 > 0]
    if len(f0_valid) > 0:
        median_f0 = np.median(f0_valid)
        mean_f0 = np.mean(f0_valid)
    else:
        median_f0 = mean_f0 = 0
    
    # Estimate formant peaks (simplified)
    avg_spectrum = np.mean(sp, axis=0)
    freqs = np.linspace(0, sr/2, len(avg_spectrum))
    
    # Find first 3 formant peaks (crude estimation)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(avg_spectrum, distance=20)
    formant_freqs = freqs[peaks][:3]
    
    print(f"\n📊 Spectral Analysis - {label}:")
    print(f"   Median F0: {median_f0:.1f} Hz")
    print(f"   Mean F0: {mean_f0:.1f} Hz")
    if len(formant_freqs) >= 3:
        print(f"   Estimated Formants: F1={formant_freqs[0]:.0f}Hz, "
              f"F2={formant_freqs[1]:.0f}Hz, F3={formant_freqs[2]:.0f}Hz")
    print(f"   Spectral Centroid: {np.mean(freqs * avg_spectrum):.0f} Hz")
    
    return median_f0  # Return median F0 for pitch shift calculation


def external_pitch_shift(audio, sr, target_f0_hz):
    """
    Apply pitch shift AFTER voice conversion to avoid quality loss.
    
    This preserves the formants (timbre) while shifting pitch.
    Use this instead of ENABLE_PITCH_MATCH during VC.
    
    Args:
        audio: Output audio from VC
        sr: Sample rate
        target_f0_hz: Target F0 in Hz (e.g., 111 for male voice)
    
    Returns:
        Pitch-shifted audio with preserved formants
    """
    if not PYWORLD_AVAILABLE:
        log_step("⚠️  External pitch shift requires PyWorld. Skipping.")
        return audio
    
    log_step(f"🔧 Applying external pitch shift to {target_f0_hz:.1f} Hz...")
    
    # Convert to float64 for PyWorld
    audio_f64 = audio.astype(np.float64)
    
    # Extract F0, spectral envelope, aperiodicity
    f0, timeaxis = pw.dio(audio_f64, sr)
    f0 = pw.stonemask(audio_f64, f0, timeaxis, sr)
    sp = pw.cheaptrick(audio_f64, f0, timeaxis, sr)
    ap = pw.d4c(audio_f64, f0, timeaxis, sr)
    
    # Calculate current median F0
    f0_valid = f0[f0 > 0]
    if len(f0_valid) == 0:
        log_step("⚠️  No voiced segments found. Skipping pitch shift.")
        return audio
    
    current_f0 = np.median(f0_valid)
    shift_ratio = target_f0_hz / current_f0
    
    log_step(f"   Current F0: {current_f0:.1f} Hz")
    log_step(f"   Target F0: {target_f0_hz:.1f} Hz")
    log_step(f"   Shift ratio: {shift_ratio:.3f}× ({12 * np.log2(shift_ratio):.1f} semitones)")
    
    # Shift F0 (preserve contours, just scale the base frequency)
    f0_shifted = np.where(f0 > 0, f0 * shift_ratio, 0)
    
    # Resynthesize with shifted F0 but PRESERVED spectral envelope (formants)
    modified = pw.synthesize(f0_shifted, sp, ap, sr)
    
    log_step(f"✅ External pitch shift complete")
    return modified.astype(np.float32)

def spectral_whitening(audio, sr, alpha=0.6):
    """Remove spectral coloration from source speaker (BALANCED mode)"""
    if not PREPROCESSING_AVAILABLE:
        log_step("Spectral whitening skipped (scipy not available)")
        return audio
    log_step("Starting spectral whitening (BALANCED mode)...")
    start = time.time()
    
    # STFT
    stft = librosa.stft(audio, n_fft=2048, hop_length=512)
    mag, phase = np.abs(stft), np.angle(stft)
    
    # Compute spectral envelope (smoothed magnitude)
    envelope = scipy.ndimage.gaussian_filter1d(mag, sigma=5, axis=0)
    
    # Whiten (reduce envelope influence) - BALANCED alpha=0.6
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

def preprocess_audio_pipeline(audio_path, target_path, sr=16000):
    """Apply preprocessing pipeline to improve target similarity.
    
    Applies:
    - Spectral whitening to remove source timbre
    - Dynamic range compression to flatten emotional dynamics
    - Energy envelope transfer to impose target characteristics
    
    Args:
        audio_path: Path to source audio file
        target_path: Path to target voice audio file
        sr: Sample rate for processing (default 16000)
    
    Returns:
        Preprocessed audio as numpy array
    """
    log_step("="*60)
    log_step("STARTING PREPROCESSING PIPELINE")
    log_step("="*60)
    
    if not PREPROCESSING_AVAILABLE:
        log_step("WARNING: Preprocessing disabled (scipy not installed)")
        log_step("Returning audio without preprocessing")
        audio, _ = librosa.load(audio_path, sr=sr)
        return audio
    
    pipeline_start = time.time()
    
    # Load audio
    log_step(f"Loading source audio: {audio_path}")
    audio, _ = librosa.load(audio_path, sr=sr)
    log_step(f"Source audio length: {len(audio)/sr:.2f}s")
    
    log_step(f"Loading target audio: {target_path}")
    target, _ = librosa.load(target_path, sr=sr)
    log_step(f"Target audio length: {len(target)/sr:.2f}s")
    
    # Apply preprocessing steps
    # 1. Spectral whitening (remove source timbre) - BALANCED mode
    audio = spectral_whitening(audio, sr, alpha=0.6)
    
    # 2. Dynamic range compression (flatten dynamics) - LIGHT
    audio = compress_dynamics(audio, sr, threshold_db=-20, ratio=3.0)
    
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
if USE_HYBRID_ENCODER and HYBRID_ENCODER_AVAILABLE and HybridCAMPPlusEncoder is not None:
    try:
        log_step("\n🔧 Injecting HYBRID ENCODER into S3Gen speaker encoder...")
        log_step("   (This replaces CAMPPlus X-Vector with ECAPA-guided version)")
        
        # Wrap the existing CAMPPlus encoder
        original_campplus = model.s3gen.speaker_encoder
        hybrid_campplus = HybridCAMPPlusEncoder(
            campplus_encoder=original_campplus,
            device=device,
            projection_strength=HYBRID_PROJECTION_STRENGTH,
        ).to(device).eval()
        
        # Optionally wrap with embedding extrapolation
        if USE_EMBEDDING_EXTRAPOLATION:
            log_step(f"\n🔬 Adding EMBEDDING EXTRAPOLATION (strength: {EXTRAPOLATION_STRENGTH:.2f})")
            
            class ExtrapolatingEncoder:
                """Wrapper that extrapolates embeddings beyond target."""
                def __init__(self, base_encoder, extrapolation_strength=1.3):
                    self.base_encoder = base_encoder
                    self.extrapolation_strength = extrapolation_strength
                    self._source_embedding = None
                    
                def inference(self, wav):
                    # Get hybrid embedding
                    embed = self.base_encoder.inference(wav)
                    
                    # Store first call as "source" reference
                    if self._source_embedding is None:
                        self._source_embedding = embed.clone()
                        return embed
                    
                    # For subsequent calls (target), extrapolate
                    # direction = target - source
                    # extrapolated = source + strength * direction
                    #              = source + strength * (target - source)
                    #              = (1 - strength) * source + strength * target
                    # But we don't have target here, so we extrapolate from source
                    # Actually, we compute: target + alpha * (target - source)
                    direction = embed - self._source_embedding
                    extrapolated = embed + (self.extrapolation_strength - 1.0) * direction
                    
                    # Normalize
                    if extrapolated.dim() == 2:
                        extrapolated = extrapolated / extrapolated.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    else:
                        extrapolated = extrapolated / extrapolated.norm().clamp(min=1e-8)
                    
                    return extrapolated
                
                def to(self, device):
                    self.base_encoder.to(device)
                    return self
                
                def eval(self):
                    self.base_encoder.eval()
                    return self
            
            hybrid_campplus = ExtrapolatingEncoder(hybrid_campplus, EXTRAPOLATION_STRENGTH)
            log_step("✅ Embedding extrapolation enabled!")
            log_step(f"   Will push embeddings {(EXTRAPOLATION_STRENGTH-1)*100:.0f}% beyond target")
        
        # Replace in the model
        model.s3gen.speaker_encoder = hybrid_campplus
        log_step("✅ Hybrid encoder injected successfully!")
        log_step(f"   Projection strength: {HYBRID_PROJECTION_STRENGTH:.2f}")
        log_step("   This should break past embedding saturation!")
    except Exception as e:
        log_step(f"⚠️  Failed to inject hybrid encoder: {e}")
        log_step(f"   Error details: {type(e).__name__}")
        log_step("   Model will use standard CAMPPlus encoder")
        import traceback
        traceback.print_exc()
elif USE_HYBRID_ENCODER and not HYBRID_ENCODER_AVAILABLE:
    log_step("\n⚠️  Hybrid encoder requested but speechbrain not installed")
    log_step("   Install with: pip install speechbrain")
    log_step("   Using standard CAMPPlus encoder")

log_step("Model loaded successfully")

# Prepare target conditioning (single reference)
if USE_MULTI_REFERENCE and len(MULTI_REFERENCE_PATHS) > 1:
    log_step(f"\n🔄 Using MULTI-REFERENCE target with {len(MULTI_REFERENCE_PATHS)} clips")
    model.set_target_voices(
        MULTI_REFERENCE_PATHS,
        mode="mean",
        robust=True
    )
    log_step(f"✅ Multi-reference target set (averaged {len(MULTI_REFERENCE_PATHS)} clips)")
else:
    model.set_target_voice(TARGET_VOICE_PATH)

# ---------------- Preprocessing Phase ----------------
log_step("\n" + "="*80)
log_step("PHASE 1: PREPROCESSING")
log_step("="*80)

# Apply preprocessing pipeline
if ENABLE_PREPROCESSING:
    source_audio, _ = librosa.load(SOURCE_AUDIO, sr=16000)
    
    if PREPROCESSING_STRATEGY == "formant_shift" and PYWORLD_AVAILABLE:
        log_step(f"Using ADVANCED formant shifting ({GENDER_SHIFT})")
        preprocessed_audio = formant_shift_preprocessing(
            audio=source_audio,
            sr=16000,
            gender_shift=GENDER_SHIFT,
            shift_strength=FORMANT_STRENGTH
        )
    elif PREPROCESSING_STRATEGY == "source_filter" and PYWORLD_AVAILABLE:
        log_step("Using source-filter neutralization")
        preprocessed_audio = source_filter_neutralization(
            audio=source_audio,
            sr=16000,
            smoothing_sigma=8
        )
    elif PREPROCESSING_STRATEGY == "combined" and PYWORLD_AVAILABLE:
        log_step(f"Using COMBINED advanced preprocessing ({GENDER_SHIFT})")
        preprocessed_audio = combined_preprocessing_pipeline(
            audio=source_audio,
            sr=16000,
            gender_shift=GENDER_SHIFT,
            formant_strength=FORMANT_STRENGTH,
            neutralize_vocal_tract=NEUTRALIZE_VOCAL_TRACT
        )
    elif PREPROCESSING_STRATEGY == "legacy" and PREPROCESSING_AVAILABLE:
        log_step("Using LEGACY preprocessing (spectral whitening + dynamics)")
        preprocessed_audio = preprocess_audio_pipeline(
            SOURCE_AUDIO,
            TARGET_VOICE_PATH,
            sr=16000
        )
    else:
        log_step("⚠️  Required preprocessing method not available, loading audio directly")
        if not PYWORLD_AVAILABLE and PREPROCESSING_STRATEGY in ["formant_shift", "source_filter", "combined"]:
            log_step("   Install PyWorld: pip install pyworld")
        preprocessed_audio = source_audio
    
    # Optional: Analyze spectral characteristics
    if PYWORLD_AVAILABLE:
        try:
            original_audio, _ = librosa.load(SOURCE_AUDIO, sr=16000)
            target_audio_16k, _ = librosa.load(TARGET_VOICE_PATH, sr=16000)
            
            analyze_spectral_characteristics(original_audio, 16000, "Source (Original)")
            analyze_spectral_characteristics(preprocessed_audio, 16000, "Source (Preprocessed)")
            analyze_spectral_characteristics(target_audio_16k, 16000, "Target")
        except Exception as e:
            log_step(f"Spectral analysis failed: {e}")
else:
    log_step("Preprocessing disabled, loading audio directly")
    preprocessed_audio, _ = librosa.load(SOURCE_AUDIO, sr=16000)

# Save preprocessed audio for inspection
preprocessed_path = "/content/preprocessed.wav"
sf.write(preprocessed_path, preprocessed_audio, 16000)
log_step(f"Preprocessed audio saved to: {preprocessed_path}")

# ---------------- Conversion (Primary) ----------------
log_step("\n" + "="*80)
log_step("PHASE 2: VOICE CONVERSION")
log_step("="*80)

conversion_start = time.time()

log_step(f"Voice conversion parameters:")
log_step(f"  - speaker_strength: {SPEAKER_STRENGTH}")
log_step(f"  - flow_cfg_rate: {FLOW_CFG_RATE}")
log_step(f"  - prune_tokens: {PRUNE_TOKENS}")
log_step(f"  - pitch_match: {ENABLE_PITCH_MATCH}")

wav = model.generate(
    audio=preprocessed_path,
    target_voice_path=TARGET_VOICE_PATH,
    speaker_strength=SPEAKER_STRENGTH,
    flow_cfg_rate=FLOW_CFG_RATE,
    prune_tokens=PRUNE_TOKENS,
    pitch_match=ENABLE_PITCH_MATCH,
    pitch_tolerance=PITCH_TOLERANCE,
    max_pitch_shift=MAX_PITCH_SHIFT,
)

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

if ENABLE_POSTPROCESSING:
    target_audio, _ = librosa.load(TARGET_VOICE_PATH, sr=model.sr)
    
    if POSTPROCESSING_STRATEGY == "spectral_transfer" and PYWORLD_AVAILABLE:
        log_step(f"Using ADVANCED spectral transfer (strength={TIMBRE_STRENGTH})")
        postprocessed_audio = adaptive_spectral_transfer(
            output_audio=output_audio,
            target_audio=target_audio,
            sr=model.sr,
            timbre_strength=TIMBRE_STRENGTH,
            preserve_dynamics=True
        )
    elif POSTPROCESSING_STRATEGY == "formant_shift" and PYWORLD_AVAILABLE and POST_FORMANT_SHIFT:
        log_step(f"Using formant shift postprocessing ({POST_FORMANT_SHIFT})")
        postprocessed_audio = formant_shift_postprocessing(
            output_audio=output_audio,
            sr=model.sr,
            gender_shift=POST_FORMANT_SHIFT,
            shift_strength=POST_FORMANT_STRENGTH
        )
    elif POSTPROCESSING_STRATEGY == "combined" and PYWORLD_AVAILABLE:
        log_step(f"Using COMBINED postprocessing")
        postprocessed_audio = combined_postprocessing_pipeline(
            output_audio=output_audio,
            target_audio=target_audio,
            sr=model.sr,
            spectral_transfer_strength=TIMBRE_STRENGTH,
            formant_shift=POST_FORMANT_SHIFT,
            formant_shift_strength=POST_FORMANT_STRENGTH
        )
    elif POSTPROCESSING_STRATEGY == "legacy" and PREPROCESSING_AVAILABLE:
        log_step("Using LEGACY spectral morphing")
        postprocessed_audio = spectral_morphing_postprocess(
            output_audio,
            target_audio,
            model.sr,
            alpha=0.6
        )
    else:
        log_step("⚠️  Required postprocessing method not available, using output as-is")
        if not PYWORLD_AVAILABLE and POSTPROCESSING_STRATEGY in ["spectral_transfer", "formant_shift", "combined"]:
            log_step("   Install PyWorld: pip install pyworld")
        postprocessed_audio = output_audio
    
    # Optional: Analyze output characteristics
    if PYWORLD_AVAILABLE:
        try:
            analyze_spectral_characteristics(
                output_audio, model.sr, "Output (Before Postprocessing)"
            )
            analyze_spectral_characteristics(
                postprocessed_audio, model.sr, "Output (After Postprocessing)"
            )
        except Exception as e:
            log_step(f"Output spectral analysis failed: {e}")
else:
    log_step("Postprocessing disabled, using output as-is")
    postprocessed_audio = output_audio

postprocessed_path = "/content/output_postprocessed.wav"
sf.write(postprocessed_path, postprocessed_audio, model.sr)
log_step(f"Postprocessed audio saved to: {postprocessed_path}")

# ---------------- External Pitch Shift (Applied AFTER VC) ----------------
if ENABLE_EXTERNAL_PITCH_SHIFT and PYWORLD_AVAILABLE:
    log_step("\n" + "="*80)
    log_step("PHASE 3B: EXTERNAL PITCH SHIFT")
    log_step("="*80)
    log_step("Applying pitch shift AFTER voice conversion (avoids quality loss)")
    
    # Get target F0 if not specified
    if TARGET_PITCH_HZ is None:
        log_step("Calculating target pitch from target audio...")
        target_audio, _ = librosa.load(TARGET_VOICE_PATH, sr=model.sr)
        target_f0 = analyze_spectral_characteristics(target_audio, model.sr, "Target")
        if target_f0 and target_f0 > 0:
            TARGET_PITCH_HZ = target_f0
            log_step(f"   Using target pitch: {TARGET_PITCH_HZ:.1f} Hz")
        else:
            log_step("⚠️  Could not detect target pitch. Skipping pitch shift.")
            TARGET_PITCH_HZ = None
    
    if TARGET_PITCH_HZ:
        # Apply external pitch shift to postprocessed audio
        pitch_shifted_audio = external_pitch_shift(
            postprocessed_audio, model.sr, TARGET_PITCH_HZ
        )
        
        # Save pitch-shifted version
        pitch_shifted_path = "/content/output_final_pitched.wav"
        sf.write(pitch_shifted_path, pitch_shifted_audio, model.sr)
        log_step(f"Pitch-shifted audio saved to: {pitch_shifted_path}")
        
        # Analyze final output
        if PYWORLD_AVAILABLE:
            try:
                final_f0 = analyze_spectral_characteristics(
                    pitch_shifted_audio, model.sr, "Final Output (Pitch Shifted)"
                )
                log_step(f"\n✅ Pitch shift successful: {final_f0:.1f} Hz (target: {TARGET_PITCH_HZ:.1f} Hz)")
            except Exception as e:
                log_step(f"Final analysis failed: {e}")
        
        # Use pitch-shifted version as final output
        final_output_path = pitch_shifted_path
    else:
        final_output_path = postprocessed_path
else:
    if ENABLE_EXTERNAL_PITCH_SHIFT and not PYWORLD_AVAILABLE:
        log_step("\n⚠️  External pitch shift requires PyWorld. Install with: pip install pyworld")
    final_output_path = postprocessed_path

log_step("\n" + "="*80)
log_step("ALL PHASES COMPLETE")
log_step("="*80)
log_step(f"Original output: {out_path}")
log_step(f"Postprocessed output: {postprocessed_path}")
if ENABLE_EXTERNAL_PITCH_SHIFT and PYWORLD_AVAILABLE and TARGET_PITCH_HZ:
    log_step(f"Final output (pitch-shifted): {final_output_path}")

display(Audio(filename=out_path, rate=model.sr))
log_step("\n[Playing: Output without postprocessing]")

if ENABLE_POSTPROCESSING:
    display(Audio(filename=postprocessed_path, rate=model.sr))
    log_step("[Playing: Output with postprocessing]")

if ENABLE_EXTERNAL_PITCH_SHIFT and PYWORLD_AVAILABLE and TARGET_PITCH_HZ:
    display(Audio(filename=final_output_path, rate=model.sr))
    log_step("[Playing: Final output with pitch shift] ⭐ RECOMMENDED")

print(f"\nSettings -> flow_cfg_rate={FLOW_CFG_RATE}, speaker_strength={SPEAKER_STRENGTH}, prune_tokens={PRUNE_TOKENS}, pitch_match={ENABLE_PITCH_MATCH}")

# ---------------- Encoder Discrimination Diagnostic ----------------
if USE_HYBRID_ENCODER and HYBRID_ENCODER_AVAILABLE and isinstance(model.s3gen.speaker_encoder, HybridCAMPPlusEncoder):
    log_step("\n" + "="*80)
    log_step("ENCODER DISCRIMINATION DIAGNOSTIC")
    log_step("="*80)
    log_step("Comparing CAMPPlus vs ECAPA discrimination on source/target pair...")
    
    try:
        import torch.nn.functional as F
        
        # Load source and target audio
        source_wav, _ = librosa.load(SOURCE_AUDIO, sr=16000)
        target_wav, _ = librosa.load(TARGET_VOICE_PATH, sr=16000)
        
        # Convert to tensors
        source_tensor = torch.from_numpy(source_wav).unsqueeze(0).to(device)
        target_tensor = torch.from_numpy(target_wav).unsqueeze(0).to(device)
        
        # Get raw CAMPPlus embeddings (192-dim)
        campplus_only = model.s3gen.speaker_encoder.campplus_encoder
        with torch.no_grad():
            src_camp = campplus_only.inference(source_tensor)
            tgt_camp = campplus_only.inference(target_tensor)
        
        # Get ECAPA embeddings (192-dim)
        with torch.no_grad():
            ecapa_src = model.s3gen.speaker_encoder.embed_ecapa(source_tensor[0], sr=16000)
            ecapa_tgt = model.s3gen.speaker_encoder.embed_ecapa(target_tensor[0], sr=16000)
        
        # Compute similarities
        camp_sim = F.cosine_similarity(src_camp, tgt_camp, dim=-1).item()
        ecapa_sim = F.cosine_similarity(ecapa_src.unsqueeze(0), ecapa_tgt.unsqueeze(0), dim=-1).item()
        
        # Compute advantage (negative = ECAPA sees MORE difference)
        ecapa_advantage = camp_sim - ecapa_sim
        
        print(f"\n📊 ENCODER DISCRIMINATION RESULTS:")
        print(f"   CAMPPlus similarity:  {camp_sim:.6f}")
        print(f"   ECAPA similarity:     {ecapa_sim:.6f}")
        print(f"   ECAPA advantage:      {ecapa_advantage:.6f}")
        
        if ecapa_advantage > 0.01:
            print(f"\n   ✅ ECAPA discriminates BETTER than CAMPPlus by {ecapa_advantage:.4f}")
            print(f"      → Hybrid encoder should provide meaningful improvement!")
        elif ecapa_advantage > 0.001:
            print(f"\n   ⚠️  ECAPA slightly better than CAMPPlus by {ecapa_advantage:.4f}")
            print(f"      → Hybrid encoder may provide small improvement")
        elif ecapa_advantage > -0.001:
            print(f"\n   ⚠️  ECAPA and CAMPPlus perform equally (difference: {abs(ecapa_advantage):.4f})")
            print(f"      → Hybrid encoder unlikely to help with this pair")
        else:
            print(f"\n   ❌ ECAPA discriminates WORSE than CAMPPlus by {abs(ecapa_advantage):.4f}")
            print(f"      → Hybrid encoder may degrade performance")
        
        # Analyze absolute discrimination
        if camp_sim > 0.999 and ecapa_sim > 0.999:
            print(f"\n   🔴 EXTREME SATURATION DETECTED (both >0.999)")
            print(f"      Both encoders see these speakers as nearly identical!")
            print(f"      RECOMMENDATION: Try different speaker pairs")
        elif camp_sim > 0.995 and ecapa_sim > 0.995:
            print(f"\n   🟡 HIGH SATURATION (both >0.995)")
            print(f"      Limited room for improvement with any encoder")
        else:
            print(f"\n   🟢 MODERATE DISCRIMINATION (both <0.995)")
            print(f"      Good test case for hybrid encoder!")
        
        # ===== NEW: Evaluate ACTUAL OUTPUT using CAMPPlus/ECAPA =====
        log_step("\n" + "="*80)
        log_step("ACTUAL OUTPUT EVALUATION (using CAMPPlus/ECAPA)")
        log_step("="*80)
        log_step("Evaluating converted audio with the ACTUAL encoders used for VC...")
        
        # Load output audio
        output_wav, _ = librosa.load(out_path, sr=16000)
        output_postproc_wav, _ = librosa.load(postprocessed_path, sr=16000)
        
        output_tensor = torch.from_numpy(output_wav).unsqueeze(0).to(device)
        output_postproc_tensor = torch.from_numpy(output_postproc_wav).unsqueeze(0).to(device)
        
        # Get CAMPPlus embeddings for outputs
        with torch.no_grad():
            out_camp = campplus_only.inference(output_tensor)
            out_post_camp = campplus_only.inference(output_postproc_tensor)
        
        # Get ECAPA embeddings for outputs
        with torch.no_grad():
            out_ecapa = model.s3gen.speaker_encoder.embed_ecapa(output_tensor[0], sr=16000)
            out_post_ecapa = model.s3gen.speaker_encoder.embed_ecapa(output_postproc_tensor[0], sr=16000)
        
        # CAMPPlus similarities
        camp_out_src = F.cosine_similarity(out_camp, src_camp, dim=-1).item()
        camp_out_tgt = F.cosine_similarity(out_camp, tgt_camp, dim=-1).item()
        camp_identity_gain = camp_out_tgt - camp_out_src
        
        camp_out_post_src = F.cosine_similarity(out_post_camp, src_camp, dim=-1).item()
        camp_out_post_tgt = F.cosine_similarity(out_post_camp, tgt_camp, dim=-1).item()
        camp_identity_gain_post = camp_out_post_tgt - camp_out_post_src
        
        # ECAPA similarities
        ecapa_out_src = F.cosine_similarity(out_ecapa.unsqueeze(0), ecapa_src.unsqueeze(0), dim=-1).item()
        ecapa_out_tgt = F.cosine_similarity(out_ecapa.unsqueeze(0), ecapa_tgt.unsqueeze(0), dim=-1).item()
        ecapa_identity_gain = ecapa_out_tgt - ecapa_out_src
        
        ecapa_out_post_src = F.cosine_similarity(out_post_ecapa.unsqueeze(0), ecapa_src.unsqueeze(0), dim=-1).item()
        ecapa_out_post_tgt = F.cosine_similarity(out_post_ecapa.unsqueeze(0), ecapa_tgt.unsqueeze(0), dim=-1).item()
        ecapa_identity_gain_post = ecapa_out_post_tgt - ecapa_out_post_src
        
        print(f"\n🎯 CAMPPlus Evaluation (192-dim, used for VC):")
        print(f"   [Preprocessed Only]")
        print(f"   Cos(output, source): {camp_out_src:.6f}")
        print(f"   Cos(output, target): {camp_out_tgt:.6f}")
        print(f"   Identity gain:       {camp_identity_gain:.6f}")
        
        print(f"\n   [Preprocessed + Postprocessed]")
        print(f"   Cos(output, source): {camp_out_post_src:.6f}")
        print(f"   Cos(output, target): {camp_out_post_tgt:.6f}")
        print(f"   Identity gain:       {camp_identity_gain_post:.6f} [Δ{camp_identity_gain_post - camp_identity_gain:+.6f}]")
        
        print(f"\n🎯 ECAPA Evaluation (192-dim, used in hybrid):")
        print(f"   [Preprocessed Only]")
        print(f"   Cos(output, source): {ecapa_out_src:.6f}")
        print(f"   Cos(output, target): {ecapa_out_tgt:.6f}")
        print(f"   Identity gain:       {ecapa_identity_gain:.6f}")
        
        print(f"\n   [Preprocessed + Postprocessed]")
        print(f"   Cos(output, source): {ecapa_out_post_src:.6f}")
        print(f"   Cos(output, target): {ecapa_out_post_tgt:.6f}")
        print(f"   Identity gain:       {ecapa_identity_gain_post:.6f} [Δ{ecapa_identity_gain_post - ecapa_identity_gain:+.6f}]")
        
        print(f"\n📈 COMPARISON:")
        print(f"   Baseline (source vs target):")
        print(f"      CAMPPlus:  {camp_sim:.6f}")
        print(f"      ECAPA:     {ecapa_sim:.6f}")
        
        print(f"\n   Identity Shift (postprocessed):")
        print(f"      CAMPPlus gain:  {camp_identity_gain_post:.6f}")
        print(f"      ECAPA gain:     {ecapa_identity_gain_post:.6f}")
        print(f"      Ratio (ECAPA/CAMPPlus): {ecapa_identity_gain_post/camp_identity_gain_post if camp_identity_gain_post != 0 else float('inf'):.2f}×")
        
        if camp_identity_gain_post > 0.05:
            print(f"\n   ✅ STRONG IDENTITY SHIFT (CAMPPlus gain > 0.05)")
            print(f"      Hybrid encoder working well!")
        elif camp_identity_gain_post > 0.02:
            print(f"\n   ⚠️  MODERATE IDENTITY SHIFT (CAMPPlus gain 0.02-0.05)")
            print(f"      Hybrid encoder providing some benefit")
        else:
            print(f"\n   ❌ WEAK IDENTITY SHIFT (CAMPPlus gain < 0.02)")
            print(f"      Consider: higher projection_strength, speaker_strength, or extrapolation")
        
    except Exception as e:
        log_step(f"⚠️  Encoder diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

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
log_step("LSTM encoder loaded: 256-dim embeddings")

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
        if isinstance(model.s3gen.speaker_encoder, HybridCAMPPlusEncoder) if HybridCAMPPlusEncoder else False:
            print(f"   ✅ Hybrid CAMPPlus encoder is ACTIVE (projection strength: {HYBRID_PROJECTION_STRENGTH:.2f})")
            print(f"      Using ECAPA-TDNN to break past saturation ceiling!")
        else:
            print(f"   💡 TIP: Hybrid encoder failed to load - check speechbrain installation")
    else:
        print("   💡 TIP: Enable USE_HYBRID_ENCODER=True and install speechbrain")

print("\n" + "="*80)
