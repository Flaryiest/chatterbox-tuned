"""
Advanced Preprocessing/Postprocessing for Cross-Gender Voice Conversion

This module provides sophisticated audio processing techniques to improve
timbre transfer, especially for challenging cross-gender conversions where
standard parameter tuning fails.

INSTALLATION:
    pip install pyworld scipy

USAGE:
    from advanced_preprocessing import (
        formant_shift_preprocessing,
        adaptive_spectral_transfer,
        source_filter_neutralization
    )
    
    # Preprocessing (before tokenization)
    processed = formant_shift_preprocessing(
        audio=source_audio,
        sr=16000,
        gender_shift='female_to_male'
    )
    
    # Postprocessing (after vocoding)
    enhanced = adaptive_spectral_transfer(
        output_audio=converted_audio,
        target_audio=target_reference,
        sr=24000,
        timbre_strength=0.7
    )
"""

import numpy as np
import librosa
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

try:
    import pyworld as pw
    PYWORLD_AVAILABLE = True
except ImportError:
    PYWORLD_AVAILABLE = False
    print("WARNING: pyworld not installed. Advanced processing disabled.")
    print("Install with: pip install pyworld")


# ============================================================================
# PREPROCESSING (Applied to Source Audio BEFORE Conversion)
# ============================================================================

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
        print("⚠️  Formant shifting requires pyworld. Returning original audio.")
        return audio
    
    print(f"\n🔧 Applying formant shifting ({gender_shift})...")
    
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
        print(f"   Shifting female formants DOWN by {(1-alpha)*100:.1f}%")
    elif gender_shift == 'male_to_female':
        alpha = 1.0 / shift_strength  # e.g., 1/0.85 = 1.176 (expand up)
        print(f"   Shifting male formants UP by {(alpha-1)*100:.1f}%")
    else:
        alpha = 1.0  # No shift
    
    # Warp spectral envelope (formant shifting)
    sp_warped = warp_spectral_envelope(sp, sr, alpha)
    
    # Resynthesize with original F0 but warped formants
    modified = pw.synthesize(f0, sp_warped, ap, sr)
    
    print(f"✅ Formant shifting complete")
    return modified.astype(np.float32)


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
        print("⚠️  Source-filter decomposition requires pyworld.")
        return audio
    
    print(f"\n🔧 Neutralizing source vocal tract (sigma={smoothing_sigma})...")
    
    audio_f64 = audio.astype(np.float64)
    
    # Extract components
    f0, timeaxis = pw.dio(audio_f64, sr)
    f0 = pw.stonemask(audio_f64, f0, timeaxis, sr)
    sp = pw.cheaptrick(audio_f64, f0, timeaxis, sr)
    ap = pw.d4c(audio_f64, f0, timeaxis, sr)
    
    # Smooth spectral envelope to reduce speaker-specific peaks
    sp_smoothed = gaussian_filter1d(sp, sigma=smoothing_sigma, axis=1)
    
    # Optional: also flatten magnitude (more aggressive)
    # sp_smoothed = sp_smoothed / np.mean(sp_smoothed, axis=1, keepdims=True)
    
    # Resynthesize
    neutral_audio = pw.synthesize(f0, sp_smoothed, ap, sr)
    
    print(f"✅ Vocal tract neutralization complete")
    return neutral_audio.astype(np.float32)


def combined_preprocessing(audio, sr, gender_shift='female_to_male',
                           formant_strength=0.85, neutralize_vocal_tract=True):
    """
    Apply multiple preprocessing techniques in sequence.
    
    Recommended for difficult cross-gender conversions.
    """
    print("\n" + "="*80)
    print("ADVANCED PREPROCESSING PIPELINE")
    print("="*80)
    
    # Step 1: Formant shifting (most important)
    audio = formant_shift_preprocessing(audio, sr, gender_shift, formant_strength)
    
    # Step 2: Vocal tract neutralization (optional, more aggressive)
    if neutralize_vocal_tract:
        audio = source_filter_neutralization(audio, sr, smoothing_sigma=6)
    
    print("="*80)
    return audio


# ============================================================================
# POSTPROCESSING (Applied to Output Audio AFTER Conversion)
# ============================================================================

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
        print("⚠️  Spectral transfer requires pyworld.")
        return output_audio
    
    print(f"\n🔧 Applying adaptive spectral transfer (strength={timbre_strength})...")
    
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
    
    print(f"✅ Spectral transfer complete")
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
    
    print(f"\n🔧 Post-processing formant shift ({gender_shift})...")
    
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
    
    print(f"✅ Post-processing formant shift complete")
    return result.astype(np.float32)


def combined_postprocessing(output_audio, target_audio, sr,
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
    print("\n" + "="*80)
    print("ADVANCED POSTPROCESSING PIPELINE")
    print("="*80)
    
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
    
    print("="*80)
    return output_audio


# ============================================================================
# DIAGNOSTIC UTILITIES
# ============================================================================

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


if __name__ == "__main__":
    print("Advanced Preprocessing Module for Voice Conversion")
    print("="*60)
    print("\nFeatures:")
    print("  ✓ Formant shifting (pre/post processing)")
    print("  ✓ Source-filter neutralization")
    print("  ✓ Adaptive spectral transfer")
    print("  ✓ Spectral analysis tools")
    print("\nInstallation: pip install pyworld scipy")
    print(f"\nPyWorld available: {PYWORLD_AVAILABLE}")
