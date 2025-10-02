"""
SAFE audio processing - gentle, non-destructive adjustments only.
No aggressive DSP that causes artifacts or lost intelligibility.
"""

import numpy as np
import librosa
import torch
from scipy import signal


class SafeAudioProcessor:
    """Minimal, safe audio adjustments that don't cause artifacts."""
    
    def __init__(self, sr=16000):
        self.sr = sr
    
    def gentle_rms_match(self, output_wav, target_wav, strength=0.3):
        """
        Gently match RMS energy to target without destroying dynamics.
        Safe and artifact-free.
        """
        if isinstance(output_wav, torch.Tensor):
            output_wav = output_wav.cpu().numpy()
        if isinstance(target_wav, torch.Tensor):
            target_wav = target_wav.cpu().numpy()
            
        output_wav = output_wav.squeeze().astype(np.float32)
        target_wav = target_wav.squeeze().astype(np.float32)
        
        # Calculate RMS
        output_rms = np.sqrt(np.mean(output_wav**2))
        target_rms = np.sqrt(np.mean(target_wav**2))
        
        if output_rms < 1e-6:
            return output_wav
        
        # Gentle scaling toward target RMS
        scale_factor = target_rms / output_rms
        scale_factor = 1.0 + (scale_factor - 1.0) * strength
        scale_factor = np.clip(scale_factor, 0.7, 1.3)  # Safety limits
        
        return output_wav * scale_factor
    
    def gentle_spectral_tilt(self, wav, strength=0.2):
        """
        Very gentle spectral tilt correction - barely noticeable, no artifacts.
        """
        if strength <= 0:
            return wav
            
        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()
        wav = wav.squeeze().astype(np.float32)
        
        # Extremely mild preemphasis (much gentler than 0.95)
        coef = 0.3 * strength  # Max 0.3 instead of 0.95
        if coef > 0:
            wav = np.append(wav[0], wav[1:] - coef * wav[:-1])
        
        return wav


def extract_median_f0(wav, sr=16000):
    """Extract median F0 from audio."""
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            wav,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            frame_length=2048
        )
        
        if f0 is None or len(f0[~np.isnan(f0)]) < 10:
            return None
        
        return np.nanmedian(f0[f0 > 0])
        
    except Exception:
        return None
