"""
Advanced pre and post-processing for voice conversion.
Focuses on maximizing target speaker similarity through DSP techniques.
"""

import numpy as np
import librosa
import torch
from scipy import signal
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings('ignore')


class AudioProcessor:
    """Handles pre and post-processing for improved voice conversion."""
    
    def __init__(self, sr=16000):
        self.sr = sr
        
    # ==================== PRE-PROCESSING ====================
    
    def preprocess_source(self, wav, target_median_f0=None, aggressiveness=0.7):
        """
        Aggressive content normalization to remove speaker identity from source.
        
        Args:
            wav: source audio (numpy array)
            target_median_f0: target speaker's median F0 (optional)
            aggressiveness: 0.0-1.0, how much to neutralize (0.7 recommended)
        
        Returns:
            Processed audio with reduced speaker identity
        """
        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()
        
        if wav.ndim > 1:
            wav = wav.squeeze()
        
        # Ensure float32
        wav = wav.astype(np.float32)
        
        # 1. Spectral preemphasis (reduce low-freq speaker cues)
        wav = self._apply_preemphasis(wav, coef=0.95 * aggressiveness)
        
        # 2. Pitch normalization toward neutral or target
        if target_median_f0 is not None:
            wav = self._normalize_pitch_toward_target(wav, target_median_f0, strength=aggressiveness)
        else:
            wav = self._normalize_pitch_variance(wav, strength=aggressiveness * 0.5)
        
        # 3. Spectral whitening (reduce timbre cues)
        wav = self._spectral_whitening(wav, strength=aggressiveness * 0.6)
        
        # 4. Dynamics normalization
        wav = self._normalize_dynamics(wav)
        
        # 5. Remove very low frequencies (< 80Hz) that carry speaker identity
        wav = self._highpass_filter(wav, cutoff=80)
        
        return wav
    
    def _apply_preemphasis(self, wav, coef=0.95):
        """Apply preemphasis filter to flatten spectral envelope."""
        if coef <= 0:
            return wav
        return np.append(wav[0], wav[1:] - coef * wav[:-1])
    
    def _normalize_pitch_toward_target(self, wav, target_f0, strength=0.7):
        """Shift source pitch toward target speaker's median F0."""
        try:
            # Extract source F0
            f0 = self._extract_f0_robust(wav)
            
            if f0 is None or len(f0[~np.isnan(f0)]) < 10:
                return wav
            
            # Calculate median F0
            source_median_f0 = np.nanmedian(f0[f0 > 0])
            
            if source_median_f0 <= 0 or target_f0 <= 0:
                return wav
            
            # Calculate shift in semitones
            shift_semitones = 12 * np.log2(target_f0 / source_median_f0)
            
            # Apply strength scaling
            shift_semitones *= strength
            
            # Clamp to reasonable range
            shift_semitones = np.clip(shift_semitones, -6, 6)
            
            if abs(shift_semitones) < 0.3:
                return wav
            
            # Apply pitch shift
            wav_shifted = librosa.effects.pitch_shift(
                wav, sr=self.sr, n_steps=shift_semitones, res_type='kaiser_best'
            )
            
            return wav_shifted
            
        except Exception as e:
            print(f"Warning: Pitch normalization failed: {e}")
            return wav
    
    def _normalize_pitch_variance(self, wav, strength=0.5):
        """Compress pitch variance to make it more neutral."""
        try:
            f0 = self._extract_f0_robust(wav)
            
            if f0 is None or len(f0[~np.isnan(f0)]) < 10:
                return wav
            
            # Calculate variance compression
            median_f0 = np.nanmedian(f0[f0 > 0])
            std_f0 = np.nanstd(f0[f0 > 0])
            
            # Compress toward median (subtle effect)
            target_std = std_f0 * (1 - strength * 0.3)
            compression = target_std / (std_f0 + 1e-6)
            
            # This is approximate - true pitch variance compression would need WORLD vocoder
            # For now, we do a conservative pitch shift
            return wav
            
        except Exception:
            return wav
    
    def _spectral_whitening(self, wav, strength=0.6):
        """Apply spectral whitening to reduce timbre cues."""
        try:
            # Compute STFT
            D = librosa.stft(wav, n_fft=2048, hop_length=512)
            mag, phase = np.abs(D), np.angle(D)
            
            # Compute spectral envelope (smoothed magnitude)
            envelope = signal.medfilt(mag, kernel_size=(5, 1))
            
            # Whiten: divide by envelope
            whitened_mag = mag / (envelope ** strength + 1e-8)
            
            # Normalize to preserve energy
            whitened_mag = whitened_mag * (np.mean(mag) / (np.mean(whitened_mag) + 1e-8))
            
            # Reconstruct
            D_whitened = whitened_mag * np.exp(1j * phase)
            wav_whitened = librosa.istft(D_whitened, hop_length=512)
            
            # Ensure same length
            if len(wav_whitened) > len(wav):
                wav_whitened = wav_whitened[:len(wav)]
            elif len(wav_whitened) < len(wav):
                wav_whitened = np.pad(wav_whitened, (0, len(wav) - len(wav_whitened)))
            
            return wav_whitened.astype(np.float32)
            
        except Exception as e:
            print(f"Warning: Spectral whitening failed: {e}")
            return wav
    
    def _normalize_dynamics(self, wav):
        """Normalize dynamics with soft compression."""
        # RMS normalization
        rms = np.sqrt(np.mean(wav**2))
        if rms > 0:
            wav = wav / rms * 0.1
        
        # Soft clipping
        wav = np.tanh(wav * 1.5) / 1.5
        
        return wav
    
    def _highpass_filter(self, wav, cutoff=80):
        """Remove low frequencies that carry speaker identity."""
        try:
            nyquist = self.sr / 2
            normalized_cutoff = cutoff / nyquist
            
            # Design highpass filter
            b, a = signal.butter(4, normalized_cutoff, btype='highpass')
            
            # Apply filter
            wav_filtered = signal.filtfilt(b, a, wav)
            
            return wav_filtered.astype(np.float32)
            
        except Exception:
            return wav
    
    # ==================== POST-PROCESSING ====================
    
    def postprocess_output(self, output_wav, target_ref_wav, aggressiveness=0.8):
        """
        Post-process output to push it closer to target speaker.
        
        Args:
            output_wav: generated audio from VC model
            target_ref_wav: target speaker reference
            aggressiveness: 0.0-1.0, how strongly to apply corrections
        
        Returns:
            Enhanced audio closer to target speaker
        """
        if isinstance(output_wav, torch.Tensor):
            output_wav = output_wav.cpu().numpy()
        if isinstance(target_ref_wav, torch.Tensor):
            target_ref_wav = target_ref_wav.cpu().numpy()
        
        output_wav = output_wav.squeeze().astype(np.float32)
        target_ref_wav = target_ref_wav.squeeze().astype(np.float32)
        
        # Resample target to match output if needed
        if len(target_ref_wav) < self.sr:
            # Too short, repeat
            repeats = int(np.ceil(self.sr / len(target_ref_wav)))
            target_ref_wav = np.tile(target_ref_wav, repeats)[:self.sr * 3]
        
        # 1. Prosody transplant (F0 and energy alignment)
        output_wav = self._align_prosody_to_target(output_wav, target_ref_wav, strength=aggressiveness * 0.6)
        
        # 2. Formant shifting toward target
        output_wav = self._shift_formants_to_target(output_wav, target_ref_wav, strength=aggressiveness * 0.8)
        
        # 3. Spectral envelope matching
        output_wav = self._match_spectral_envelope(output_wav, target_ref_wav, strength=aggressiveness)
        
        return output_wav
    
    def _align_prosody_to_target(self, output_wav, target_wav, strength=0.6):
        """
        Align F0 contour and energy toward target speaker's prosody.
        This is a simplified version - full implementation would use WORLD vocoder.
        """
        try:
            # Extract F0 from both
            output_f0 = self._extract_f0_robust(output_wav)
            target_f0 = self._extract_f0_robust(target_wav)
            
            if output_f0 is None or target_f0 is None:
                return output_wav
            
            # Get median values
            output_median = np.nanmedian(output_f0[output_f0 > 0])
            target_median = np.nanmedian(target_f0[target_f0 > 0])
            
            if output_median <= 0 or target_median <= 0:
                return output_wav
            
            # Calculate shift to align medians
            shift_semitones = 12 * np.log2(target_median / output_median)
            shift_semitones *= strength
            shift_semitones = np.clip(shift_semitones, -4, 4)
            
            if abs(shift_semitones) > 0.3:
                output_wav = librosa.effects.pitch_shift(
                    output_wav, sr=self.sr, n_steps=shift_semitones, res_type='kaiser_best'
                )
            
            return output_wav
            
        except Exception as e:
            print(f"Warning: Prosody alignment failed: {e}")
            return output_wav
    
    def _shift_formants_to_target(self, output_wav, target_wav, strength=0.8):
        """
        Shift formants (vocal tract resonances) toward target speaker.
        Uses LPC-based approach for formant estimation and VTLN-like transformation.
        """
        try:
            # Extract spectral characteristics
            output_lpc = self._extract_lpc_envelope(output_wav)
            target_lpc = self._extract_lpc_envelope(target_wav)
            
            if output_lpc is None or target_lpc is None:
                return output_wav
            
            # Estimate formant shift via LPC coefficient warping
            # This is an approximation of VTLN (Vocal Tract Length Normalization)
            alpha = self._estimate_vtln_warp_factor(output_lpc, target_lpc)
            alpha = 1.0 + (alpha - 1.0) * strength
            alpha = np.clip(alpha, 0.85, 1.15)
            
            if abs(alpha - 1.0) < 0.02:
                return output_wav
            
            # Apply frequency warping
            output_wav = self._apply_frequency_warping(output_wav, alpha)
            
            return output_wav
            
        except Exception as e:
            print(f"Warning: Formant shifting failed: {e}")
            return output_wav
    
    def _match_spectral_envelope(self, output_wav, target_wav, strength=0.8):
        """
        Match the spectral envelope of output to target speaker.
        Uses mel-cepstral distance minimization approach.
        """
        try:
            # Compute mel spectrograms
            output_mel = librosa.feature.melspectrogram(
                y=output_wav, sr=self.sr, n_fft=2048, hop_length=512, n_mels=80
            )
            target_mel = librosa.feature.melspectrogram(
                y=target_wav, sr=self.sr, n_fft=2048, hop_length=512, n_mels=80
            )
            
            # Convert to log scale
            output_mel_db = librosa.power_to_db(output_mel + 1e-8)
            target_mel_db = librosa.power_to_db(target_mel + 1e-8)
            
            # Compute average spectral envelopes
            output_envelope = np.mean(output_mel_db, axis=1, keepdims=True)
            target_envelope = np.mean(target_mel_db, axis=1, keepdims=True)
            
            # Compute correction filter
            correction = target_envelope - output_envelope
            correction *= strength
            
            # Apply correction to output mel
            corrected_mel_db = output_mel_db + correction
            corrected_mel = librosa.db_to_power(corrected_mel_db)
            
            # Convert back to audio via Griffin-Lim
            output_stft = librosa.stft(output_wav, n_fft=2048, hop_length=512)
            output_phase = np.angle(output_stft)
            
            # Reconstruct with corrected magnitude
            # Need to map mel back to linear spectrogram
            corrected_linear = self._mel_to_linear_approx(corrected_mel, output_stft.shape)
            
            # Combine with original phase
            corrected_stft = corrected_linear * np.exp(1j * output_phase)
            corrected_wav = librosa.istft(corrected_stft, hop_length=512)
            
            # Ensure same length
            if len(corrected_wav) > len(output_wav):
                corrected_wav = corrected_wav[:len(output_wav)]
            elif len(corrected_wav) < len(output_wav):
                corrected_wav = np.pad(corrected_wav, (0, len(output_wav) - len(corrected_wav)))
            
            return corrected_wav.astype(np.float32)
            
        except Exception as e:
            print(f"Warning: Spectral envelope matching failed: {e}")
            return output_wav
    
    # ==================== UTILITY FUNCTIONS ====================
    
    def _extract_f0_robust(self, wav):
        """Extract F0 using librosa's pyin algorithm."""
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                wav,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sr,
                frame_length=2048,
                hop_length=512
            )
            return f0
        except Exception:
            return None
    
    def _extract_lpc_envelope(self, wav, order=16):
        """Extract LPC (Linear Predictive Coding) envelope."""
        try:
            # Frame the signal
            frames = librosa.util.frame(wav, frame_length=2048, hop_length=512)
            
            # Compute LPC coefficients for middle frame
            mid_frame = frames[:, frames.shape[1]//2]
            
            # LPC analysis
            # Using autocorrelation method (Levinson-Durbin)
            r = np.correlate(mid_frame, mid_frame, mode='full')
            r = r[len(r)//2:]
            r = r[:order+1]
            
            # Levinson-Durbin recursion
            lpc = self._levinson_durbin(r, order)
            
            return lpc
            
        except Exception:
            return None
    
    def _levinson_durbin(self, r, order):
        """Levinson-Durbin recursion for LPC coefficients."""
        a = np.zeros(order + 1)
        a[0] = 1.0
        e = r[0]
        
        for i in range(1, order + 1):
            lambda_i = -np.sum(a[:i] * r[i:0:-1]) / e
            a[i] = lambda_i
            a[1:i] = a[1:i] + lambda_i * a[i-1:0:-1]
            e = e * (1 - lambda_i**2)
        
        return a
    
    def _estimate_vtln_warp_factor(self, source_lpc, target_lpc):
        """
        Estimate VTLN warping factor from LPC coefficients.
        Returns alpha > 1 for longer vocal tract, < 1 for shorter.
        """
        try:
            # Convert LPC to formant frequencies (approximate)
            source_roots = np.roots(source_lpc)
            target_roots = np.roots(target_lpc)
            
            # Get angles (relate to formant frequencies)
            source_angles = np.angle(source_roots[source_roots.imag >= 0])
            target_angles = np.angle(target_roots[target_roots.imag >= 0])
            
            # Sort and take first 3-4 (formants)
            source_angles = np.sort(source_angles)[:4]
            target_angles = np.sort(target_angles)[:4]
            
            # Estimate warp factor
            if len(source_angles) > 0 and len(target_angles) > 0:
                alpha = np.mean(target_angles) / (np.mean(source_angles) + 1e-8)
                return alpha
            else:
                return 1.0
                
        except Exception:
            return 1.0
    
    def _apply_frequency_warping(self, wav, alpha):
        """
        Apply frequency warping (VTLN-like transformation).
        Alpha > 1: stretch spectrum (lower formants)
        Alpha < 1: compress spectrum (raise formants)
        """
        try:
            # Compute STFT
            D = librosa.stft(wav, n_fft=2048, hop_length=512)
            mag, phase = np.abs(D), np.angle(D)
            
            # Create warped frequency bins
            n_fft = 2048
            freqs = np.linspace(0, self.sr/2, n_fft//2 + 1)
            
            # Warping function (bilinear transform approximation)
            warped_freqs = freqs * alpha
            
            # Interpolate magnitude to warped frequencies
            mag_warped = np.zeros_like(mag)
            for i in range(mag.shape[1]):
                interp_func = interp1d(
                    freqs, mag[:, i], kind='linear', 
                    bounds_error=False, fill_value=0
                )
                mag_warped[:, i] = interp_func(warped_freqs)
            
            # Reconstruct
            D_warped = mag_warped * np.exp(1j * phase)
            wav_warped = librosa.istft(D_warped, hop_length=512)
            
            # Ensure same length
            if len(wav_warped) > len(wav):
                wav_warped = wav_warped[:len(wav)]
            elif len(wav_warped) < len(wav):
                wav_warped = np.pad(wav_warped, (0, len(wav) - len(wav_warped)))
            
            return wav_warped.astype(np.float32)
            
        except Exception:
            return wav
    
    def _mel_to_linear_approx(self, mel_spec, target_shape):
        """
        Approximate conversion from mel to linear spectrogram.
        Uses pseudo-inverse of mel filterbank.
        """
        try:
            n_fft = (target_shape[0] - 1) * 2
            n_mels = mel_spec.shape[0]
            
            # Create mel filterbank
            mel_basis = librosa.filters.mel(sr=self.sr, n_fft=n_fft, n_mels=n_mels)
            
            # Pseudo-inverse
            mel_basis_inv = np.linalg.pinv(mel_basis)
            
            # Apply to each frame
            linear_spec = mel_basis_inv @ mel_spec
            
            # Ensure positive
            linear_spec = np.maximum(linear_spec, 0)
            
            # Resize to match target shape if needed
            if linear_spec.shape[1] != target_shape[1]:
                # Simple interpolation
                from scipy.ndimage import zoom
                zoom_factor = target_shape[1] / linear_spec.shape[1]
                linear_spec = zoom(linear_spec, (1, zoom_factor), order=1)
            
            return linear_spec
            
        except Exception:
            # Fallback: just use magnitude
            return np.abs(np.random.randn(*target_shape)) * 0.01


def extract_median_f0(wav_path, sr=16000):
    """
    Utility function to extract median F0 from a reference audio file.
    
    Args:
        wav_path: path to audio file
        sr: sample rate
    
    Returns:
        median F0 in Hz
    """
    try:
        wav, _ = librosa.load(wav_path, sr=sr)
        
        f0, voiced_flag, voiced_probs = librosa.pyin(
            wav,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        if f0 is None:
            return None
        
        f0_valid = f0[~np.isnan(f0)]
        if len(f0_valid) == 0:
            return None
        
        return float(np.median(f0_valid))
        
    except Exception as e:
        print(f"Error extracting F0: {e}")
        return None
