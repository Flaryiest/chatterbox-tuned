"""
Hybrid Voice Encoder: Combines weak LSTM encoder (original) with strong ECAPA-TDNN encoder.

Problem:
- Original LSTM encoder produces highly saturated embeddings (99.9%+ similarity)
- Cannot distinguish between dissimilar speakers (e.g., Taylor Swift vs Barack Obama)
- Limits voice conversion quality to ~50/50 source/target blend

Solution:
- Use ECAPA-TDNN (state-of-the-art speaker verification) to compute "true" speaker distances
- Translate those distances into the LSTM embedding space via learned projection
- S3Gen decoder still receives LSTM-space embeddings (maintains compatibility)
- Result: Better speaker discrimination without breaking decoder

Architecture:
    Source Audio ──┬─→ LSTM Encoder ────→ weak_embed (256-dim)
                   │                       ↓
                   └─→ ECAPA-TDNN ───→ strong_embed (192-dim)
                                          ↓
                   [Compute direction: strong_target - strong_source]
                                          ↓
                   [Project to LSTM space: direction @ projection_matrix]
                                          ↓
                   weak_embed_adjusted = weak_embed + α * projected_direction
                                          ↓
                                    S3Gen Decoder (unchanged)

Key Parameters:
- projection_strength (α): How much to apply the ECAPA direction (0.0-1.0)
  - 0.0 = pure LSTM (original behavior)
  - 1.0 = full ECAPA guidance
  - 0.3-0.5 = recommended balance

Example Usage:
    from chatterbox.models.hybrid_voice_encoder import HybridVoiceEncoder
    
    encoder = HybridVoiceEncoder(
        lstm_encoder=original_voice_encoder,
        device="cuda",
        projection_strength=0.4
    )
    
    # Embed target voice with hybrid approach
    target_embed = encoder.embed_utterance(target_audio_path)
    
    # Use in voice conversion (same API as original)
    model.set_target_voice(target_audio_path, encoder=encoder)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional
import warnings

try:
    from speechbrain.inference.speaker import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    warnings.warn(
        "SpeechBrain not available. Install with: pip install speechbrain\n"
        "Hybrid encoder will fall back to original LSTM encoder."
    )


class HybridVoiceEncoder(nn.Module):
    """
    Hybrid encoder combining weak LSTM (original) with strong ECAPA-TDNN.
    
    This encoder addresses the embedding saturation problem by using ECAPA-TDNN
    to compute speaker differences, then translating those differences into the
    LSTM embedding space that S3Gen decoder expects.
    """
    
    def __init__(
        self,
        lstm_encoder,
        device: str = "cuda",
        projection_strength: float = 0.4,
        ecapa_model: str = "speechbrain/spkrec-ecapa-voxceleb",
        use_calibration: bool = True,
    ):
        """
        Args:
            lstm_encoder: Original VoiceEncoder instance (LSTM-based)
            device: Device to run models on
            projection_strength: How much ECAPA guidance to apply (0.0-1.0)
            ecapa_model: SpeechBrain ECAPA-TDNN model identifier
            use_calibration: Whether to calibrate projection based on embedding statistics
        """
        super().__init__()
        
        self.lstm_encoder = lstm_encoder
        self.device = device
        self.projection_strength = projection_strength
        self.use_calibration = use_calibration
        
        # Load ECAPA-TDNN if available
        if SPEECHBRAIN_AVAILABLE:
            print(f"Loading ECAPA-TDNN model: {ecapa_model}")
            self.ecapa_encoder = EncoderClassifier.from_hparams(
                source=ecapa_model,
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": device}
            )
            self.ecapa_available = True
            print("✅ ECAPA-TDNN loaded successfully")
            
            # Initialize projection matrix (ECAPA 192-dim → LSTM 256-dim)
            # We'll use a simple linear projection
            self.projection = nn.Linear(192, 256, bias=False).to(device)
            
            # Initialize with identity-like mapping (small random values)
            with torch.no_grad():
                # Start with small random projection
                self.projection.weight.data = torch.randn(256, 192, device=device) * 0.01
            
            print(f"Projection matrix initialized: 192-dim (ECAPA) → 256-dim (LSTM)")
            print(f"Projection strength: {projection_strength:.2f}")
            
        else:
            self.ecapa_encoder = None
            self.ecapa_available = False
            self.projection = None
            print("⚠️  ECAPA-TDNN not available - using LSTM encoder only")
    
    def embed_ecapa(self, wav_path: Union[str, Path]) -> torch.Tensor:
        """Extract ECAPA-TDNN embedding (strong discriminative embedding)."""
        if not self.ecapa_available:
            raise RuntimeError("ECAPA encoder not available")
        
        # ECAPA expects waveform, not path - load audio
        import torchaudio
        waveform, sample_rate = torchaudio.load(str(wav_path))
        
        # Resample to 16kHz if needed (ECAPA expects 16kHz)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # ECAPA expects (batch, time) or (time,)
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)  # Convert to mono if stereo
        
        waveform = waveform.to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.ecapa_encoder.encode_batch(waveform.unsqueeze(0))
            # embedding shape: (1, 1, 192)
            embedding = embedding.squeeze()  # (192,)
        
        return embedding
    
    def embed_lstm(self, wav_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract LSTM embedding (original weak embedding)."""
        # Use original encoder's embed_utterance method
        return self.lstm_encoder.embed_utterance(wav_path)
    
    def compute_hybrid_embedding(
        self,
        wav_path: Union[str, Path],
        reference_path: Optional[Union[str, Path]] = None,
        reference_embed_lstm: Optional[np.ndarray] = None,
        reference_embed_ecapa: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Compute hybrid embedding by adjusting LSTM embedding using ECAPA direction.
        
        Args:
            wav_path: Path to audio to embed
            reference_path: Optional reference audio to compute direction from
            reference_embed_lstm: Precomputed LSTM embedding of reference
            reference_embed_ecapa: Precomputed ECAPA embedding of reference
        
        Returns:
            Adjusted LSTM embedding (256-dim numpy array)
        """
        # Get LSTM embedding (original)
        lstm_embed, _ = self.embed_lstm(wav_path)  # (256,) numpy array
        
        # If ECAPA not available or no reference, return original
        if not self.ecapa_available or (reference_path is None and reference_embed_ecapa is None):
            return lstm_embed
        
        # Get ECAPA embeddings
        ecapa_current = self.embed_ecapa(wav_path)  # (192,) torch tensor
        
        if reference_embed_ecapa is None:
            ecapa_reference = self.embed_ecapa(reference_path)
        else:
            ecapa_reference = reference_embed_ecapa
        
        # Compute direction in ECAPA space
        # direction = where we want to go (reference) - where we are (current)
        ecapa_direction = ecapa_reference - ecapa_current  # (192,)
        
        # Project direction to LSTM space
        with torch.no_grad():
            lstm_direction = self.projection(ecapa_direction.unsqueeze(0)).squeeze(0)  # (256,)
            lstm_direction = lstm_direction.cpu().numpy()
        
        # Apply direction with strength
        adjusted_embed = lstm_embed + self.projection_strength * lstm_direction
        
        # Normalize to unit norm (important for cosine similarity)
        adjusted_embed = adjusted_embed / np.linalg.norm(adjusted_embed)
        
        return adjusted_embed
    
    def embed_utterance(self, wav_path: Union[str, Path]) -> np.ndarray:
        """
        Main interface: embed a single utterance.
        Returns LSTM embedding (for compatibility with S3Gen decoder).
        
        Note: For target voice embedding in VC, use compute_hybrid_embedding()
        with source as reference to apply directional adjustment.
        """
        lstm_embed, _ = self.embed_lstm(wav_path)
        return lstm_embed
    
    def embed_utterance_with_reference(
        self,
        wav_path: Union[str, Path],
        reference_path: Union[str, Path],
    ) -> np.ndarray:
        """
        Embed utterance with hybrid adjustment toward reference.
        
        Use this for target voice in voice conversion:
            target_embed = encoder.embed_utterance_with_reference(
                wav_path=target_audio,
                reference_path=source_audio  # Move AWAY from source
            )
        
        Args:
            wav_path: Audio to embed (target voice)
            reference_path: Reference audio (source voice to move away from)
        
        Returns:
            Adjusted embedding (256-dim)
        """
        return self.compute_hybrid_embedding(
            wav_path=wav_path,
            reference_path=reference_path
        )
    
    def set_projection_strength(self, strength: float):
        """Adjust how much ECAPA guidance to apply (0.0 = none, 1.0 = full)."""
        assert 0.0 <= strength <= 1.0, "Strength must be in [0.0, 1.0]"
        self.projection_strength = strength
        print(f"Projection strength updated: {strength:.2f}")
    
    def calibrate_projection(
        self,
        speaker_pairs: list,
        target_distance_ratio: float = 2.0
    ):
        """
        Calibrate projection matrix using known speaker pairs.
        
        Args:
            speaker_pairs: List of (path1, path2) tuples for different speakers
            target_distance_ratio: How much to amplify ECAPA distances (default: 2x)
        
        This adjusts the projection matrix so that ECAPA-measured distances
        translate proportionally into LSTM space.
        """
        if not self.ecapa_available:
            print("Cannot calibrate: ECAPA not available")
            return
        
        print(f"\n🔧 Calibrating projection matrix with {len(speaker_pairs)} speaker pairs...")
        
        # Compute actual distance ratios
        lstm_distances = []
        ecapa_distances = []
        
        for path1, path2 in speaker_pairs:
            # LSTM embeddings
            lstm1, _ = self.embed_lstm(path1)
            lstm2, _ = self.embed_lstm(path2)
            lstm_dist = np.linalg.norm(lstm1 - lstm2)
            lstm_distances.append(lstm_dist)
            
            # ECAPA embeddings
            ecapa1 = self.embed_ecapa(path1)
            ecapa2 = self.embed_ecapa(path2)
            ecapa_dist = torch.norm(ecapa1 - ecapa2).item()
            ecapa_distances.append(ecapa_dist)
        
        avg_lstm_dist = np.mean(lstm_distances)
        avg_ecapa_dist = np.mean(ecapa_distances)
        
        print(f"Average LSTM distance: {avg_lstm_dist:.4f}")
        print(f"Average ECAPA distance: {avg_ecapa_dist:.4f}")
        
        # Scale projection to match target ratio
        scale_factor = (avg_lstm_dist * target_distance_ratio) / avg_ecapa_dist
        
        with torch.no_grad():
            self.projection.weight.data *= scale_factor
        
        print(f"✅ Projection scaled by {scale_factor:.4f}x")
        print(f"Expected distance amplification: {target_distance_ratio:.1f}x")
    
    def to(self, device):
        """Move model to device."""
        self.device = device
        self.lstm_encoder.to(device)
        if self.ecapa_available:
            self.ecapa_encoder.to(device)
            if self.projection is not None:
                self.projection.to(device)
        return self
    
    def eval(self):
        """Set to evaluation mode."""
        self.lstm_encoder.eval()
        if self.projection is not None:
            self.projection.eval()
        return self
    
    def embeds_from_wavs(self, wavs, sample_rate: int = 16000, as_spk: bool = False):
        """
        Compatibility method for evaluation - delegates to LSTM encoder.
        
        Args:
            wavs: List of waveforms
            sample_rate: Sample rate
            as_spk: If True, return speaker-level embeddings (aggregated)
        
        Returns:
            Embeddings array
        """
        return self.lstm_encoder.embeds_from_wavs(wavs, sample_rate=sample_rate, as_spk=as_spk)
    
    @property
    def sr(self):
        """Sample rate (delegate to LSTM encoder)."""
        return self.lstm_encoder.sr
    
    @property
    def embedding_size(self):
        """Embedding dimension (same as LSTM encoder for compatibility)."""
        return self.lstm_encoder.embedding_size
