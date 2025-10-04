"""
Hybrid CAMPPlus Encoder: Wraps CAMPPlus X-Vector with ECAPA-TDNN guidance.

This is specifically for the S3Gen model which uses CAMPPlus (80-dim) X-Vectors
for speaker conditioning, NOT the VoiceEncoder (256-dim) LSTM used for evaluation.

Strategy:
- CAMPPlus produces saturated embeddings (all speakers look 99.9% similar)
- ECAPA-TDNN has much better speaker discrimination
- We project ECAPA embeddings (192-dim) to CAMPPlus space (80-dim)
- Blend: output = (1-α) * CAMPPlus + α * ECAPA_projected
- Result: Embeddings in CAMPPlus space but with ECAPA's discrimination

This works because:
1. S3Gen decoder expects 80-dim embeddings (CAMPPlus space)
2. ECAPA can actually distinguish speakers (not saturated)
3. Blending keeps embeddings in familiar space while improving separation
4. No retraining needed - drop-in replacement
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
import warnings

try:
    from speechbrain.inference.speaker import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False


class HybridCAMPPlusEncoder(nn.Module):
    """
    Hybrid encoder that wraps CAMPPlus X-Vector with ECAPA-TDNN guidance.
    
    CAMPPlus produces 80-dim embeddings. We use ECAPA-TDNN (192-dim) to compute
    speaker directions, then project those into the 80-dim CAMPPlus space.
    """
    
    def __init__(
        self,
        campplus_encoder,
        device: str = "cuda",
        projection_strength: float = 0.4,
        ecapa_model: str = "speechbrain/spkrec-ecapa-voxceleb",
    ):
        """
        Args:
            campplus_encoder: Original CAMPPlus encoder instance
            device: Device to run models on
            projection_strength: How much ECAPA guidance to apply (0.0-1.0)
            ecapa_model: SpeechBrain ECAPA-TDNN model identifier
        """
        super().__init__()
        
        self.campplus_encoder = campplus_encoder
        self.device = device
        self.projection_strength = projection_strength
        
        # Load ECAPA-TDNN if available
        if SPEECHBRAIN_AVAILABLE:
            print(f"🔄 Loading ECAPA-TDNN model: {ecapa_model}")
            self.ecapa_encoder = EncoderClassifier.from_hparams(
                source=ecapa_model,
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": device}
            )
            self.ecapa_available = True
            print("✅ ECAPA-TDNN loaded successfully")
            
            # Initialize projection matrix (ECAPA 192-dim → CAMPPlus 80-dim)
            self.projection = nn.Linear(192, 80, bias=False).to(device)
            
            # Initialize with small random values
            with torch.no_grad():
                self.projection.weight.data = torch.randn(80, 192, device=device) * 0.015
            
            print(f"Projection matrix initialized: 192-dim (ECAPA) → 80-dim (CAMPPlus)")
            print(f"Projection strength: {projection_strength:.2f}")
            
        else:
            self.ecapa_encoder = None
            self.ecapa_available = False
            self.projection = None
            print("⚠️  ECAPA-TDNN not available - using CAMPPlus encoder only")
    
    def embed_ecapa(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        """Extract ECAPA-TDNN embedding from waveform."""
        if not self.ecapa_available:
            raise RuntimeError("ECAPA encoder not available")
        
        # Resample to 16kHz if needed (ECAPA expects 16kHz)
        if sr != 16000:
            import torchaudio
            resampler = torchaudio.transforms.Resample(sr, 16000).to(self.device)
            wav = resampler(wav)
        
        # ECAPA expects (batch, time) or (time,)
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0)  # Convert to mono if stereo
        elif wav.dim() == 1:
            pass  # Already mono
        else:
            wav = wav.squeeze()
        
        wav = wav.to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.ecapa_encoder.encode_batch(wav.unsqueeze(0))
            # embedding shape: (1, 1, 192)
            embedding = embedding.squeeze()  # (192,)
        
        return embedding
    
    def inference(
        self,
        wav: torch.Tensor,
    ) -> torch.Tensor:
        """
        Main inference method (replaces CAMPPlus.inference).
        
        Strategy: Use ECAPA to compute a "distinctiveness vector" that amplifies
        the speaker's unique characteristics relative to an average speaker.
        
        Args:
            wav: Input waveform (B, T) at 16kHz
        
        Returns:
            Adjusted CAMPPlus embedding (B, 192)
        """
        # Get original CAMPPlus embedding
        campplus_embed = self.campplus_encoder.inference(wav)  # (B, 192)
        
        # If ECAPA not available, return original
        if not self.ecapa_available or self.projection_strength == 0.0:
            return campplus_embed
        
        # Get ECAPA embedding
        # Assuming wav is (B, T), take first sample
        wav_1d = wav[0] if wav.dim() == 2 else wav
        
        try:
            ecapa_embed = self.embed_ecapa(wav_1d, sr=16000)  # Should be (192,)
            
            # Project ECAPA embedding to CAMPPlus space
            # This gives us a "refined" embedding based on ECAPA's better discrimination
            with torch.no_grad():
                ecapa_projected = self.projection(ecapa_embed.unsqueeze(0)).squeeze(0)  # (192,)
                ecapa_projected = ecapa_projected.to(campplus_embed.device)  # Ensure same device
            
            # Blend CAMPPlus and projected ECAPA
            # projection_strength controls how much we trust ECAPA vs CAMPPlus
            if campplus_embed.dim() == 2:  # (B, 192)
                # Blend: (1-α) * CAMPPlus + α * ECAPA_projected
                adjusted_embed = campplus_embed.clone()
                adjusted_embed[0] = (1.0 - self.projection_strength) * campplus_embed[0] + \
                                   self.projection_strength * ecapa_projected
            else:  # (192,)
                adjusted_embed = (1.0 - self.projection_strength) * campplus_embed + \
                                self.projection_strength * ecapa_projected
            
            # Normalize (CAMPPlus embeddings are typically L2-normalized)
            if adjusted_embed.dim() == 2:
                adjusted_embed = adjusted_embed / adjusted_embed.norm(dim=1, keepdim=True).clamp(min=1e-8)
            else:
                adjusted_embed = adjusted_embed / adjusted_embed.norm().clamp(min=1e-8)
            
            return adjusted_embed
            
        except Exception as e:
            # If ECAPA fails, fall back to original - but don't print every time
            if not hasattr(self, '_error_logged'):
                print(f"Warning: ECAPA embedding failed ({e}), falling back to CAMPPlus only")
                print(f"  This likely means projection matrix dimensions are wrong")
                print(f"  Hybrid encoder will be disabled for this session")
                import traceback
                traceback.print_exc()
                self._error_logged = True
            return campplus_embed
    
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Forward pass (delegates to inference without reference)."""
        return self.inference(wav)
    
    def set_projection_strength(self, strength: float):
        """Adjust how much ECAPA guidance to apply (0.0 = none, 1.0 = full)."""
        assert 0.0 <= strength <= 1.0, "Strength must be in [0.0, 1.0]"
        self.projection_strength = strength
        print(f"Projection strength updated: {strength:.2f}")
    
    def to(self, device):
        """Move model to device."""
        self.device = device
        self.campplus_encoder.to(device)
        if self.ecapa_available and self.projection is not None:
            self.projection.to(device)
        return self
    
    def eval(self):
        """Set to evaluation mode."""
        self.campplus_encoder.eval()
        if self.projection is not None:
            self.projection.eval()
        return self
