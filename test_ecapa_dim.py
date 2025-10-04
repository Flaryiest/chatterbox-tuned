# Quick test to verify ECAPA-TDNN output dimensions

import torch
from speechbrain.inference.speaker import EncoderClassifier

# Load ECAPA
ecapa = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device": "cuda"}
)

# Test with dummy audio (1 second at 16kHz)
dummy_audio = torch.randn(16000).cuda()

# Get embedding
with torch.no_grad():
    embedding = ecapa.encode_batch(dummy_audio.unsqueeze(0))
    print(f"ECAPA output shape: {embedding.shape}")
    embedding_squeezed = embedding.squeeze()
    print(f"ECAPA squeezed shape: {embedding_squeezed.shape}")
    print(f"ECAPA embedding dim: {embedding_squeezed.numel()}")
