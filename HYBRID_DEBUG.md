# Quick Fix Instructions

If the hybrid encoder keeps failing with dimension mismatches, you can temporarily disable it:

## Option 1: Disable Hybrid Encoder

In `colab.py`, change:
```python
USE_HYBRID_ENCODER = False  # Temporarily disable
```

## Option 2: Check ECAPA Dimensions

Run this in a cell to check actual ECAPA output:
```python
from speechbrain.inference.speaker import EncoderClassifier
import torch

ecapa = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device": "cuda"}
)

dummy = torch.randn(16000).cuda()
emb = ecapa.encode_batch(dummy.unsqueeze(0))
print(f"ECAPA shape: {emb.shape}")
print(f"ECAPA squeezed: {emb.squeeze().shape}")
```

## What to Look For in Debug Output

When you run with debug prints enabled, look for:
```
DEBUG: ecapa_embed shape: torch.Size([???])
DEBUG: campplus_embed shape: torch.Size([1, 80])  # or torch.Size([80])
```

The ECAPA shape should be `[192]`, but if it's something else (like `[1, 192]` or a different number), that's the issue.

## Expected Next Run Output

You should see:
```
DEBUG: ecapa_embed shape: torch.Size([192]), dtype: torch.float32, device: cuda:0
DEBUG: campplus_embed shape: torch.Size([1, 80]), dtype: torch.float32, device: cuda:0
DEBUG: ecapa_projected shape: torch.Size([80])
DEBUG: Hybrid encoding successful!
```

If dimensions don't match, the traceback will show exactly where.
