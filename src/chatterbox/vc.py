from pathlib import Path

import librosa
import torch
import perth
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.s3tokenizer import S3_SR
from .models.s3gen import S3GEN_SR, S3Gen


REPO_ID = "ResembleAI/chatterbox"


class ChatterboxVC:
    """Voice conversion interface over S3Gen.

    Features:
      - Load pretrained/local checkpoints
      - Single or multi-reference speaker conditioning
      - Runtime adjustable classifier-free guidance rate (flow stage)
      - Speaker embedding strength scaling
      - Optional token pruning (drop earliest tokens)
      - Watermarking via Perth implicit watermark

    Args:
        s3gen: Initialized ``S3Gen`` acoustic model instance.
        device: Torch device string ("cuda", "cpu", "mps", etc.).
        ref_dict: Optional pre-computed reference conditioning dict.
        flow_cfg_rate: Classifier-free guidance rate for flow model (0..1+).
        speaker_strength: Scalar multiplier applied to speaker embedding.
        prune_tokens: Number of initial tokens to drop before decoding.
    """

    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        s3gen: S3Gen,
        device: str,
        ref_dict: dict = None,
        *,
        flow_cfg_rate: float = 0.8,
        speaker_strength: float = 1.0,
        prune_tokens: int = 0,
    ) -> None:
        # Basic validation / clamping (avoid silent misuse)
        if flow_cfg_rate is not None:
            flow_cfg_rate = float(flow_cfg_rate)
        if speaker_strength is not None:
            speaker_strength = float(speaker_strength)
        if flow_cfg_rate is not None and flow_cfg_rate < 0:
            raise ValueError("flow_cfg_rate must be >= 0")
        if speaker_strength is not None and speaker_strength <= 0:
            raise ValueError("speaker_strength must be > 0")

        self.sr = S3GEN_SR
        self.s3gen = s3gen
        self.device = device
        self.watermarker = perth.PerthImplicitWatermarker()
        # configure runtime knobs if supported by underlying model
        if hasattr(self.s3gen, 'set_inference_cfg_rate'):
            self.s3gen.set_inference_cfg_rate(flow_cfg_rate)
        if hasattr(self.s3gen, 'set_speaker_strength'):
            self.s3gen.set_speaker_strength(speaker_strength)
        self.prune_tokens = int(prune_tokens) if prune_tokens and prune_tokens > 0 else 0
        if ref_dict is None:
            self.ref_dict = None
        else:
            self.ref_dict = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in ref_dict.items()
            }

    @classmethod
    def from_local(cls, ckpt_dir, device, *, flow_cfg_rate: float = 0.7, speaker_strength: float = 1.0, prune_tokens: int = 0) -> 'ChatterboxVC':
        ckpt_dir = Path(ckpt_dir)
        
        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None
            
        ref_dict = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            states = torch.load(builtin_voice, map_location=map_location)
            ref_dict = states['gen']

        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen.safetensors"), strict=False
        )
        s3gen.to(device).eval()

        # Pass configuration into constructor
        return cls(
            s3gen,
            device,
            ref_dict=ref_dict,
            flow_cfg_rate=flow_cfg_rate,
            speaker_strength=speaker_strength,
            prune_tokens=prune_tokens,
        )

    @classmethod
    def from_pretrained(cls, device, *, flow_cfg_rate: float = 0.7, speaker_strength: float = 1.0, prune_tokens: int = 0) -> 'ChatterboxVC':
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"
            
        for fpath in ["s3gen.safetensors", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)
        return cls.from_local(
            Path(local_path).parent,
            device,
            flow_cfg_rate=flow_cfg_rate,
            speaker_strength=speaker_strength,
            prune_tokens=prune_tokens,
        )

    def set_target_voice(self, wav_fpath: str | Path) -> None:
        """Set single target voice reference from a wav file.

        Truncates to ``DEC_COND_LEN`` seconds to match training reference length.
        """
        wav_fpath = Path(wav_fpath)
        if not wav_fpath.exists():
            raise FileNotFoundError(wav_fpath)
        s3gen_ref_wav, _sr = librosa.load(str(wav_fpath), sr=S3GEN_SR)
        s3gen_ref_wav = s3gen_ref_wav[: self.DEC_COND_LEN]
        self.ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

    def set_target_voice_multi(
        self,
        wav_fpaths: list[str | Path],
        *,
        weights: list[float] | None = None,
        aggregate: str = "mean",
    ) -> None:
        """Set target voice from multiple reference wav files.

        Args:
            wav_fpaths: List of wav paths.
            weights: Optional list of positive weights (same length). Normalized internally.
            aggregate: 'mean' (weighted) or 'median'.

        Notes:
            - Each individual embedding dict contains tensors with shape [...]. We aggregate per-key.
            - Non-tensor items are copied from the first reference.
        """
        if not wav_fpaths:
            raise ValueError("wav_fpaths must be non-empty")
        paths = [Path(p) for p in wav_fpaths]
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(p)
        if weights is not None:
            if len(weights) != len(paths):
                raise ValueError("weights length must match wav_fpaths length")
            if any(w < 0 for w in weights):
                raise ValueError("weights must be non-negative")
            w = torch.tensor(weights, dtype=torch.float32)
            if w.sum() == 0:
                raise ValueError("weights must sum > 0")
            w = w / w.sum()
        else:
            w = torch.ones(len(paths), dtype=torch.float32) / len(paths)

        # Collect embeddings
        emb_list = []
        for p in paths:
            wav, _sr = librosa.load(str(p), sr=S3GEN_SR)
            wav = wav[: self.DEC_COND_LEN]
            emb_list.append(self.s3gen.embed_ref(wav, S3GEN_SR, device=self.device))

        # Keys consistency
        keys = emb_list[0].keys()
        out: dict[str, torch.Tensor] = {}
        for k in keys:
            if torch.is_tensor(emb_list[0][k]):
                stack = torch.stack([e[k].to(self.device) * w[i] for i, e in enumerate(emb_list)], dim=0)
                if aggregate == "mean":
                    out[k] = stack.sum(dim=0)
                elif aggregate == "median":
                    # Convert back to original scale (weights ignored for median)
                    out[k] = torch.median(stack, dim=0).values
                else:
                    raise ValueError("aggregate must be 'mean' or 'median'")
            else:
                # Non-tensor value; copy first
                out[k] = emb_list[0][k]
        self.ref_dict = out

    def generate(
        self,
        audio,
        target_voice_path=None,
        *,
        prune_tokens: int = None,
        speaker_strength: float = None,
        flow_cfg_rate: float = None,
        return_dict: bool = False,
    ):
        """Generate converted audio.

        Args:
            audio: Path to source audio wav.
            target_voice_path: Optional single reference wav (overrides existing ref_dict).
            prune_tokens: Override prune_tokens for this call.
            speaker_strength: Override embedding scale.
            flow_cfg_rate: Override flow guidance rate.
            return_dict: If True return (audio_tensor, meta_dict) instead of tensor only.
        """
        if target_voice_path:
            self.set_target_voice(target_voice_path)
        else:
            assert self.ref_dict is not None, "Please `prepare_conditionals` first or specify `target_voice_path`"

        # Allow per-call overrides
        if speaker_strength is not None and hasattr(self.s3gen, 'set_speaker_strength'):
            self.s3gen.set_speaker_strength(speaker_strength)
        if flow_cfg_rate is not None and hasattr(self.s3gen, 'set_inference_cfg_rate'):
            self.s3gen.set_inference_cfg_rate(flow_cfg_rate)
        if prune_tokens is not None:
            active_prune = int(prune_tokens)
        else:
            active_prune = self.prune_tokens

        with torch.inference_mode():
            src_wav, _ = librosa.load(audio, sr=S3_SR)
            src_wav_t = torch.from_numpy(src_wav).float().to(self.device)[None, ]

            s3_tokens, _ = self.s3gen.tokenizer(src_wav_t)
            token_count_before = s3_tokens.size(1)
            if active_prune > 0 and s3_tokens.size(1) > active_prune:
                s3_tokens = s3_tokens[:, active_prune:]
            token_count_after = s3_tokens.size(1)
            wav, _ = self.s3gen.inference(
                speech_tokens=s3_tokens,
                ref_dict=self.ref_dict,
            )
            wav_np = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav_np, sample_rate=self.sr)

        audio_out = torch.from_numpy(watermarked_wav).unsqueeze(0)
        if not return_dict:
            return audio_out
        # Gather meta diagnostics
        cfg_rate = None
        spk_strength = None
        if hasattr(self.s3gen, '_inference_cfg_rate'):
            cfg_rate = getattr(self.s3gen, '_inference_cfg_rate')
        if hasattr(self.s3gen, 'speaker_strength'):
            spk_strength = getattr(self.s3gen, 'speaker_strength')
        meta = {
            'tokens_before': int(token_count_before),
            'tokens_after': int(token_count_after),
            'pruned': int(token_count_before - token_count_after),
            'active_prune': int(active_prune),
            'flow_cfg_rate': cfg_rate,
            'speaker_strength': spk_strength,
        }
        return audio_out, meta