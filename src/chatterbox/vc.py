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
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        s3gen: S3Gen,
        device: str,
        ref_dict: dict=None,
        *,
        flow_cfg_rate: float = 0.8,
        speaker_strength: float = 1.0,
        prune_tokens: int = 0,
    ):
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

    def set_target_voice(self, wav_fpath):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        self.ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

    def generate(
        self,
        audio,
        target_voice_path=None,
        *,
        prune_tokens: int = None,
        speaker_strength: float = None,
        flow_cfg_rate: float = None,
    ):
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
            audio_16, _ = librosa.load(audio, sr=S3_SR)
            audio_16 = torch.from_numpy(audio_16).float().to(self.device)[None, ]

            s3_tokens, _ = self.s3gen.tokenizer(audio_16)
            if active_prune > 0 and s3_tokens.size(1) > active_prune:
                s3_tokens = s3_tokens[:, active_prune:]
            wav, _ = self.s3gen.inference(
                speech_tokens=s3_tokens,
                ref_dict=self.ref_dict,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)