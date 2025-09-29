from pathlib import Path
import math
from typing import Iterable, List

import librosa
import numpy as np
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
        enable_pitch_cache: bool = True,
    ):
        self.sr = S3GEN_SR
        self.s3gen = s3gen
        self.device = device
        self.watermarker = perth.PerthImplicitWatermarker()
        # Pitch / prosody caches
        self._target_median_f0 = None
        self._enable_pitch_cache = enable_pitch_cache
        # Track last applied semitone pitch shift (None means no pitch matching attempted)
        self._last_pitch_shift_semitones = None  # type: float | None
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
        # Cache target median F0 for later pitch matching
        if self._enable_pitch_cache:
            self._target_median_f0 = self._extract_median_f0(s3gen_ref_wav, S3GEN_SR)

    # -------- Multi-reference target handling --------
    def set_target_voices(
        self,
        wav_paths: Iterable[str],
        mode: str = "mean",
        robust: bool = True,
        max_refs: int = 8,
    ):
        """Build an aggregated target speaker conditioning from multiple utterances.

        Currently aggregates ONLY the speaker embedding (timbre). The prompt tokens / mels are
        taken from the *first* utterance to keep shapes consistent. Future improvements could
        concatenate or randomly sample prompt segments.

        Args:
            wav_paths: iterable of wav file paths.
            mode: 'mean' (others could be added e.g. 'pca').
            robust: if True, perform a simple outlier rejection on embeddings before averaging.
            max_refs: safety cap to avoid excessive memory.
        """
        wav_list = list(wav_paths)[:max_refs]
        assert len(wav_list) > 0, "No reference paths provided"
        ref_entries = []
        for p in wav_list:
            w, _ = librosa.load(p, sr=S3GEN_SR)
            w = w[:self.DEC_COND_LEN]
            ref_entries.append(self.s3gen.embed_ref(w, S3GEN_SR, device=self.device))
        # Base dict from first
        base = ref_entries[0]
        embs = torch.stack([r["embedding"].squeeze(0) for r in ref_entries], dim=0)  # [N, D]

        # Simple robust filtering
        if robust and embs.size(0) >= 3:
            with torch.no_grad():
                dists = torch.cdist(embs, embs)  # [N,N]
                mean_dist = dists.mean(dim=1)
                thresh = mean_dist.median() + 1.5 * mean_dist.std()
                keep_mask = mean_dist <= thresh
                if keep_mask.sum() >= 2:  # ensure we keep at least two
                    embs = embs[keep_mask]

        if mode == "mean":
            agg = embs.mean(dim=0, keepdim=True)
        else:
            raise ValueError(f"Unsupported aggregation mode: {mode}")
        base["embedding"] = agg
        self.ref_dict = base
        if self._enable_pitch_cache:
            # average median F0 across kept refs
            f0_vals = []
            for p in wav_list:
                try:
                    w, _ = librosa.load(p, sr=S3GEN_SR)
                    val = self._extract_median_f0(w, S3GEN_SR)
                    if val is not None:
                        f0_vals.append(val)
                except Exception:
                    pass
            if f0_vals:
                self._target_median_f0 = float(np.median(f0_vals))
        return self.ref_dict

    # -------- Pitch utilities --------
    def _extract_median_f0(self, wav: np.ndarray | torch.Tensor, sr: int) -> float | None:
        """Extract a rough median F0 using librosa.pyin. Returns None if extraction fails."""
        try:
            if isinstance(wav, torch.Tensor):
                wav = wav.detach().cpu().numpy()
            # pyin prefers float64
            wav64 = wav.astype(np.float64)
            f0, _, _ = librosa.pyin(
                wav64,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr,
            )
            if f0 is None:
                return None
            f0_valid = f0[~np.isnan(f0)]
            if len(f0_valid) == 0:
                return None
            return float(np.median(f0_valid))
        except Exception:
            return None

    def _compute_semitone_shift(self, src_f0: float, tgt_f0: float, max_shift: float) -> float:
        if src_f0 is None or tgt_f0 is None or src_f0 <= 0 or tgt_f0 <= 0:
            return 0.0
        n_steps = 12.0 * math.log2(tgt_f0 / src_f0)
        return float(np.clip(n_steps, -max_shift, max_shift))

    def generate(
        self,
        audio,
        target_voice_path=None,
        *,
        prune_tokens: int = None,
        speaker_strength: float = None,
        flow_cfg_rate: float = None,
        pitch_match: bool = False,
        pitch_tolerance: float = 0.4,
        max_pitch_shift: float = 6.0,
        guidance_ramp: bool | None = None,
        guidance_ramp_min: float = 0.55,
        guidance_ramp_max: float = None,
        speaker_ramp: bool | None = None,
        speaker_ramp_start: float = 0.6,
        ramp_shape: str = "sigmoid",
    ):
        if target_voice_path:
            self.set_target_voice(target_voice_path)
        if target_voice_path is None:
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

            # Optional pitch matching BEFORE tokenization
            if pitch_match and self._target_median_f0 is not None:
                src_med_f0 = self._extract_median_f0(audio_16.squeeze(0).cpu().numpy(), S3_SR)
                raw_shift = self._compute_semitone_shift(src_med_f0, self._target_median_f0, max_pitch_shift)
                # Dead zone: ignore small absolute shifts (< 1 semitone)
                if abs(raw_shift) < 1.0:
                    effective_shift = 0.0
                else:
                    # Soft ramp: subtract dead-zone margin then scale
                    sign = 1.0 if raw_shift > 0 else -1.0
                    effective_shift = (abs(raw_shift) - 1.0) * 0.5 * sign  # shrink aggressiveness
                # Additional global scaling factor
                effective_shift *= 0.8
                # Clamp tighter than original max to preserve intelligibility
                if effective_shift > 3.0:
                    effective_shift = 3.0
                elif effective_shift < -3.0:
                    effective_shift = -3.0

                if abs(effective_shift) > pitch_tolerance and effective_shift != 0.0:
                    try:
                        shifted = librosa.effects.pitch_shift(
                            audio_16.squeeze(0).cpu().numpy(), sr=S3_SR, n_steps=effective_shift
                        )
                        audio_16 = torch.from_numpy(shifted).float().to(self.device).unsqueeze(0)
                        self._last_pitch_shift_semitones = float(effective_shift)
                    except Exception:
                        self._last_pitch_shift_semitones = None
                else:
                    # Either below tolerance or no effective shift
                    self._last_pitch_shift_semitones = 0.0
            else:
                self._last_pitch_shift_semitones = None

            s3_tokens, _ = self.s3gen.tokenizer(audio_16)
            if active_prune > 0 and s3_tokens.size(1) > active_prune:
                s3_tokens = s3_tokens[:, active_prune:]
            # -------- Optional scheduling (guidance & speaker scaling) --------
            # Number of internal flow steps is currently fixed at 10 (see flow_matching inference call n_timesteps)
            n_steps = 10

            # Guidance ramp: if enabled, build per-step cfg schedule
            if guidance_ramp is None:
                guidance_ramp = False
            if guidance_ramp:
                base_cfg = guidance_ramp_min
                target_cfg = flow_cfg_rate if flow_cfg_rate is not None else getattr(self.s3gen, '_inference_cfg_rate', 0.8)
                peak_cfg = guidance_ramp_max if guidance_ramp_max is not None else target_cfg
                if ramp_shape == "sigmoid":
                    xs = torch.linspace(-4, 4, n_steps)
                    sig = torch.sigmoid(xs)
                    cfg_sched = (base_cfg + (peak_cfg - base_cfg) * sig).tolist()
                else:
                    cfg_sched = torch.linspace(base_cfg, peak_cfg, n_steps).tolist()
                self.s3gen.set_cfg_rate_schedule(cfg_sched)
            else:
                if hasattr(self.s3gen, 'set_cfg_rate_schedule'):
                    self.s3gen.set_cfg_rate_schedule(None)

            # Speaker scaling ramp
            if speaker_ramp is None:
                speaker_ramp = False
            if speaker_ramp:
                final_strength = speaker_strength if speaker_strength is not None else getattr(self.s3gen, 'speaker_strength', 1.0)
                start_strength = speaker_ramp_start
                if ramp_shape == "sigmoid":
                    xs = torch.linspace(-4, 4, n_steps)
                    sig = torch.sigmoid(xs)
                    scales = (start_strength + (final_strength - start_strength) * sig) / max(final_strength, 1e-6)
                else:
                    scales = torch.linspace(start_strength, final_strength, n_steps) / max(final_strength, 1e-6)
                self.s3gen.set_speaker_scale_schedule(scales.tolist())
            else:
                if hasattr(self.s3gen, 'set_speaker_scale_schedule'):
                    self.s3gen.set_speaker_scale_schedule(None)
            # Now run inference with schedules applied
            wav, _ = self.s3gen.inference(
                speech_tokens=s3_tokens,
                ref_dict=self.ref_dict,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)

    def get_last_pitch_shift(self) -> float | None:
        """Return the semitone pitch shift applied in the most recent generate call (if any)."""
        return self._last_pitch_shift_semitones