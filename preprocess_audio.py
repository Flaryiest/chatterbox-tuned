"""CLI for Chatterbox audio pre-processing (reference & source neutralization).

Usage (PowerShell examples):

# Preprocess a reference file
python preprocess_audio.py --reference path\to\target.wav --out ref_proc.wav

# Preprocess a source file (needs a reference for neutralization alignment)
python preprocess_audio.py --reference path\to\target.wav --source path\to\source.wav --out-source source_proc.wav

If both --reference and --source are specified and --out / --out-source given, both are processed.

Outputs are raw waveform .wav files at their processing sample rates.
"""
from __future__ import annotations
import argparse
import soundfile as sf
import numpy as np
from pathlib import Path

from chatterbox.audio_preprocessing import (
    preprocess_reference,
    preprocess_source,
    ReferencePreprocessConfig,
)
import librosa


def save_wav(path: str, y: np.ndarray, sr: int):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y, sr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=str, help="Path to reference (target voice) wav.")
    ap.add_argument("--source", type=str, help="Optional path to source wav for neutralization.")
    ap.add_argument("--out", type=str, help="Output path for processed reference audio.")
    ap.add_argument("--out-source", type=str, help="Output path for processed source audio.")
    ap.add_argument("--no-gate", action="store_true", help="Disable spectral gating.")
    ap.add_argument("--no-stable", action="store_true", help="Disable stable window selection.")
    ap.add_argument("--no-tilt", action="store_true", help="Disable spectral tilt normalization.")
    ap.add_argument("--target-lufs", type=float, default=-20.0, help="Target LUFS for reference.")
    args = ap.parse_args()

    if not args.reference:
        raise SystemExit("--reference is required")
    if not args.out:
        raise SystemExit("--out (reference output path) is required")

    cfg = ReferencePreprocessConfig(
        target_lufs=args.target_lufs,
        apply_gate=not args.no_gate,
        use_stable_window=not args.no_stable,
        spectral_tilt=not args.no_tilt,
    )

    ref_audio, info = preprocess_reference(args.reference, cfg)
    save_wav(args.out, ref_audio, cfg.sr)
    print(f"Processed reference saved -> {args.out} ({info})")

    if args.source:
        if not args.out_source:
            raise SystemExit("--out-source required when --source is provided")
        # Downsample reference for alignment segment
        ref_16 = librosa.resample(ref_audio, orig_sr=cfg.sr, target_sr=16000)
        src_audio = preprocess_source(args.source, ref_16, source_sr=16000)
        save_wav(args.out_source, src_audio, 16000)
        print(f"Processed source saved -> {args.out_source}")


if __name__ == "__main__":
    main()
