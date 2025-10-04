"""Colab Voice Conversion Demo with Hybrid Encoder.

This script:
1. Loads a pretrained ChatterboxVC model.
2. Injects a Hybrid Encoder (ECAPA-TDNN + CAMPPlus) to fix embedding saturation.
3. Applies optional preprocessing to remove source speaker characteristics:
   - Spectral Whitening: Removes source timbre/formant character
   - Dynamic Range Compression: Flattens emotional dynamics
   - Energy Envelope Transfer: Imposes target speaker's energy patterns
4. Converts audio into target speaker voice with configurable parameters.
5. Applies optional postprocessing to enhance target similarity:
   - Spectral Morphing: Morphs output spectrum toward target characteristics
6. Computes objective speaker similarity metrics.

DEPENDENCIES: 
- scipy (for signal processing): pip install scipy
- speechbrain (for hybrid encoder): pip install speechbrain
"""

import torch
import soundfile as sf
import librosa
import numpy as np
from IPython.display import Audio, display
import time
from datetime import datetime

# Preprocessing dependencies (install with: pip install scipy)
try:
    import scipy.ndimage
    from scipy.signal import medfilt
    PREPROCESSING_AVAILABLE = True
except ImportError:
    print("WARNING: scipy not installed. Preprocessing will be disabled.")
    print("Install with: pip install scipy")
    PREPROCESSING_AVAILABLE = False

from chatterbox.vc import ChatterboxVC
from chatterbox.models.voice_encoder import VoiceEncoder

# Try to import hybrid encoder (requires speechbrain)
try:
    from chatterbox.models.hybrid_voice_encoder import HybridVoiceEncoder, HybridCAMPPlusEncoder
    HYBRID_ENCODER_AVAILABLE = True
except ImportError:
    HYBRID_ENCODER_AVAILABLE = False
    HybridVoiceEncoder = None  # Define as None for isinstance checks
    HybridCAMPPlusEncoder = None

# ---------------- Installation (Run first in Colab) ----------------
# !pip install scipy

# ---------------- User Config ----------------
SOURCE_AUDIO = "/content/TaylorSwiftShort.wav"
TARGET_VOICE_PATH = "/content/Barack Obama.mp3"

# PREPROCESSING
ENABLE_PREPROCESSING = True  # Apply spectral whitening, compression, energy transfer
ENABLE_POSTPROCESSING = True  # Apply spectral morphing to output

# VOICE CONVERSION PARAMETERS
FLOW_CFG_RATE = 0.70       # Classifier-free guidance rate (0.5-1.0 typical)
SPEAKER_STRENGTH = 1.1     # Speaker embedding scaling (1.0-1.3 typical)
PRUNE_TOKENS = 0           # Token pruning (0-8, use to reduce source leakage)
ENABLE_PITCH_MATCH = True  # Match pitch to target
PITCH_TOLERANCE = 0.6      # Ignore pitch shifts smaller than this (semitones)
MAX_PITCH_SHIFT = 2.0      # Clamp extreme pitch shifts

# HYBRID VOICE ENCODER (Fixes Embedding Saturation)
USE_HYBRID_ENCODER = True  # Use ECAPA-TDNN + CAMPPlus hybrid encoder
HYBRID_PROJECTION_STRENGTH = 0.4  # ECAPA guidance strength (0.0-1.0, recommend 0.3-0.5)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ============================================================================
# PREPROCESSING FUNCTIONS WITH LOGGING
# ============================================================================

def log_step(message):
    """Log with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")

def spectral_whitening(audio, sr, alpha=0.7):
    """Remove spectral coloration from source speaker"""
    if not PREPROCESSING_AVAILABLE:
        log_step("Spectral whitening skipped (scipy not available)")
        return audio
    log_step("Starting spectral whitening...")
    start = time.time()
    
    # STFT
    stft = librosa.stft(audio, n_fft=2048, hop_length=512)
    mag, phase = np.abs(stft), np.angle(stft)
    
    # Compute spectral envelope (smoothed magnitude)
    envelope = scipy.ndimage.gaussian_filter1d(mag, sigma=5, axis=0)
    
    # Whiten (reduce envelope influence)
    whitened_mag = mag / (envelope ** alpha + 1e-8)
    
    # Reconstruct
    whitened_stft = whitened_mag * np.exp(1j * phase)
    result = librosa.istft(whitened_stft, hop_length=512)
    
    elapsed = time.time() - start
    log_step(f"Spectral whitening complete in {elapsed:.3f}s")
    return result

def compress_dynamics(audio, sr, threshold_db=-20, ratio=4.0):
    """Apply compression to flatten volume dynamics"""
    log_step("Starting dynamic range compression...")
    start = time.time()
    
    # Simple RMS-based normalization
    rms = np.sqrt(np.mean(audio**2))
    normalized = audio / (rms + 1e-8) * 0.1
    
    # Apply simple compression
    audio_db = 20 * np.log10(np.abs(normalized) + 1e-8)
    compressed = np.where(
        audio_db > threshold_db,
        np.sign(normalized) * np.power(10, (threshold_db + (audio_db - threshold_db) / ratio) / 20),
        normalized
    )
    
    # Normalize back
    compressed = compressed / (np.max(np.abs(compressed)) + 1e-8) * 0.95
    
    elapsed = time.time() - start
    log_step(f"Dynamic compression complete in {elapsed:.3f}s")
    return compressed

def transfer_energy_envelope(source_audio, target_audio, sr):
    """Replace source energy contour with target's"""
    if not PREPROCESSING_AVAILABLE:
        log_step("Energy envelope transfer skipped (scipy not available)")
        return source_audio
    log_step("Starting energy envelope transfer...")
    start = time.time()
    
    # Extract energy envelopes
    src_rms = librosa.feature.rms(y=source_audio, frame_length=2048, hop_length=512)[0]
    tgt_rms = librosa.feature.rms(y=target_audio, frame_length=2048, hop_length=512)[0]
    
    # Smooth target envelope
    tgt_envelope_smooth = medfilt(tgt_rms, kernel_size=21)
    
    # Match lengths
    if len(tgt_envelope_smooth) != len(src_rms):
        tgt_envelope_smooth = np.interp(
            np.linspace(0, 1, len(src_rms)),
            np.linspace(0, 1, len(tgt_envelope_smooth)),
            tgt_envelope_smooth
        )
    
    # Apply target envelope to source
    src_stft = librosa.stft(source_audio, hop_length=512)
    src_mag, src_phase = np.abs(src_stft), np.angle(src_stft)
    
    # Scale magnitude by envelope ratio
    envelope_ratio = tgt_envelope_smooth / (src_rms + 1e-8)
    scaled_mag = src_mag * envelope_ratio[None, :]
    
    modified_stft = scaled_mag * np.exp(1j * src_phase)
    result = librosa.istft(modified_stft, hop_length=512)
    
    elapsed = time.time() - start
    log_step(f"Energy envelope transfer complete in {elapsed:.3f}s")
    return result

def spectral_morphing_postprocess(output_audio, target_audio, sr, alpha=0.6):
    """Morph output spectrum toward target's spectral characteristics"""
    if not PREPROCESSING_AVAILABLE:
        log_step("Spectral morphing skipped (scipy not available)")
        return output_audio
    log_step("Starting spectral morphing postprocess...")
    start = time.time()
    
    # Get spectral envelopes
    out_stft = librosa.stft(output_audio)
    
    # Match target length to output
    if len(target_audio) < len(output_audio):
        target_audio = np.pad(target_audio, (0, len(output_audio) - len(target_audio)), mode='reflect')
    else:
        target_audio = target_audio[:len(output_audio)]
    
    tgt_stft = librosa.stft(target_audio)
    
    out_mag = np.abs(out_stft)
    tgt_mag = np.abs(tgt_stft)
    out_phase = np.angle(out_stft)
    
    # Match time dimensions
    min_frames = min(out_mag.shape[1], tgt_mag.shape[1])
    out_mag = out_mag[:, :min_frames]
    tgt_mag = tgt_mag[:, :min_frames]
    out_phase = out_phase[:, :min_frames]
    
    # Get smoothed spectral envelopes
    out_env = scipy.ndimage.gaussian_filter1d(out_mag, sigma=3, axis=0)
    tgt_env = scipy.ndimage.gaussian_filter1d(tgt_mag, sigma=3, axis=0)
    
    # Morph envelope
    morphed_env = out_env ** (1 - alpha) * tgt_env ** alpha
    
    # Apply to output
    morphed_mag = out_mag * (morphed_env / (out_env + 1e-8))
    morphed_stft = morphed_mag * np.exp(1j * out_phase)
    
    result = librosa.istft(morphed_stft)
    
    elapsed = time.time() - start
    log_step(f"Spectral morphing complete in {elapsed:.3f}s")
    return result

def preprocess_audio_pipeline(audio_path, target_path, sr=16000):
    """Apply preprocessing pipeline to improve target similarity.
    
    Applies:
    - Spectral whitening to remove source timbre
    - Dynamic range compression to flatten emotional dynamics
    - Energy envelope transfer to impose target characteristics
    
    Args:
        audio_path: Path to source audio file
        target_path: Path to target voice audio file
        sr: Sample rate for processing (default 16000)
    
    Returns:
        Preprocessed audio as numpy array
    """
    log_step("="*60)
    log_step("STARTING PREPROCESSING PIPELINE")
    log_step("="*60)
    
    if not PREPROCESSING_AVAILABLE:
        log_step("WARNING: Preprocessing disabled (scipy not installed)")
        log_step("Returning audio without preprocessing")
        audio, _ = librosa.load(audio_path, sr=sr)
        return audio
    
    pipeline_start = time.time()
    
    # Load audio
    log_step(f"Loading source audio: {audio_path}")
    audio, _ = librosa.load(audio_path, sr=sr)
    log_step(f"Source audio length: {len(audio)/sr:.2f}s")
    
    log_step(f"Loading target audio: {target_path}")
    target, _ = librosa.load(target_path, sr=sr)
    log_step(f"Target audio length: {len(target)/sr:.2f}s")
    
    # Apply preprocessing steps
    # 1. Spectral whitening (remove source timbre)
    audio = spectral_whitening(audio, sr, alpha=0.7)
    
    # 2. Dynamic range compression (flatten dynamics)
    audio = compress_dynamics(audio, sr, threshold_db=-20, ratio=4.0)
    
    # 3. Energy envelope transfer (impose target characteristics)
    audio = transfer_energy_envelope(audio, target, sr)
    
    pipeline_elapsed = time.time() - pipeline_start
    log_step("="*60)
    log_step(f"PREPROCESSING COMPLETE - Total time: {pipeline_elapsed:.3f}s")
    log_step("="*60)
    
    return audio

# ---------------- Load Model ----------------
log_step("Loading Chatterbox voice conversion model...")
model = ChatterboxVC.from_pretrained(
    device,
    flow_cfg_rate=FLOW_CFG_RATE,
    speaker_strength=SPEAKER_STRENGTH,
    prune_tokens=PRUNE_TOKENS,
)

# Inject hybrid encoder into model if enabled
if USE_HYBRID_ENCODER and HYBRID_ENCODER_AVAILABLE and HybridCAMPPlusEncoder is not None:
    try:
        log_step("\n🔧 Injecting HYBRID ENCODER into S3Gen speaker encoder...")
        log_step("   (This replaces CAMPPlus X-Vector with ECAPA-guided version)")
        
        # Wrap the existing CAMPPlus encoder
        original_campplus = model.s3gen.speaker_encoder
        hybrid_campplus = HybridCAMPPlusEncoder(
            campplus_encoder=original_campplus,
            device=device,
            projection_strength=HYBRID_PROJECTION_STRENGTH,
        ).to(device).eval()
        
        # Replace in the model
        model.s3gen.speaker_encoder = hybrid_campplus
        log_step("✅ Hybrid encoder injected successfully!")
        log_step(f"   Projection strength: {HYBRID_PROJECTION_STRENGTH:.2f}")
        log_step("   This should break past embedding saturation!")
    except Exception as e:
        log_step(f"⚠️  Failed to inject hybrid encoder: {e}")
        log_step(f"   Error details: {type(e).__name__}")
        log_step("   Model will use standard CAMPPlus encoder")
        import traceback
        traceback.print_exc()
elif USE_HYBRID_ENCODER and not HYBRID_ENCODER_AVAILABLE:
    log_step("\n⚠️  Hybrid encoder requested but speechbrain not installed")
    log_step("   Install with: pip install speechbrain")
    log_step("   Using standard CAMPPlus encoder")

log_step("Model loaded successfully")

# Prepare target conditioning (single reference)
model.set_target_voice(TARGET_VOICE_PATH)

# ---------------- Preprocessing Phase ----------------
log_step("\n" + "="*80)
log_step("PHASE 1: PREPROCESSING")
log_step("="*80)

# Apply preprocessing pipeline
if ENABLE_PREPROCESSING:
    preprocessed_audio = preprocess_audio_pipeline(
        SOURCE_AUDIO,
        TARGET_VOICE_PATH,
        sr=16000
    )
else:
    log_step("Preprocessing disabled, loading audio directly")
    preprocessed_audio, _ = librosa.load(SOURCE_AUDIO, sr=16000)

# Save preprocessed audio for inspection
preprocessed_path = "/content/preprocessed.wav"
sf.write(preprocessed_path, preprocessed_audio, 16000)
log_step(f"Preprocessed audio saved to: {preprocessed_path}")

# ---------------- Conversion (Primary) ----------------
log_step("\n" + "="*80)
log_step("PHASE 2: VOICE CONVERSION")
log_step("="*80)

conversion_start = time.time()

log_step(f"Voice conversion parameters:")
log_step(f"  - speaker_strength: {SPEAKER_STRENGTH}")
log_step(f"  - flow_cfg_rate: {FLOW_CFG_RATE}")
log_step(f"  - prune_tokens: {PRUNE_TOKENS}")
log_step(f"  - pitch_match: {ENABLE_PITCH_MATCH}")

wav = model.generate(
    audio=preprocessed_path,
    target_voice_path=TARGET_VOICE_PATH,
    speaker_strength=SPEAKER_STRENGTH,
    flow_cfg_rate=FLOW_CFG_RATE,
    prune_tokens=PRUNE_TOKENS,
    pitch_match=ENABLE_PITCH_MATCH,
    pitch_tolerance=PITCH_TOLERANCE,
    max_pitch_shift=MAX_PITCH_SHIFT,
)

conversion_elapsed = time.time() - conversion_start
log_step(f"\nVoice conversion completed in {conversion_elapsed:.3f}s")

out_path = "/content/output_preprocessed.wav"
sf.write(out_path, wav.squeeze(0).cpu().numpy(), model.sr)

# ---------------- Postprocessing Phase ----------------
log_step("\n" + "="*80)
log_step("PHASE 3: POSTPROCESSING")
log_step("="*80)

# Apply spectral morphing postprocess
output_audio = wav.squeeze(0).cpu().numpy()

if ENABLE_POSTPROCESSING:
    target_audio, _ = librosa.load(TARGET_VOICE_PATH, sr=model.sr)
    postprocessed_audio = spectral_morphing_postprocess(
        output_audio,
        target_audio,
        model.sr,
        alpha=0.6
    )
else:
    log_step("Postprocessing disabled, using output as-is")
    postprocessed_audio = output_audio

postprocessed_path = "/content/output_postprocessed.wav"
sf.write(postprocessed_path, postprocessed_audio, model.sr)
log_step(f"Postprocessed audio saved to: {postprocessed_path}")

log_step("\n" + "="*80)
log_step("ALL PHASES COMPLETE")
log_step("="*80)
log_step(f"Original output: {out_path}")
log_step(f"Postprocessed output: {postprocessed_path}")

display(Audio(filename=out_path, rate=model.sr))
log_step("\n[Playing: Output without postprocessing]")

if ENABLE_POSTPROCESSING:
    display(Audio(filename=postprocessed_path, rate=model.sr))
    log_step("[Playing: Output with postprocessing]")

print(f"\nSettings -> flow_cfg_rate={FLOW_CFG_RATE}, speaker_strength={SPEAKER_STRENGTH}, prune_tokens={PRUNE_TOKENS}, pitch_match={ENABLE_PITCH_MATCH}")

# ---------------- Encoder Discrimination Diagnostic ----------------
if USE_HYBRID_ENCODER and HYBRID_ENCODER_AVAILABLE and isinstance(model.s3gen.speaker_encoder, HybridCAMPPlusEncoder):
    log_step("\n" + "="*80)
    log_step("ENCODER DISCRIMINATION DIAGNOSTIC")
    log_step("="*80)
    log_step("Comparing CAMPPlus vs ECAPA discrimination on source/target pair...")
    
    try:
        import torch.nn.functional as F
        
        # Load source and target audio
        source_wav, _ = librosa.load(SOURCE_AUDIO, sr=16000)
        target_wav, _ = librosa.load(TARGET_VOICE_PATH, sr=16000)
        
        # Convert to tensors
        source_tensor = torch.from_numpy(source_wav).unsqueeze(0).to(device)
        target_tensor = torch.from_numpy(target_wav).unsqueeze(0).to(device)
        
        # Get raw CAMPPlus embeddings (192-dim)
        campplus_only = model.s3gen.speaker_encoder.campplus_encoder
        with torch.no_grad():
            src_camp = campplus_only.inference(source_tensor)
            tgt_camp = campplus_only.inference(target_tensor)
        
        # Get ECAPA embeddings (192-dim)
        with torch.no_grad():
            ecapa_src = model.s3gen.speaker_encoder.embed_ecapa(source_tensor[0], sr=16000)
            ecapa_tgt = model.s3gen.speaker_encoder.embed_ecapa(target_tensor[0], sr=16000)
        
        # Compute similarities
        camp_sim = F.cosine_similarity(src_camp, tgt_camp, dim=-1).item()
        ecapa_sim = F.cosine_similarity(ecapa_src.unsqueeze(0), ecapa_tgt.unsqueeze(0), dim=-1).item()
        
        # Compute advantage (negative = ECAPA sees MORE difference)
        ecapa_advantage = camp_sim - ecapa_sim
        
        print(f"\n📊 ENCODER DISCRIMINATION RESULTS:")
        print(f"   CAMPPlus similarity:  {camp_sim:.6f}")
        print(f"   ECAPA similarity:     {ecapa_sim:.6f}")
        print(f"   ECAPA advantage:      {ecapa_advantage:.6f}")
        
        if ecapa_advantage > 0.01:
            print(f"\n   ✅ ECAPA discriminates BETTER than CAMPPlus by {ecapa_advantage:.4f}")
            print(f"      → Hybrid encoder should provide meaningful improvement!")
        elif ecapa_advantage > 0.001:
            print(f"\n   ⚠️  ECAPA slightly better than CAMPPlus by {ecapa_advantage:.4f}")
            print(f"      → Hybrid encoder may provide small improvement")
        elif ecapa_advantage > -0.001:
            print(f"\n   ⚠️  ECAPA and CAMPPlus perform equally (difference: {abs(ecapa_advantage):.4f})")
            print(f"      → Hybrid encoder unlikely to help with this pair")
        else:
            print(f"\n   ❌ ECAPA discriminates WORSE than CAMPPlus by {abs(ecapa_advantage):.4f}")
            print(f"      → Hybrid encoder may degrade performance")
        
        # Analyze absolute discrimination
        if camp_sim > 0.999 and ecapa_sim > 0.999:
            print(f"\n   🔴 EXTREME SATURATION DETECTED (both >0.999)")
            print(f"      Both encoders see these speakers as nearly identical!")
            print(f"      RECOMMENDATION: Try different speaker pairs")
        elif camp_sim > 0.995 and ecapa_sim > 0.995:
            print(f"\n   🟡 HIGH SATURATION (both >0.995)")
            print(f"      Limited room for improvement with any encoder")
        else:
            print(f"\n   🟢 MODERATE DISCRIMINATION (both <0.995)")
            print(f"      Good test case for hybrid encoder!")
        
    except Exception as e:
        log_step(f"⚠️  Encoder diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

# ---------------- Identity Shift Evaluation ----------------
log_step("\n" + "="*80)
log_step("PHASE 4: IDENTITY SHIFT EVALUATION")
log_step("="*80)

def load_embeds_utterance(path: str, ve: VoiceEncoder, sr_target: int = 16000):
    """Return both utterance-level (speaker) embedding and raw partial embeddings for richer metrics."""
    wav, _ = librosa.load(path, sr=sr_target)
    # as_spk=False returns utterance embeddings (partials) shape (U, E)
    partial_embeds = ve.embeds_from_wavs([wav], sample_rate=sr_target, as_spk=False)
    partial_embeds_t = torch.from_numpy(partial_embeds)  # (U, E)
    spk_embed = VoiceEncoder.utt_to_spk_embed(partial_embeds)  # aggregated
    spk_embed_t = torch.from_numpy(spk_embed)
    return spk_embed_t, partial_embeds_t

def cosine(a: torch.Tensor, b: torch.Tensor):
    return float(torch.dot(a, b) / (a.norm() * b.norm()))

def l2(a: torch.Tensor, b: torch.Tensor):
    return float(torch.norm(a - b))

# Load base LSTM encoder
log_step("Loading voice encoder for evaluation...")
lstm_encoder = VoiceEncoder().to(device).eval()
log_step("LSTM encoder loaded: 256-dim embeddings")

# Optionally wrap with hybrid encoder
if USE_HYBRID_ENCODER and HYBRID_ENCODER_AVAILABLE:
    try:
        log_step("\n🔄 Initializing HYBRID ENCODER (ECAPA-TDNN + LSTM)...")
        voice_encoder = HybridVoiceEncoder(
            lstm_encoder=lstm_encoder,
            device=device,
            projection_strength=HYBRID_PROJECTION_STRENGTH,
        ).to(device).eval()
        log_step(f"✅ Hybrid encoder ready (projection strength: {HYBRID_PROJECTION_STRENGTH:.2f})")
        log_step("   This should break past embedding saturation!")
    except Exception as e:
        log_step(f"⚠️  Failed to load hybrid encoder: {e}")
        log_step("   Falling back to standard LSTM encoder")
        voice_encoder = lstm_encoder
else:
    if USE_HYBRID_ENCODER and not HYBRID_ENCODER_AVAILABLE:
        log_step("⚠️  Hybrid encoder requested but not available")
        log_step("   Install: pip install speechbrain")
    voice_encoder = lstm_encoder
    log_step("Using standard LSTM encoder")

log_step("Computing embeddings for source, target, and outputs...")
embed_start = time.time()

source_spk, source_partials = load_embeds_utterance(SOURCE_AUDIO, voice_encoder)
target_spk, target_partials = load_embeds_utterance(TARGET_VOICE_PATH, voice_encoder)
output_preprocessed_spk, output_preprocessed_partials = load_embeds_utterance(out_path, voice_encoder)
output_postprocessed_spk, output_postprocessed_partials = load_embeds_utterance(postprocessed_path, voice_encoder)

embed_elapsed = time.time() - embed_start
log_step(f"Embedding computation completed in {embed_elapsed:.3f}s")

# Partial-level averaged metrics (can be more discriminative)
def mean_pairwise_cos(A: torch.Tensor, B: torch.Tensor):
    # A: (m,E), B: (n,E)
    return float((A @ B.T).mean() / (torch.norm(A, dim=1).mean() * torch.norm(B, dim=1).mean()))

# Baseline metrics
sim_source_target = cosine(source_spk, target_spk)

# Preprocessed output metrics
sim_preproc_source = cosine(output_preprocessed_spk, source_spk)
sim_preproc_target = cosine(output_preprocessed_spk, target_spk)
identity_gain_preproc = sim_preproc_target - sim_preproc_source

partial_cos_preproc_target = mean_pairwise_cos(output_preprocessed_partials, target_partials)
partial_cos_preproc_source = mean_pairwise_cos(output_preprocessed_partials, source_partials)
partial_gain_preproc = partial_cos_preproc_target - partial_cos_preproc_source

spk_l2_preproc_source = l2(output_preprocessed_spk, source_spk)
spk_l2_preproc_target = l2(output_preprocessed_spk, target_spk)

# Postprocessed output metrics
sim_postproc_source = cosine(output_postprocessed_spk, source_spk)
sim_postproc_target = cosine(output_postprocessed_spk, target_spk)
identity_gain_postproc = sim_postproc_target - sim_postproc_source

partial_cos_postproc_target = mean_pairwise_cos(output_postprocessed_partials, target_partials)
partial_cos_postproc_source = mean_pairwise_cos(output_postprocessed_partials, source_partials)
partial_gain_postproc = partial_cos_postproc_target - partial_cos_postproc_source

spk_l2_postproc_source = l2(output_postprocessed_spk, source_spk)
spk_l2_postproc_target = l2(output_postprocessed_spk, target_spk)

print("\n" + "="*80)
print("IDENTITY SHIFT METRICS COMPARISON")
print("="*80)

print("\n[Baseline]")
print(f"Cos(source, target): {sim_source_target:.4f}")

print("\n[Preprocessed Only – Speaker Level]")
print(f"Cos(output, source): {sim_preproc_source:.4f}")
print(f"Cos(output, target): {sim_preproc_target:.4f}")
print(f"Identity gain (target - source): {identity_gain_preproc:.4f}")
print(f"L2(output, source): {spk_l2_preproc_source:.4f}")
print(f"L2(output, target): {spk_l2_preproc_target:.4f}")

print("\n[Preprocessed Only – Partial/Segment Level]")
print(f"Mean partial cos (out vs source): {partial_cos_preproc_source:.4f}")
print(f"Mean partial cos (out vs target): {partial_cos_preproc_target:.4f}")
print(f"Partial identity gain: {partial_gain_preproc:.4f}")

print("\n[Preprocessed + Postprocessed – Speaker Level]")
print(f"Cos(output, source): {sim_postproc_source:.4f}")
print(f"Cos(output, target): {sim_postproc_target:.4f}")
print(f"Identity gain (target - source): {identity_gain_postproc:.4f} [Δ{identity_gain_postproc - identity_gain_preproc:+.4f}]")
print(f"L2(output, source): {spk_l2_postproc_source:.4f}")
print(f"L2(output, target): {spk_l2_postproc_target:.4f}")

print("\n[Preprocessed + Postprocessed – Partial/Segment Level]")
print(f"Mean partial cos (out vs source): {partial_cos_postproc_source:.4f}")
print(f"Mean partial cos (out vs target): {partial_cos_postproc_target:.4f}")
print(f"Partial identity gain: {partial_gain_postproc:.4f} [Δ{partial_gain_postproc - partial_gain_preproc:+.4f}]")

print("\n[IMPROVEMENT SUMMARY]")
print(f"Preprocessing gain: {identity_gain_preproc:.4f}")
print(f"Preprocessing + Postprocessing gain: {identity_gain_postproc:.4f}")
print(f"Additional postprocessing benefit: {identity_gain_postproc - identity_gain_preproc:+.4f}")

if (sim_postproc_target < sim_postproc_source) or (partial_cos_postproc_target < partial_cos_postproc_source):
    print("\n⚠️  WARNING: Output still closer to source than target on at least one metric")
else:
    print("\n✅ SUCCESS: Output closer to target across primary metrics")
    if identity_gain_postproc > 0.15:
        print("   Strong identity shift achieved!")
    elif identity_gain_postproc > 0.08:
        print("   Good identity shift achieved.")
    else:
        print("   Moderate identity shift. Consider tuning preprocessing parameters.")

# Add diagnosis for embedding saturation
if sim_source_target > 0.999:
    print("\n⚠️  EMBEDDING SATURATION DETECTED")
    print(f"   Source/target similarity: {sim_source_target:.4f} (>0.999)")
    print("   The voice encoder sees these speakers as nearly identical.")
    print("   RECOMMENDATION: Preprocessing has limited effect in this case.")
    print("   - Try using a different source or target voice")
    print("   - Focus on model parameters (speaker_strength, prune_tokens, cfg_rate)")
    print("   - The small identity gain may be as good as this model can achieve")
    if USE_HYBRID_ENCODER and HYBRID_ENCODER_AVAILABLE:
        if isinstance(model.s3gen.speaker_encoder, HybridCAMPPlusEncoder) if HybridCAMPPlusEncoder else False:
            print(f"   ✅ Hybrid CAMPPlus encoder is ACTIVE (projection strength: {HYBRID_PROJECTION_STRENGTH:.2f})")
            print(f"      Using ECAPA-TDNN to break past saturation ceiling!")
        else:
            print(f"   💡 TIP: Hybrid encoder failed to load - check speechbrain installation")
    else:
        print("   💡 TIP: Enable USE_HYBRID_ENCODER=True and install speechbrain")

print("\n" + "="*80)
