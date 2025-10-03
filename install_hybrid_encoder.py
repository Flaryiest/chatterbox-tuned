"""
Quick installation script for Hybrid Voice Encoder dependencies.

Run this in your Colab notebook BEFORE running colab.py:

!python install_hybrid_encoder.py

Or manually:

!pip install speechbrain

"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip."""
    print(f"\n{'='*60}")
    print(f"Installing {package}...")
    print('='*60)
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print(f"✅ {package} installed successfully!\n")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("HYBRID VOICE ENCODER - INSTALLATION")
    print("="*60)
    print("\nThis will install:")
    print("  - speechbrain (for ECAPA-TDNN encoder)")
    print("  - Required dependencies (torch, torchaudio, etc.)")
    print("\nEstimated download size: ~200MB")
    print("First run will also download pretrained models: ~80MB")
    print("="*60)
    
    try:
        # Install SpeechBrain
        install_package("speechbrain")
        
        print("\n" + "="*60)
        print("✅ INSTALLATION COMPLETE!")
        print("="*60)
        print("\nHybrid encoder is ready to use!")
        print("\nIn colab.py, set:")
        print("  USE_HYBRID_ENCODER = True")
        print("  HYBRID_PROJECTION_STRENGTH = 0.4")
        print("\nThen run your voice conversion script.")
        print("="*60 + "\n")
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ INSTALLATION FAILED")
        print("="*60)
        print(f"\nError: {e}")
        print("\nTry manual installation:")
        print("  !pip install speechbrain")
        print("="*60 + "\n")
        sys.exit(1)
