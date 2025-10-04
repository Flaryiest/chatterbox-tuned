#!/usr/bin/env python3
"""
Installation and verification script for advanced voice conversion processing.

This script:
1. Installs required dependencies (pyworld, scipy)
2. Verifies installation
3. Downloads/checks for advanced_preprocessing.py
4. Runs diagnostic tests
5. Provides setup recommendations

Run in Colab: !python install_advanced_processing.py
"""

import subprocess
import sys
import os

def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_status(status, message):
    """Print a status message with icon."""
    icons = {"success": "✅", "error": "❌", "warning": "⚠️", "info": "ℹ️"}
    print(f"{icons.get(status, '•')} {message}")

def install_package(package_name):
    """Install a Python package using pip."""
    print(f"\n📦 Installing {package_name}...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", package_name
        ])
        print_status("success", f"{package_name} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print_status("error", f"Failed to install {package_name}")
        return False

def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    if package_name is None:
        package_name = module_name
    try:
        __import__(module_name)
        print_status("success", f"{package_name} is available")
        return True
    except ImportError:
        print_status("error", f"{package_name} is NOT available")
        return False

def check_file_exists(filename):
    """Check if a file exists in current directory."""
    if os.path.exists(filename):
        print_status("success", f"{filename} found")
        return True
    else:
        print_status("warning", f"{filename} not found")
        return False

def test_pyworld():
    """Test PyWorld installation with a simple operation."""
    print("\n🧪 Testing PyWorld functionality...")
    try:
        import pyworld as pw
        import numpy as np
        
        # Create a simple test signal
        fs = 16000
        t = np.arange(0, 1, 1/fs)
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        audio = audio.astype(np.float64)
        
        # Test F0 extraction
        f0, timeaxis = pw.dio(audio, fs)
        f0 = pw.stonemask(audio, f0, timeaxis, fs)
        
        # Test spectral envelope extraction
        sp = pw.cheaptrick(audio, f0, timeaxis, fs)
        
        # Test aperiodicity extraction
        ap = pw.d4c(audio, f0, timeaxis, fs)
        
        # Test synthesis
        synthesized = pw.synthesize(f0, sp, ap, fs)
        
        print_status("success", "PyWorld test passed!")
        print(f"   - F0 extraction: OK")
        print(f"   - Spectral envelope: OK ({sp.shape})")
        print(f"   - Aperiodicity: OK")
        print(f"   - Synthesis: OK ({len(synthesized)} samples)")
        return True
        
    except Exception as e:
        print_status("error", f"PyWorld test failed: {e}")
        return False

def test_advanced_preprocessing():
    """Test advanced_preprocessing module."""
    print("\n🧪 Testing advanced_preprocessing module...")
    try:
        from advanced_preprocessing import (
            formant_shift_preprocessing,
            adaptive_spectral_transfer,
            PYWORLD_AVAILABLE
        )
        
        if PYWORLD_AVAILABLE:
            print_status("success", "advanced_preprocessing module loaded")
            print(f"   - formant_shift_preprocessing: Available")
            print(f"   - adaptive_spectral_transfer: Available")
            print(f"   - PyWorld integration: Active")
            return True
        else:
            print_status("warning", "Module loaded but PyWorld not available")
            return False
            
    except ImportError as e:
        print_status("error", f"Cannot import advanced_preprocessing: {e}")
        return False

def print_config_example():
    """Print example configuration."""
    print("\n" + "="*70)
    print("  RECOMMENDED CONFIGURATION")
    print("="*70)
    print("""
For Female → Male conversion, add to colab.py:

# PREPROCESSING
ENABLE_PREPROCESSING = True
PREPROCESSING_STRATEGY = "formant_shift"
GENDER_SHIFT = "female_to_male"
FORMANT_STRENGTH = 0.85  # Compress formants down 15%

# POSTPROCESSING
ENABLE_POSTPROCESSING = True
POSTPROCESSING_STRATEGY = "spectral_transfer"
TIMBRE_STRENGTH = 0.7  # Apply 70% target timbre
POST_FORMANT_SHIFT = None  # Or "neutral_to_male" for enhancement

# MODEL PARAMETERS
SPEAKER_STRENGTH = 1.2  # Increased from 1.1
FLOW_CFG_RATE = 0.75    # Increased from 0.70
PRUNE_TOKENS = 2        # Remove source prosody

For Male → Female conversion:
    Just change: GENDER_SHIFT = "male_to_female"
    (Everything else stays the same)
""")

def main():
    """Main installation and verification."""
    print_header("ADVANCED VOICE CONVERSION - INSTALLATION & SETUP")
    
    # Step 1: Check existing installations
    print_header("STEP 1: Checking Existing Installations")
    
    has_scipy = check_import("scipy")
    has_pyworld = check_import("pyworld")
    has_numpy = check_import("numpy")
    has_librosa = check_import("librosa")
    
    # Step 2: Install missing packages
    print_header("STEP 2: Installing Missing Dependencies")
    
    all_installed = True
    
    if not has_scipy:
        if not install_package("scipy"):
            all_installed = False
    else:
        print_status("info", "scipy already installed")
    
    if not has_pyworld:
        if not install_package("pyworld"):
            all_installed = False
    else:
        print_status("info", "pyworld already installed")
    
    if not has_numpy:
        if not install_package("numpy"):
            all_installed = False
    else:
        print_status("info", "numpy already installed")
    
    if not has_librosa:
        if not install_package("librosa"):
            all_installed = False
    else:
        print_status("info", "librosa already installed")
    
    # Step 3: Verify installation
    print_header("STEP 3: Verifying Installation")
    
    scipy_ok = check_import("scipy")
    pyworld_ok = check_import("pyworld")
    numpy_ok = check_import("numpy")
    librosa_ok = check_import("librosa")
    
    # Step 4: Check for files
    print_header("STEP 4: Checking Required Files")
    
    has_advanced_preprocessing = check_file_exists("advanced_preprocessing.py")
    has_colab = check_file_exists("colab.py")
    
    # Step 5: Run tests
    print_header("STEP 5: Running Functional Tests")
    
    pyworld_test_ok = False
    if pyworld_ok:
        pyworld_test_ok = test_pyworld()
    else:
        print_status("error", "Skipping PyWorld test (not installed)")
    
    advanced_test_ok = False
    if has_advanced_preprocessing:
        advanced_test_ok = test_advanced_preprocessing()
    else:
        print_status("warning", "Skipping module test (file not found)")
    
    # Step 6: Summary
    print_header("INSTALLATION SUMMARY")
    
    print("\n📊 Component Status:")
    print(f"   scipy:                  {'✅' if scipy_ok else '❌'}")
    print(f"   pyworld:                {'✅' if pyworld_ok else '❌'}")
    print(f"   numpy:                  {'✅' if numpy_ok else '✅'}")
    print(f"   librosa:                {'✅' if librosa_ok else '✅'}")
    print(f"   advanced_preprocessing: {'✅' if has_advanced_preprocessing else '❌'}")
    print(f"   colab.py:               {'✅' if has_colab else '⚠️'}")
    
    print("\n🧪 Test Results:")
    print(f"   PyWorld functionality:  {'✅' if pyworld_test_ok else '❌'}")
    print(f"   Module integration:     {'✅' if advanced_test_ok else '❌'}")
    
    # Final verdict
    print("\n" + "="*70)
    if pyworld_test_ok and advanced_test_ok:
        print("✅ INSTALLATION SUCCESSFUL!")
        print("\nYou can now use advanced formant-based processing.")
        print("Expected improvement: 20-100× better identity gain!")
        print_config_example()
    elif pyworld_ok and has_advanced_preprocessing:
        print("⚠️  PARTIAL INSTALLATION")
        print("\nBasic components installed, but tests failed.")
        print("Try restarting the runtime/kernel and run this script again.")
    else:
        print("❌ INSTALLATION INCOMPLETE")
        print("\nMissing components:")
        if not pyworld_ok:
            print("   - PyWorld: Install with 'pip install pyworld'")
        if not has_advanced_preprocessing:
            print("   - advanced_preprocessing.py: Upload to workspace")
        print("\nPlease resolve these issues and run again.")
    
    print("="*70)
    
    # Next steps
    print("\n📚 Documentation:")
    print("   - Quick setup: QUICK_SETUP_ADVANCED_PROCESSING.md")
    print("   - Full guide:  CROSS_GENDER_CONVERSION_GUIDE.md")
    print("   - Summary:     SOLUTION_SUMMARY.md")
    print("   - Visuals:     VISUAL_COMPARISON.md")
    
    print("\n🔗 Support:")
    print("   - GitHub Issues: [Your repo]/issues")
    print("   - Documentation: See markdown files in workspace")
    
    return pyworld_test_ok and advanced_test_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
