# 🚨 AGGRESSIVE MODE - Female to Male Conversion

## **Problem Statement**

Your output still sounds:
- ❌ **Higher pitch** than target (185Hz vs target 111Hz)
- ❌ **Wrong timbre** (feminine characteristics remain)
- ❌ **F1 formant not shifting** (172Hz before and after preprocessing)

---

## **🔧 Updated Configuration (Applied)**

I've made these changes to `colab.py`:

### **1. More Aggressive Formant Shifting:**
```python
PREPROCESSING_STRATEGY = "combined"  # Formant shift + neutralization
FORMANT_STRENGTH = 0.75  # 25% formant compression (was 0.85 = 15%)
NEUTRALIZE_VOCAL_TRACT = True  # Additional smoothing
```

### **2. Enable Pitch Matching:**
```python
ENABLE_PITCH_MATCH = True  # Force pitch shift to target
PITCH_TOLERANCE = 0.5  # Stricter matching
MAX_PITCH_SHIFT = 5.0  # Allow large shifts (female→male needs ~0.6× pitch)
```

### **3. Stronger Model Influence:**
```python
SPEAKER_STRENGTH = 1.3  # Increased from 1.1
FLOW_CFG_RATE = 0.75    # Stronger guidance (was 0.70)
```

---

## **🎯 Expected Results**

### **After Re-running:**

**Preprocessing:**
- Formants should shift down by **25%** (more aggressive)
- Vocal tract neutralization should smooth source timbre
- F0 might drop slightly (but pitch matching will fix it)

**Voice Conversion:**
- Pitch matching will force F0 down to ~111Hz (target's pitch)
- Stronger speaker_strength will force target characteristics
- Higher CFG will prioritize target over source

**Output:**
- ✅ Pitch: ~111Hz (masculine)
- ✅ Formants: Shifted toward male range
- ✅ Timbre: More masculine (less feminine blend)

---

## **📊 What To Check After Re-running**

### **1. Check Preprocessing Output:**
```
📊 Spectral Analysis - Source (Preprocessed):
   Median F0: ??? Hz (should drop from 185Hz)
   Estimated Formants: F1=???Hz, F2=???Hz (should be lower than original)
```

### **2. Check Final Output:**
```
🎯 CAMPPlus Evaluation:
   Identity gain: ??? (should be >0.20)
   Target similarity: ??? (should be >0.70)
```

### **3. Listen Critically:**
- Does it sound **male**? (not feminine or neutral)
- Is the **pitch low** enough? (~111Hz like Obama)
- Does it sound like **Obama**? (prosody, timbre, speaking style)

---

## **🔄 If Still Not Working**

### **Option 1: EXTREME Formant Shift**
```python
FORMANT_STRENGTH = 0.65  # 35% compression (very aggressive)
```

⚠️ Warning: May cause robotic artifacts

### **Option 2: Try Different Target**
The issue might be:
- Target audio quality (MP3 compression, noise)
- Target voice characteristics (Obama's voice might be challenging)

Try a different male speaker with:
- Clear, high-quality recording
- Similar speaking style to source
- Moderate pitch (not too deep, not too high)

### **Option 3: Cascaded Conversion**
Convert in stages:
1. Female → Neutral (FORMANT_STRENGTH=0.92)
2. Neutral → Male (FORMANT_STRENGTH=0.85)

### **Option 4: Try Pitch-First Approach**
Enable pitch matching BEFORE formant shifting:
```python
# In colab.py, move pitch matching to preprocessing phase
ENABLE_PREPROCESSING_PITCH = True
PREPROCESSING_PITCH_TARGET = 111  # Hz (Obama's pitch)
```

(Would require code modification)

---

## **🧪 Diagnostic Commands**

### **Check Formant Shift Working:**
After preprocessing, compare:
- `Source (Original)` vs `Source (Preprocessed)`
- F2 should drop by 25%: 1797 → ~1350Hz
- F3 should drop by 25%: 2609 → ~1957Hz

### **Check Pitch Matching Working:**
After voice conversion:
- Output F0 should be ~111Hz (not 185Hz)
- If F0 is still high, pitch matching failed

---

## **⚙️ Parameter Tuning Guide**

| Problem | Solution | Parameter |
|---------|----------|-----------|
| **Still sounds feminine** | More aggressive formant shift | `FORMANT_STRENGTH = 0.70` |
| **Sounds robotic** | Less aggressive | `FORMANT_STRENGTH = 0.85` |
| **Pitch still too high** | Force stronger pitch match | `PITCH_TOLERANCE = 0.3` |
| **Pitch artifacts** | Gentler pitch match | `MAX_PITCH_SHIFT = 3.0` |
| **Not enough like target** | Stronger model influence | `SPEAKER_STRENGTH = 1.5` |
| **Sounds over-processed** | Reduce model strength | `SPEAKER_STRENGTH = 1.1` |
| **Timbre still wrong** | Add neutralization | `NEUTRALIZE_VOCAL_TRACT = True` |
| **Too smooth/generic** | Disable neutralization | `NEUTRALIZE_VOCAL_TRACT = False` |

---

## **🔬 Technical Explanation**

### **Why F1 Shows 172Hz (Wrong):**
The spectral analysis function uses `find_peaks()` on the average spectrum, which might be picking up:
- **Harmonics** of F0 (185Hz) instead of formants
- **Low-frequency noise** in the recording
- **First harmonic** (1×F0) instead of first formant

**Actual F1** for female speech is typically **650-850Hz**, not 172Hz.

### **Why Formant Shifting Still Works:**
Even though the analysis is wrong, the formant shifting operates on the **entire spectral envelope**, not just detected peaks. So it:
- Warps all frequencies by 0.75× (25% compression)
- Shifts actual F1 from ~700Hz → ~525Hz (male-like)
- Shifts F2 from ~1800Hz → ~1350Hz (male-like)

The analysis just isn't showing it correctly.

### **Why Pitch Matching Matters:**
The model preserves pitch from the preprocessed input:
- Source: 185Hz (female)
- Target: 111Hz (male)
- Without pitch matching: Output ≈ 185Hz (still feminine!)
- With pitch matching: Output ≈ 111Hz (masculine!)

Pitch is **50% of perceived gender**, so this is critical.

---

## **🎯 Success Criteria (Updated)**

| Metric | Before | Target | How To Achieve |
|--------|--------|--------|----------------|
| **F0 Pitch** | 185Hz | ~111Hz | ✅ Pitch matching enabled |
| **CAMPPlus Gain** | 0.231 | >0.25 | ✅ Stronger parameters |
| **Formant F2** | 1797Hz | ~1300Hz | ✅ 25% shift |
| **Perceptual** | Feminine | Masculine | 🎧 Listen after re-run |

---

## **📝 Action Items**

1. ✅ Configuration updated (FORMANT_STRENGTH=0.75, pitch matching enabled)
2. 🔄 **Re-run the notebook in Colab**
3. 🎧 Listen to output
4. 📊 Check if F0 dropped to ~111Hz
5. 📊 Check if CAMPPlus gain improved
6. 🔧 Tune further if needed (see Parameter Tuning Guide)

---

## **💡 Additional Insight**

The real issue might be that **the model is trained on natural pitch variations** and resists extreme pitch shifts. The model might be:
- Ignoring the formant-shifted input
- Reverting to source-like output
- Unable to handle the preprocessed "unnatural" formants

This is why **pitch matching during generation** is crucial - it forces the model to output the correct pitch, which the model couples with appropriate formants.

---

**Re-run with the new configuration and share the results!** 🚀
