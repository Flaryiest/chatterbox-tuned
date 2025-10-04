# ⚡ QUICK START - Final Configuration

## **🎯 What's New**

**EXTERNAL PITCH SHIFT** - Applied AFTER voice conversion to avoid quality loss!

---

## **✅ Current Configuration**

```python
# PREPROCESSING
ENABLE_PREPROCESSING = True
PREPROCESSING_STRATEGY = "formant_shift"
FORMANT_STRENGTH = 0.70  # 30% formant compression

# VOICE CONVERSION
SPEAKER_STRENGTH = 1.5   # Maximum target influence
FLOW_CFG_RATE = 0.80     # Strong guidance
ENABLE_PITCH_MATCH = False  # DISABLED (causes quality loss)

# EXTERNAL PITCH SHIFT (NEW!)
ENABLE_EXTERNAL_PITCH_SHIFT = True  # ✅ ENABLED
TARGET_PITCH_HZ = 111  # Barack Obama's pitch
```

---

## **📊 What Will Happen**

1. **Preprocessing:** Formants shifted 30% down (female → male)
2. **Voice Conversion:** High-quality VC (no pitch matching)
3. **External Pitch Shift:** Pitch shifted 185Hz → 111Hz

**Result:** Masculine timbre + masculine pitch + high quality ✅

---

## **🎧 Output Files**

1. `/content/output_preprocessed.wav` - VC only (high pitch)
2. `/content/output_postprocessed.wav` - Same (postprocessing disabled)
3. **`/content/output_final_pitched.wav`** ⭐ **LISTEN TO THIS!**

---

## **🚀 Action**

**Just re-run your notebook!** The external pitch shift will automatically apply.

---

## **🔧 If Needed**

### **Pitch too high/low:**
```python
TARGET_PITCH_HZ = 105  # Deeper
TARGET_PITCH_HZ = 120  # Higher
```

### **Timbre not masculine enough:**
```python
FORMANT_STRENGTH = 0.65  # More aggressive (35% shift)
```

### **Sounds robotic:**
```python
FORMANT_STRENGTH = 0.80  # Gentler (20% shift)
```

---

**That's it! Re-run and enjoy high-quality cross-gender voice conversion!** 🎉
