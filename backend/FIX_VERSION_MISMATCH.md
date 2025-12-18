# Fixing "No module named '_loss'" Error

## Problem

The error occurs because the hybrid model (`hybrid_collaborative_content_model.pkl`) was saved with a different version of scikit-learn than what's currently installed.

## Solution Options

### Option 1: Use Content-Based Model Only (Quick Fix)

The backend now automatically falls back to the content-based model if the hybrid model fails to load. **Your API will still work**, just using Model 1 instead of Model 2.

**Status**: ✅ Already implemented - backend will use content-based model as fallback

### Option 2: Re-save the Hybrid Model (Proper Fix)

Re-save the hybrid model in the notebook with the current scikit-learn version:

1. **Check current scikit-learn version:**
```python
import sklearn
print(sklearn.__version__)
```

2. **In the notebook, after training Model 2, re-save it:**
```python
# Re-save the hybrid model (this will use current scikit-learn version)
import joblib
joblib.dump(model2_hybrid, 'models/hybrid_collaborative_content_model.pkl')
joblib.dump(scaler_hybrid, 'models/hybrid_scaler.pkl')
```

3. **Restart the backend server**

### Option 3: Match scikit-learn Versions

Install the same scikit-learn version that was used to train the model:

```bash
# If you know the version (check notebook output)
pip install scikit-learn==1.3.2

# Or check what version the notebook used
# Then install that specific version
```

## Current Status

✅ **Content-based model loads successfully**  
❌ **Hybrid model fails to load** (version mismatch)  
✅ **Backend falls back to content-based model automatically**

## Impact

- **API still works** - all endpoints function
- **Recommendations work** - using content-based model
- **Slightly lower accuracy** - hybrid model typically performs ~2% better
- **No SVD component** - collaborative filtering not available

## Verification

Check the health endpoint:
```bash
curl http://localhost:8000/health
```

You should see:
```json
{
  "status": "healthy",
  "models_loaded": ["content"],  // Only content model
  ...
}
```

## Recommendation

For now, **the backend works fine with just the content-based model**. The fallback is automatic. You can fix the hybrid model later by re-saving it in the notebook.

