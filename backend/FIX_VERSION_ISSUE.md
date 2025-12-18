# Fix scikit-learn Version Mismatch

The hybrid model was saved with a different scikit-learn version, causing the `_loss` module error.

## Quick Fix: Re-save Models in Notebook

Add this cell to your notebook **AFTER** training the models:

```python
"""
Re-save Models with Current scikit-learn Version
This fixes version compatibility issues
"""
import joblib
import sklearn

print(f"Current scikit-learn version: {sklearn.__version__}")
print("Re-saving models with current version...")

# Re-save all models
try:
    joblib.dump(model1_rf, 'models/advanced_content_based_model.pkl')
    print("✓ Re-saved: advanced_content_based_model.pkl")
except Exception as e:
    print(f"✗ Error saving content model: {e}")

try:
    joblib.dump(model2_hybrid, 'models/hybrid_collaborative_content_model.pkl')
    print("✓ Re-saved: hybrid_collaborative_content_model.pkl")
except Exception as e:
    print(f"✗ Error saving hybrid model: {e}")

try:
    joblib.dump(scaler_content, 'models/content_scaler.pkl')
    print("✓ Re-saved: content_scaler.pkl")
except Exception as e:
    print(f"✗ Error saving content scaler: {e}")

try:
    joblib.dump(scaler_hybrid, 'models/hybrid_scaler.pkl')
    print("✓ Re-saved: hybrid_scaler.pkl")
except Exception as e:
    print(f"✗ Error saving hybrid scaler: {e}")

try:
    joblib.dump(svd_model, 'models/svd_model.pkl')
    print("✓ Re-saved: svd_model.pkl")
except Exception as e:
    print(f"✗ Error saving SVD model: {e}")

print("\n✓ All models re-saved successfully!")
print("\nNow restart your backend server.")
```

## Alternative: Match Versions

If re-saving doesn't work, ensure the backend uses the same scikit-learn version as the notebook:

1. Check notebook version:
```python
import sklearn
print(sklearn.__version__)
```

2. Update backend requirements.txt to match:
```txt
scikit-learn==1.3.2  # Use the exact version from notebook
```

3. Reinstall:
```bash
pip install -r requirements.txt --force-reinstall
```

## Verify Fix

After re-saving models, restart the backend and check:

```bash
curl http://localhost:8000/health
```

You should see:
```json
{
  "status": "healthy",
  "models_loaded": ["content", "hybrid"],
  ...
}
```

If hybrid model still doesn't load, check the backend logs for the specific error.

