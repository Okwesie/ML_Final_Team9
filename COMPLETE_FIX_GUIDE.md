# Complete Fix Guide: Make Backend Work Accurately

This guide fixes both the scikit-learn version mismatch AND improves prediction accuracy.

## Problem 1: Version Mismatch
The hybrid model fails to load with error: `No module named '_loss'`

## Problem 2: Reduced Accuracy
The backend uses zeros for TF-IDF and genome embeddings, reducing prediction accuracy.

---

## Solution: Two-Step Fix

### Step 1: Fix Version Mismatch (Required)

Add this cell to your notebook **AFTER** training models:

```python
"""
Re-save Models with Current scikit-learn Version
"""
import joblib
import sklearn

print(f"Current scikit-learn version: {sklearn.__version__}")
print("Re-saving models...")

# Re-save all models with current version
joblib.dump(model1_rf, 'models/advanced_content_based_model.pkl')
joblib.dump(model2_hybrid, 'models/hybrid_collaborative_content_model.pkl')
joblib.dump(scaler_content, 'models/content_scaler.pkl')
joblib.dump(scaler_hybrid, 'models/hybrid_scaler.pkl')
joblib.dump(svd_model, 'models/svd_model.pkl')

print("✓ All models re-saved!")
```

**Run this cell** and verify all models save successfully.

---

### Step 2: Save Features for Accuracy (Recommended)

Add this cell to your notebook **AFTER** feature engineering (after `genome_embeddings_df` is created):

```python
"""
Save Feature Data for Backend Accuracy
"""
import joblib
from pathlib import Path

# Create data directory
Path('data').mkdir(exist_ok=True)

# Save feature cache
feature_cache = {
    'genre_tfidf_df': genre_tfidf_df,
    'genome_embeddings_df': genome_embeddings_df,
    'movies_index': movies_df.set_index('movieId')
}

joblib.dump(feature_cache, 'data/movie_features.pkl')
print("✓ Saved feature cache for backend")
print(f"  - Genre TF-IDF: {genre_tfidf_df.shape}")
print(f"  - Genome embeddings: {genome_embeddings_df.shape}")
```

**Run this cell** to save the features.

---

## Step 3: Restart Backend

After completing both steps:

```bash
# Stop the backend (Ctrl+C if running)
# Then restart:
cd backend
uvicorn main:app --reload
```

---

## Verification

### Check Health Endpoint

```bash
curl http://localhost:8000/health
```

**Expected output:**
```json
{
  "status": "healthy",
  "models_loaded": ["content", "hybrid"],  // ← Both models should be loaded
  "tmdb_service": "active",
  "data_loaded": true
}
```

### Check Backend Logs

When starting the backend, you should see:

```
INFO - Loaded content-based model
INFO - Loaded hybrid model with SVD          // ← Should see this
INFO - Loaded feature cache from data/movie_features.pkl  // ← Should see this
INFO - Pre-computed features for XXXX movies
```

### Test Recommendations

```bash
curl -X POST http://localhost:8000/api/recommend/personalized \
  -H "Content-Type: application/json" \
  -d '{"movie_ids": [1, 2, 3], "model": "hybrid", "top_n": 5}'
```

Should return recommendations without errors.

---

## Troubleshooting

### If hybrid model still doesn't load:

1. **Check scikit-learn version match:**
   ```python
   # In notebook:
   import sklearn
   print(sklearn.__version__)
   ```
   
   ```bash
   # In backend:
   python -c "import sklearn; print(sklearn.__version__)"
   ```
   
   Versions should match. If not, update `backend/requirements.txt` to match.

2. **Re-train the hybrid model:**
   If re-saving doesn't work, re-run the notebook cell that trains `model2_hybrid`, then re-save.

### If features aren't loading:

1. **Check file exists:**
   ```bash
   ls -lh data/movie_features.pkl
   ```

2. **Check backend logs:**
   Look for warnings about feature cache.

3. **Verify notebook variables:**
   Make sure `genre_tfidf_df` and `genome_embeddings_df` exist before saving.

---

## What Changed

### Backend Updates (Already Done)

✅ `model_service.py` now:
- Loads `data/movie_features.pkl` if it exists
- Uses actual TF-IDF vectors instead of zeros
- Uses actual genome embeddings instead of zeros
- Falls back gracefully if cache doesn't exist

### What You Need to Do

1. ✅ Re-save models in notebook (fixes version issue)
2. ✅ Save features in notebook (improves accuracy)
3. ✅ Restart backend

---

## Expected Results

After completing all steps:

- ✅ Hybrid model loads successfully
- ✅ Predictions use actual TF-IDF features
- ✅ Predictions use actual genome embeddings
- ✅ Higher accuracy than using zeros
- ✅ Both models available in `/health` endpoint

---

## Files Created

- `SAVE_FEATURES_FOR_BACKEND.md` - Detailed feature saving instructions
- `FIX_VERSION_ISSUE.md` - Detailed version fix instructions
- `COMPLETE_FIX_GUIDE.md` - This file (complete solution)

All guides are in the project root directory.

