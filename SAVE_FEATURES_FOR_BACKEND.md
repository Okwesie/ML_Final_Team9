# Save Features for Backend Accuracy

To make the backend work accurately, you need to save the TF-IDF and genome embeddings from the notebook.

## Step 1: Add This Cell to Your Notebook

Add this cell **AFTER** the feature engineering cells (after `genome_embeddings_df` is created) and **BEFORE** model training:

```python
"""
Save Feature Data for Backend
This ensures the backend can use actual TF-IDF and genome embeddings instead of zeros
"""
import joblib
from pathlib import Path

# Create data directory if it doesn't exist
Path('data').mkdir(exist_ok=True)

# Save feature cache
feature_cache = {
    'genre_tfidf_df': genre_tfidf_df,
    'genome_embeddings_df': genome_embeddings_df,
    'movies_index': movies_df.set_index('movieId')
}

joblib.dump(feature_cache, 'data/movie_features.pkl')
print("✓ Saved feature cache for backend")
print(f"  - Genre TF-IDF shape: {genre_tfidf_df.shape}")
print(f"  - Genome embeddings shape: {genome_embeddings_df.shape}")
print(f"  - Saved to: data/movie_features.pkl")
```

## Step 2: Re-save Models with Current scikit-learn Version

After training models, add this cell to re-save them with the current version:

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
models_to_save = {
    'advanced_content_based_model.pkl': model1_rf,
    'hybrid_collaborative_content_model.pkl': model2_hybrid,
    'content_scaler.pkl': scaler_content,
    'hybrid_scaler.pkl': scaler_hybrid,
    'svd_model.pkl': svd_model
}

for filename, model in models_to_save.items():
    try:
        joblib.dump(model, f'models/{filename}')
        print(f"✓ Re-saved: {filename}")
    except Exception as e:
        print(f"✗ Error saving {filename}: {e}")

print("\n✓ All models re-saved successfully!")
```

## Step 3: Restart Backend

After saving features and re-saving models:

1. Stop the backend server (Ctrl+C)
2. Restart it: `uvicorn main:app --reload`
3. Check `/health` endpoint - hybrid model should load successfully

## Verification

The backend will automatically:
- Load `data/movie_features.pkl` if it exists
- Use actual TF-IDF and genome embeddings instead of zeros
- Provide accurate predictions

Check the logs when starting the backend - you should see:
```
INFO - Loaded feature cache from data/movie_features.pkl
INFO - Loaded hybrid model with SVD
```

