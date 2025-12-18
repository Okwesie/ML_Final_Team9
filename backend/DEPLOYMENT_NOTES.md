# Deployment Notes

## Current Implementation Status

✅ **Fully Functional**: All endpoints work and return predictions
⚠️ **Simplified Features**: TF-IDF and genome embeddings use placeholder values

## What Works Now

- Personalized recommendations (based on selected movies)
- Similar movies discovery
- Rating predictions
- Movie search
- TMDB integration (posters, descriptions)

## Accuracy Considerations

The current implementation uses simplified feature preparation:
- User features: ✅ Extracted from selected movies
- Movie features: ✅ Pre-computed from ratings
- TF-IDF features: ⚠️ Currently zeros (50 features)
- Genome embeddings: ⚠️ Currently zeros (50 features)

**Impact**: Predictions will work but may be ~10-20% less accurate than full implementation.

## To Improve Accuracy

1. In notebook, after feature engineering, save TF-IDF and genome features:
```python
import joblib
feature_cache = {
    'genre_tfidf': genre_tfidf_df,
    'genome_embeddings': genome_embeddings_df,
    'movies_index': movies_df.set_index('movieId')
}
joblib.dump(feature_cache, 'data/movie_features.pkl')
```

2. In `model_service.py`, load these in `_precompute_features()`:
```python
features_file = self.data_dir / "movie_features.pkl"
if features_file.exists():
    feature_cache = joblib.load(features_file)
    self.genre_tfidf_df = feature_cache['genre_tfidf']
    self.genome_embeddings_df = feature_cache['genome_embeddings']
```

3. Use in `_prepare_content_features()` instead of zeros.

## Quick Start

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add TMDB_API_KEY
uvicorn main:app --reload
```

## Testing

```bash
python test_api.py
```

All tests should pass if models are loaded correctly.
