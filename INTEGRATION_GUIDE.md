# Model Integration Guide

## Quick Summary: What Your Models Can Do

### ✅ **Rating Prediction**
Predict how much a user will rate any movie (0.5-5.0 scale)

### ✅ **Top-N Recommendations**  
Rank all movies by predicted rating and return top N

### ✅ **Similar Movies Discovery**
Find movies similar to a given movie using content features

### ✅ **Cold-Start Handling**
Recommend movies to new users who haven't rated anything yet

### ✅ **Personalized Recommendations**
Leverage user rating history for better accuracy (Hybrid model)

## Model Files Available

```
models/
├── advanced_content_based_model.pkl      ← Model 1 (Random Forest)
├── hybrid_collaborative_content_model.pkl ← Model 2 (Gradient Boosting)
├── svd_model.pkl                        ← Collaborative filtering component
├── content_scaler.pkl                   ← Feature scaler for Model 1
├── hybrid_scaler.pkl                    ← Feature scaler for Model 2
└── feature_names.json                   ← Feature definitions
```

## How to Use Models in Your Web Platform

### Option 1: Update Streamlit App (app.py)

The current `app.py` tries to load old models. Update it to use your new models:

```python
@st.cache_resource
def load_models():
    """Load trained models from the models directory"""
    models = {}
    models_dir = Path('models')
    
    try:
        # Load Model 1: Advanced Content-Based
        if (models_dir / 'advanced_content_based_model.pkl').exists():
            models['content_based'] = joblib.load(
                models_dir / 'advanced_content_based_model.pkl'
            )
            models['content_scaler'] = joblib.load(
                models_dir / 'content_scaler.pkl'
            )
        
        # Load Model 2: Hybrid
        if (models_dir / 'hybrid_collaborative_content_model.pkl').exists():
            models['hybrid'] = joblib.load(
                models_dir / 'hybrid_collaborative_content_model.pkl'
            )
            models['hybrid_scaler'] = joblib.load(
                models_dir / 'hybrid_scaler.pkl'
            )
            models['svd'] = joblib.load(
                models_dir / 'svd_model.pkl'
            )
        
        # Load feature names
        if (models_dir / 'feature_names.json').exists():
            with open(models_dir / 'feature_names.json', 'r') as f:
                models['feature_names'] = json.load(f)
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}
```

### Option 2: Create Backend API (for React Frontend)

Create a Flask/FastAPI backend:

```python
# backend/api.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import json

app = Flask(__name__)

# Load models once at startup
models = {}
models_dir = Path('models')

# Load Model 1
models['content_model'] = joblib.load(
    models_dir / 'advanced_content_based_model.pkl'
)
models['content_scaler'] = joblib.load(
    models_dir / 'content_scaler.pkl'
)

# Load Model 2
models['hybrid_model'] = joblib.load(
    models_dir / 'hybrid_collaborative_content_model.pkl'
)
models['hybrid_scaler'] = joblib.load(
    models_dir / 'hybrid_scaler.pkl'
)
models['svd'] = joblib.load(
    models_dir / 'svd_model.pkl'
)

# Load feature names
with open(models_dir / 'feature_names.json', 'r') as f:
    models['feature_names'] = json.load(f)

@app.route('/api/predict-rating', methods=['POST'])
def predict_rating():
    """Predict rating for user-movie pair"""
    data = request.json
    user_id = data.get('user_id')
    movie_id = data.get('movie_id')
    model_type = data.get('model_type', 'hybrid')  # 'content' or 'hybrid'
    
    # Prepare features (you'll need to implement this)
    features = prepare_features(user_id, movie_id)
    
    # Select model
    if model_type == 'content':
        scaler = models['content_scaler']
        model = models['content_model']
    else:
        scaler = models['hybrid_scaler']
        model = models['hybrid_model']
    
    # Scale and predict
    features_scaled = scaler.transform([features])
    predicted_rating = float(model.predict(features_scaled)[0])
    
    return jsonify({
        'predicted_rating': predicted_rating,
        'confidence': calculate_confidence(predicted_rating)
    })

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """Get top N recommendations for a user"""
    data = request.json
    user_id = data.get('user_id')
    n = data.get('n', 10)
    model_type = data.get('model_type', 'hybrid')
    
    # Get all movies user hasn't rated
    unrated_movies = get_unrated_movies(user_id)
    
    # Predict ratings for all
    predictions = []
    for movie_id in unrated_movies[:1000]:  # Limit for performance
        features = prepare_features(user_id, movie_id)
        # ... predict and append
    
    # Sort and return top N
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    return jsonify({
        'recommendations': predictions[:n]
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

## Feature Preparation Function

You'll need to implement `prepare_features()` that creates the feature vector:

```python
def prepare_features(user_id, movie_id, ratings_df, movies_df, 
                    user_features_df, movie_features_df,
                    genre_tfidf_df, genome_embeddings_df):
    """
    Prepare feature vector for prediction
    
    This replicates the feature engineering from the notebook
    """
    # Get user features
    user_features = user_features_df[user_features_df['userId'] == user_id].iloc[0]
    
    # Get movie features
    movie_features = movie_features_df[movie_features_df['movieId'] == movie_id].iloc[0]
    
    # Get TF-IDF features
    genre_tfidf = genre_tfidf_df.loc[movie_id].values
    
    # Get genome embeddings
    genome_emb = genome_embeddings_df.loc[movie_id].values
    
    # Combine all features in correct order (matching feature_names.json)
    features = [
        user_features['user_avg_rating'],
        user_features['user_rating_std'],
        # ... all other features in order
    ] + list(genre_tfidf) + list(genome_emb)
    
    return np.array(features)
```

## Integration with Your React Platform

Your React platform can call these APIs:

```javascript
// In your React component
const getRecommendations = async (selectedMovies) => {
  const response = await fetch('http://localhost:5000/api/recommendations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: currentUser.id,
      n: 20,
      model_type: 'hybrid'
    })
  });
  
  const data = await response.json();
  setRecommendations(data.recommendations);
};

// Rating prediction
const predictRating = async (movieId) => {
  const response = await fetch('http://localhost:5000/api/predict-rating', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: currentUser.id,
      movie_id: movieId,
      model_type: 'hybrid'
    })
  });
  
  const data = await response.json();
  return data.predicted_rating;
};
```

## Use Cases in Your Platform

### 1. **Rating Predictor** (Advanced ML Tab)
```python
predicted_rating = predict_rating(user_id, movie_id)
# Display: "You'll rate this 4.2/5.0 ⭐"
```

### 2. **Personalized Recommendations** (Main Tab)
```python
recommendations = get_recommendations(user_id, n=20)
# Show top 20 movies ranked by predicted rating
```

### 3. **Mood-Based Search**
```python
# Filter movies by genre, then rank by predicted rating
mood_movies = filter_by_genre(genre)
ranked = rank_by_predicted_rating(user_id, mood_movies)
```

### 4. **Challenge Mode**
```python
# Find high-rated movies from unexplored genres
unexplored_genres = get_unexplored_genres(user_id)
challenge_movies = filter_by_genres(unexplored_genres)
ranked = rank_by_predicted_rating(user_id, challenge_movies)
```

### 5. **Similar Movies**
```python
# Use content-based features to find similar movies
similar = find_similar_movies(movie_id, n=10)
```

## Model Selection Logic

```python
def select_best_model(user_id):
    """Select best model based on user history"""
    rating_count = get_user_rating_count(user_id)
    
    if rating_count < 10:
        return 'content'  # Cold-start: use content-based
    else:
        return 'hybrid'   # Warm user: use hybrid for better accuracy
```

## Performance Tips

1. **Pre-compute features** for all movies at startup
2. **Cache predictions** for popular movies
3. **Batch predictions** when possible
4. **Use Model 1** for new users (faster, handles cold-start)
5. **Use Model 2** for users with history (better accuracy)

## Next Steps

1. ✅ Models are trained and saved
2. ⏳ Update `app.py` to load new models
3. ⏳ Implement feature preparation function
4. ⏳ Create prediction API endpoints
5. ⏳ Integrate with React frontend
6. ⏳ Test end-to-end flow

