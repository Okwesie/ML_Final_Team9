# ML Model Capabilities & System Integration Guide

## What Your Trained Models Can Do

### Model 1: Advanced Content-Based System (Random Forest)
**File**: `models/advanced_content_based_model.pkl`

**Capabilities**:
1. **Rating Prediction**: Predicts how much a user will rate a movie (0.5-5.0 scale)
   - Uses 97+ features including:
     - User behavior (rating variance, genre diversity, activity patterns)
     - Movie characteristics (release year, genre popularity, rating momentum)
     - Temporal features (movie age at rating, user lifecycle stage)
     - TF-IDF embeddings (50 genre features)
     - Genome embeddings (50 tag-genome features)

2. **Top-N Recommendations**: Rank movies by predicted rating for a user
   - Handles cold-start (new users/movies) using content features
   - Can filter by genre, year, or other criteria

3. **Similar Movies Discovery**: Find movies similar to a given movie
   - Uses content-based similarity (genres, tags, features)

**Performance**:
- Test RMSE: 0.791
- Test MAE: 0.584
- R²: 0.392
- Precision@10: 1.0

### Model 2: Hybrid Collaborative-Content Model (SVD + Gradient Boosting)
**Files**: 
- `models/hybrid_collaborative_content_model.pkl`
- `models/svd_model.pkl`

**Capabilities**:
1. **Rating Prediction**: Combines collaborative filtering with content features
   - SVD component: Learns user-movie interaction patterns
   - Content component: Handles cold-start scenarios
   - Best for users with rating history

2. **Personalized Recommendations**: Leverages both user similarity and content similarity
   - Better for users with existing ratings
   - Falls back to content features for new users

**Performance**:
- Test RMSE: 0.789 (slightly better than Model 1)
- Test MAE: 0.581
- R²: 0.395
- Precision@10: 1.0

## How Models Fit Into Your System

### Current Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Jupyter Notebook                         │
│  (Training & Feature Engineering)                           │
│                                                             │
│  • Loads data (ratings.csv, movies.csv, etc.)              │
│  • Feature engineering (TF-IDF, genome, temporal)          │
│  • Trains Model 1 & Model 2                                 │
│  • Saves models to models/ directory                        │
│  • Generates outputs (hidden_gems.csv, etc.)                │
└──────────────────────┬────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              models/ & outputs/ Directories                 │
│                                                             │
│  models/                                                    │
│  ├── advanced_content_based_model.pkl  ← Model 1           │
│  ├── hybrid_collaborative_content_model.pkl ← Model 2      │
│  ├── svd_model.pkl                    ← CF component       │
│  ├── content_scaler.pkl               ← Feature scaler     │
│  ├── hybrid_scaler.pkl                ← Feature scaler     │
│  └── feature_names.json                ← Feature list      │
│                                                             │
│  outputs/                                                   │
│  ├── model_comparison.csv                                  │
│  ├── hidden_gems.csv                                       │
│  └── executive_summary.json                                 │
└──────────────────────┬────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Web Application (Streamlit/React)              │
│                                                             │
│  • Loads models using joblib                                │
│  • Processes user input                                     │
│  • Generates predictions                                    │
│  • Displays recommendations                                 │
└─────────────────────────────────────────────────────────────┘
```

### Integration Points

#### 1. **Rating Prediction API**
```python
def predict_rating(user_id, movie_id, model_type='hybrid'):
    """
    Predict how much a user will rate a movie
    
    Input:
    - user_id: User identifier
    - movie_id: Movie identifier
    - model_type: 'content' or 'hybrid'
    
    Output:
    - predicted_rating: Float between 0.5-5.0
    - confidence: Prediction confidence score
    """
    # Load appropriate model
    if model_type == 'content':
        model = joblib.load('models/advanced_content_based_model.pkl')
        scaler = joblib.load('models/content_scaler.pkl')
    else:
        model = joblib.load('models/hybrid_collaborative_content_model.pkl')
        scaler = joblib.load('models/hybrid_scaler.pkl')
    
    # Prepare features (user + movie + temporal)
    features = prepare_features(user_id, movie_id)
    features_scaled = scaler.transform([features])
    
    # Predict
    predicted_rating = model.predict(features_scaled)[0]
    
    return predicted_rating
```

#### 2. **Top-N Recommendations**
```python
def get_recommendations(user_id, n=10, model_type='hybrid'):
    """
    Get top N movie recommendations for a user
    
    Input:
    - user_id: User identifier
    - n: Number of recommendations
    - model_type: 'content' or 'hybrid'
    
    Output:
    - recommendations: List of (movie_id, predicted_rating, title)
    """
    # Get all movies user hasn't rated
    unrated_movies = get_unrated_movies(user_id)
    
    # Predict ratings for all unrated movies
    predictions = []
    for movie_id in unrated_movies:
        pred_rating = predict_rating(user_id, movie_id, model_type)
        predictions.append((movie_id, pred_rating))
    
    # Sort by predicted rating and return top N
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]
```

#### 3. **Similar Movies Finder**
```python
def find_similar_movies(movie_id, n=10):
    """
    Find movies similar to a given movie
    
    Uses content-based features (genres, tags, embeddings)
    """
    # Load model and feature data
    model = joblib.load('models/advanced_content_based_model.pkl')
    
    # Get movie features
    target_features = get_movie_features(movie_id)
    
    # Find similar movies using cosine similarity or KNN
    similar_movies = model.find_similar_movies(movie_id, n)
    
    return similar_movies
```

## Use Cases in Your Web Platform

### 1. **Personalized Recommendations Tab**
- User selects 3+ movies they like
- System extracts user preferences from selections
- Models predict ratings for all movies
- Display top 20 recommendations with explanations

### 2. **Rating Predictor Feature**
- User selects a movie
- Model predicts their rating (e.g., "You'll rate this 4.2/5")
- Show confidence level and reasoning

### 3. **Mood-Based Search**
- User selects mood (e.g., "Action", "Romance")
- Filter movies by genre
- Rank by predicted rating for that user

### 4. **Challenge Mode**
- Find high-rated movies from genres user hasn't explored
- Use Model 1 (content-based) for cold-start handling
- Predict ratings even without user history

### 5. **Movie Roulette**
- Randomly select movies
- Use models to filter to high-predicted ratings
- Ensure quality recommendations

### 6. **Comparison Mode**
- Compare two movies side-by-side
- Show predicted ratings for each
- Display similarity score

## Model Selection Strategy

**Use Model 1 (Content-Based) when**:
- New user (no rating history)
- New movie (no ratings yet)
- Need genre-based recommendations
- Want to explain recommendations (content features are interpretable)

**Use Model 2 (Hybrid) when**:
- User has rating history (10+ ratings)
- Want best accuracy
- Need collaborative filtering benefits
- User has diverse preferences

**Default Strategy**:
```python
def get_best_model(user_id):
    user_rating_count = get_user_rating_count(user_id)
    
    if user_rating_count < 10:
        return 'content'  # Cold-start: use content-based
    else:
        return 'hybrid'   # Warm user: use hybrid for better accuracy
```

## Feature Requirements

To use the models, you need:

1. **User Features** (for Model 1):
   - User rating history
   - User genre preferences
   - User activity patterns

2. **Movie Features** (for both models):
   - Movie metadata (title, genres, release year)
   - Movie statistics (avg rating, num ratings)
   - TF-IDF features (from genres)
   - Genome embeddings (from tags)

3. **Temporal Features** (optional but improves accuracy):
   - Current date/time
   - User join date
   - Movie release date

## Next Steps for Integration

1. **Update Streamlit App** (`app.py`):
   - Replace old model loading with new models
   - Add feature preparation functions
   - Implement rating prediction API
   - Add model selection logic

2. **Create Backend API** (if using React frontend):
   - Flask/FastAPI endpoint for predictions
   - Load models once at startup
   - Cache feature preparation
   - Return JSON responses

3. **Add Real-Time Features**:
   - Update user features as they rate movies
   - Retrain models periodically (optional)
   - A/B test between models

4. **Performance Optimization**:
   - Pre-compute features for all movies
   - Cache predictions for popular movies
   - Use batch prediction for multiple users

