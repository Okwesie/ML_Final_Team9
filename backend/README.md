# Movie Recommendation API Backend

FastAPI-based ML inference server for movie recommendations.

## Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and add your TMDB API key:
```
TMDB_API_KEY=your_tmdb_api_key_here
ENVIRONMENT=development
PORT=8000
```

**Get TMDB API Key:**
1. Go to https://www.themoviedb.org/
2. Create an account
3. Go to Settings > API
4. Request an API key
5. Copy the key to `.env`

### 3. Ensure Models Are Available

The backend expects models in the parent directory:
```
../models/
├── advanced_content_based_model.pkl
├── hybrid_collaborative_content_model.pkl
├── svd_model.pkl
├── content_scaler.pkl
├── hybrid_scaler.pkl
└── feature_names.json
```

These are created when you run the Jupyter notebook (`Team_9_Final_Project.ipynb`).

### 4. Ensure Data Files Are Available

The backend needs:
```
../movies.csv
../ratings.csv
```

## Running the Server

### Development Mode

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Server will be available at:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

### Production Mode

```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## API Endpoints

### GET `/health`
Health check with model status.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": ["content", "hybrid"],
  "tmdb_service": "active",
  "data_loaded": true,
  "total_movies": 9742,
  "uptime_seconds": 3600
}
```

### POST `/api/recommend/personalized`
Get personalized recommendations.

**Request:**
```json
{
  "movie_ids": [1, 2, 3],
  "model": "content",
  "top_n": 10
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "movieId": 318,
      "title": "The Shawshank Redemption (1994)",
      "genres": "Drama|Crime",
      "score": 0.89,
      "predicted_rating": 4.45,
      "poster_url": "https://image.tmdb.org/t/p/w342/...",
      "overview": "..."
    }
  ],
  "model_used": "content",
  "computation_time_ms": 120
}
```

### POST `/api/recommend/similar`
Find similar movies.

**Request:**
```json
{
  "movie_id": 1,
  "model": "content",
  "top_n": 5
}
```

### POST `/api/predict/rating`
Predict rating for a movie.

**Request:**
```json
{
  "user_movie_ids": [1, 2, 3],
  "target_movie_id": 50,
  "model": "content"
}
```

### GET `/api/movies/search?query=toy story`
Search movies by title.

### GET `/api/movies/{movie_id}`
Get movie details.

## Testing

Run the test script:

```bash
python test_api.py
```

Make sure the server is running first!

## Troubleshooting

### Models Not Loading
- Check that models exist in `../models/` directory
- Verify model files are not corrupted
- Check logs for specific error messages

### TMDB Not Working
- Verify API key is set in `.env`
- Check internet connection
- TMDB API has rate limits (40 req/sec)

### Slow Performance
- Models load on startup (takes ~10-30 seconds)
- First request may be slower (cold start)
- Consider pre-computing features

### CORS Errors
- In development, CORS allows all origins
- In production, update `allowed_origins` in `main.py`

## Architecture

```
main.py              → FastAPI server, endpoints, CORS
model_service.py     → ML model loading, predictions
tmdb_service.py      → TMDB API integration, caching
```

## Performance Notes

- Models are loaded once on startup (cached)
- Movie features are pre-computed for faster predictions
- TMDB data is cached to avoid repeated API calls
- Recommendations limited to top 1000 candidates for speed

## Next Steps

1. Deploy to production (Heroku, AWS, etc.)
2. Add authentication if needed
3. Implement rate limiting
4. Add monitoring/logging
5. Optimize feature computation

