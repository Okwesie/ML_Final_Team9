"""
FastAPI Backend Server for Movie Recommendation System
Main entry point for ML inference API
"""
import os
import time
import logging
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from model_service import ModelService
from tmdb_service import TMDBService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Movie Recommendation API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services
model_service = None
tmdb_service = None

# ✅ PYDANTIC MODELS - MATCH FRONTEND EXACTLY
class PredictRatingRequest(BaseModel):
    user_movie_ids: list[int]
    target_movie_id: int
    model: str = "content"

class RecommendationsRequest(BaseModel):
    movie_ids: list[int]
    top_n: int = 10
    model: str = "content"

class SimilarMoviesRequest(BaseModel):
    movie_id: int
    top_n: int = 10
    model: str = "content"

@app.on_event("startup")
async def startup_event():
    global model_service, tmdb_service
    logger.info("Starting backend...")
    try:
        model_service = ModelService()
        tmdb_service = TMDBService()
        logger.info("✓ Backend ready")
    except Exception as e:
        logger.error(f"✗ Failed to initialize services: {e}")
        raise

@app.get("/health")
async def health():
    """Check backend health"""
    if not model_service:
        return {
            "status": "degraded",
            "models_loaded": [],
            "data_ready": False
        }
    
    return {
        "status": "healthy" if model_service.is_ready() else "degraded",
        "models_loaded": model_service.get_available_models(),
        "data_ready": model_service.movies_df is not None
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Movie Recommendation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/api/movies/search")
async def search_movies(query: str):
    """Search for movies by title"""
    if not model_service or model_service.movies_df is None:
        raise HTTPException(status_code=503, detail="Movies data not loaded")
    
    try:
        results = model_service.movies_df[
            model_service.movies_df['title'].str.contains(query, case=False, na=False)
        ].head(20).to_dict('records')
        
        return {
            "results": [
                {
                    "movieId": int(r['movieId']),
                    "title": r['title'],
                    "genres": r['genres'],
                    "year": int(r.get('release_year', 0)) if r.get('release_year') else None
                }
                for r in results
            ],
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/movies/{movie_id}")
async def get_movie_details(movie_id: int):
    """Get movie details by ID"""
    if not model_service or model_service.movies_df is None:
        raise HTTPException(status_code=503, detail="Movies data not loaded")
    
    try:
        movie = model_service.movies_df[model_service.movies_df['movieId'] == movie_id]
        if movie.empty:
            raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found")
        
        m = movie.iloc[0]
        return {
            "movieId": int(m['movieId']),
            "title": m['title'],
            "genres": m['genres'],
            "year": int(m.get('release_year', 0)) if m.get('release_year') else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting movie: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/rating")
async def predict_rating(request: PredictRatingRequest):
    """Predict rating for a movie"""
    if not model_service or not model_service.is_ready():
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        logger.info(f"[PREDICT] user_movies={request.user_movie_ids}, target={request.target_movie_id}, model={request.model}")
        
        pred_rating, confidence = model_service.predict_rating(
            user_movie_ids=request.user_movie_ids,
            target_movie_id=request.target_movie_id,
            model_type=request.model
        )
        
        movie = None
        if model_service.movies_df is not None:
            movie_row = model_service.movies_df[
                model_service.movies_df['movieId'] == request.target_movie_id
            ]
            if not movie_row.empty:
                m = movie_row.iloc[0]
                movie = {
                    "movieId": int(m['movieId']),
                    "title": m['title'],
                    "genres": m['genres'],
                    "year": int(m.get('release_year', 0)) if m.get('release_year') else None
                }
        
        return {
            "predicted_rating": float(pred_rating),
            "confidence": float(confidence),
            "movie": movie,
            "model_used": request.model
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recommend/personalized")
async def get_recommendations(request: RecommendationsRequest):
    """Get personalized recommendations"""
    if not model_service or not model_service.is_ready():
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        logger.info(f"[RECOMMEND] movies={request.movie_ids}, top_n={request.top_n}, model={request.model}")
        
        recommendations = model_service.get_recommendations(
            user_movie_ids=request.movie_ids,
            top_n=request.top_n,
            model_type=request.model
        )
        
        return {
            "recommendations": recommendations,
            "model_used": request.model,
            "count": len(recommendations)
        }
    except Exception as e:
        logger.error(f"Recommendations error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recommend/similar")
async def find_similar_movies(request: SimilarMoviesRequest):
    """Find similar movies"""
    if not model_service or model_service.movies_df is None:
        raise HTTPException(status_code=503, detail="Movies data not loaded")
    
    try:
        logger.info(f"[SIMILAR] movie_id={request.movie_id}, top_n={request.top_n}")
        
        similar = model_service.find_similar_movies(
            movie_id=request.movie_id,
            top_n=request.top_n
        )
        
        # Get target movie info
        target_movie = None
        if model_service.movies_df is not None:
            movie_row = model_service.movies_df[
                model_service.movies_df['movieId'] == request.movie_id
            ]
            if not movie_row.empty:
                m = movie_row.iloc[0]
                target_movie = {
                    "movieId": int(m['movieId']),
                    "title": m['title'],
                    "genres": m['genres'],
                    "year": int(m.get('release_year', 0)) if m.get('release_year') else None
                }
        
        return {
            "target_movie": target_movie,
            "similar_movies": similar,
            "count": len(similar)
        }
    except Exception as e:
        logger.error(f"Similar movies error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

