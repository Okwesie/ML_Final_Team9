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

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
#from dotenv import load_dotenv

import pandas as pd
from model_service import ModelService
from tmdb_service import TMDBService

# Load environment variables
#load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import logging

# Environment variables are set in Render dashboard
TMDB_API_KEY = os.getenv('TMDB_API_KEY')

# Setup logging
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




# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Movie Recommendation API",
    description="ML-powered movie recommendation system with TMDB integration",
    version="1.0.0"
)

# CORS configuration
environment = os.getenv('ENVIRONMENT', 'development')
if environment == 'production':
    allowed_origins = [
        "https://your-vercel-app.vercel.app",
        # Add your production frontend URL here
    ]
else:
    allowed_origins = ["*"]  # Allow all in development

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services (loaded on startup)
model_service: Optional[ModelService] = None
tmdb_service: Optional[TMDBService] = None
start_time = time.time()

# Pydantic models for request/response
class PersonalizedRecommendationRequest(BaseModel):
    movie_ids: List[int] = Field(..., description="List of movie IDs user has selected/liked")
    model: str = Field("content", description="Model to use: 'content' or 'hybrid'")
    top_n: int = Field(10, ge=1, le=50, description="Number of recommendations to return")

class SimilarMoviesRequest(BaseModel):
    movie_id: int = Field(..., description="Movie ID to find similar movies for")
    model: str = Field("content", description="Model to use: 'content' or 'hybrid'")
    top_n: int = Field(5, ge=1, le=20, description="Number of similar movies to return")

class RatingPredictionRequest(BaseModel):
    user_movie_ids: List[int] = Field(..., description="List of movie IDs user has selected")
    target_movie_id: int = Field(..., description="Movie ID to predict rating for")
    model: str = Field("content", description="Model to use: 'content' or 'hybrid'")

class MovieResponse(BaseModel):
    movieId: int
    title: str
    genres: Optional[str] = None
    year: Optional[int] = None
    score: Optional[float] = None
    predicted_rating: Optional[float] = None
    confidence: Optional[float] = None
    similarity_score: Optional[float] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    overview: Optional[str] = None
    vote_average: Optional[float] = None

class RecommendationResponse(BaseModel):
    recommendations: List[MovieResponse]
    model_used: str
    computation_time_ms: float
    based_on_movies: List[dict]

class SimilarMoviesResponse(BaseModel):
    target_movie: MovieResponse
    similar_movies: List[MovieResponse]
    model_used: str

class RatingPredictionResponse(BaseModel):
    predicted_rating: float
    confidence: float
    movie: MovieResponse
    model_used: str

class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    tmdb_service: str
    data_loaded: bool
    total_movies: int
    uptime_seconds: float

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load models and services on startup"""
    global model_service, tmdb_service
    
    logger.info("Starting up backend server...")
    
    try:
        # Initialize model service
        model_service = ModelService()
        if model_service.is_ready():
            logger.info("Model service initialized successfully")
        else:
            logger.warning("Model service initialized but models may not be fully loaded")
        
        # Initialize TMDB service
        tmdb_service = TMDBService()
        logger.info("TMDB service initialized")
        
        logger.info("Backend server ready!")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with model status"""
    global model_service, tmdb_service, start_time
    
    models_loaded = []
    if model_service:
        models_loaded = model_service.get_available_models()
    
    tmdb_status = "active" if tmdb_service and tmdb_service.api_key else "inactive"
    data_loaded = model_service is not None and model_service.movies_df is not None
    total_movies = len(model_service.movies_df) if data_loaded else 0
    
    return HealthResponse(
        status="healthy",
        models_loaded=models_loaded,
        tmdb_service=tmdb_status,
        data_loaded=data_loaded,
        total_movies=total_movies,
        uptime_seconds=time.time() - start_time
    )

# Personalized recommendations endpoint
@app.post("/api/recommend/personalized", response_model=RecommendationResponse)
async def get_personalized_recommendations(request: PersonalizedRecommendationRequest):
    """Get personalized movie recommendations based on user's selected movies"""
    if not model_service or not model_service.is_ready():
        raise HTTPException(status_code=503, detail="Model service not ready")
    
    if not request.movie_ids or len(request.movie_ids) == 0:
        raise HTTPException(status_code=400, detail="No movie IDs provided")
    
    start_time_ms = time.time() * 1000
    
    try:
        # Get recommendations from model
        recommendations = model_service.get_recommendations(
            user_movie_ids=request.movie_ids,
            top_n=request.top_n,
            model_type=request.model
        )
        
        # Enrich with TMDB data
        enriched_recommendations = []
        for rec in recommendations:
            movie_response = MovieResponse(**rec)
            
            # Enrich with TMDB
            if tmdb_service:
                tmdb_data = tmdb_service.enrich_movie(
                    movie_id=rec['movieId'],
                    title=rec['title'],
                    year=rec.get('year')
                )
                movie_response.poster_url = tmdb_data.get('poster_url')
                movie_response.backdrop_url = tmdb_data.get('backdrop_url')
                movie_response.overview = tmdb_data.get('overview')
                movie_response.vote_average = tmdb_data.get('vote_average')
            
            enriched_recommendations.append(movie_response)
        
        # Get based_on_movies details
        based_on_movies = []
        if model_service.movies_df is not None:
            for movie_id in request.movie_ids:
                movie_row = model_service.movies_df[
                    model_service.movies_df['movieId'] == movie_id
                ]
                if not movie_row.empty:
                    movie = movie_row.iloc[0]
                    based_on_movies.append({
                        'movieId': int(movie_id),
                        'title': movie['title'],
                        'genres': movie['genres']
                    })
        
        computation_time = (time.time() * 1000) - start_time_ms
        
        return RecommendationResponse(
            recommendations=enriched_recommendations,
            model_used=request.model,
            computation_time_ms=computation_time,
            based_on_movies=based_on_movies
        )
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

# Similar movies endpoint
@app.post("/api/recommend/similar", response_model=SimilarMoviesResponse)
async def get_similar_movies(request: SimilarMoviesRequest):
    """Find movies similar to a given movie"""
    if not model_service or not model_service.is_ready():
        raise HTTPException(status_code=503, detail="Model service not ready")
    
    try:
        # Get target movie details
        target_movie_row = model_service.movies_df[
            model_service.movies_df['movieId'] == request.movie_id
        ]
        
        if target_movie_row.empty:
            raise HTTPException(status_code=404, detail=f"Movie {request.movie_id} not found")
        
        target_movie = target_movie_row.iloc[0]
        target_movie_response = MovieResponse(
            movieId=int(request.movie_id),
            title=target_movie['title'],
            genres=target_movie['genres'],
            year=int(target_movie.get('release_year', 0)) if pd.notna(target_movie.get('release_year')) else None
        )
        
        # Enrich target movie with TMDB
        if tmdb_service:
            tmdb_data = tmdb_service.enrich_movie(
                movie_id=request.movie_id,
                title=target_movie['title'],
                year=target_movie_response.year
            )
            target_movie_response.poster_url = tmdb_data.get('poster_url')
            target_movie_response.backdrop_url = tmdb_data.get('backdrop_url')
            target_movie_response.overview = tmdb_data.get('overview')
            target_movie_response.vote_average = tmdb_data.get('vote_average')
        
        # Find similar movies
        similar_movies = model_service.find_similar_movies(
            movie_id=request.movie_id,
            top_n=request.top_n
        )
        
        # Enrich similar movies with TMDB
        enriched_similar = []
        for movie in similar_movies:
            movie_response = MovieResponse(**movie)
            
            if tmdb_service:
                tmdb_data = tmdb_service.enrich_movie(
                    movie_id=movie['movieId'],
                    title=movie['title'],
                    year=movie.get('year')
                )
                movie_response.poster_url = tmdb_data.get('poster_url')
                movie_response.backdrop_url = tmdb_data.get('backdrop_url')
                movie_response.overview = tmdb_data.get('overview')
                movie_response.vote_average = tmdb_data.get('vote_average')
            
            enriched_similar.append(movie_response)
        
        return SimilarMoviesResponse(
            target_movie=target_movie_response,
            similar_movies=enriched_similar,
            model_used=request.model
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding similar movies: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error finding similar movies: {str(e)}")

# Rating prediction endpoint
@app.post("/api/predict/rating", response_model=RatingPredictionResponse)
async def predict_rating(request: RatingPredictionRequest):
    """Predict how much a user will rate a specific movie"""
    if not model_service or not model_service.is_ready():
        raise HTTPException(status_code=503, detail="Model service not ready")
    
    if not request.user_movie_ids or len(request.user_movie_ids) == 0:
        raise HTTPException(status_code=400, detail="No user movie IDs provided")
    
    try:
        # Predict rating
        predicted_rating, confidence = model_service.predict_rating(
            user_movie_ids=request.user_movie_ids,
            target_movie_id=request.target_movie_id,
            model_type=request.model
        )
        
        # Get movie details
        movie_row = model_service.movies_df[
            model_service.movies_df['movieId'] == request.target_movie_id
        ]
        
        if movie_row.empty:
            raise HTTPException(status_code=404, detail=f"Movie {request.target_movie_id} not found")
        
        movie = movie_row.iloc[0]
        movie_response = MovieResponse(
            movieId=int(request.target_movie_id),
            title=movie['title'],
            genres=movie['genres'],
            year=int(movie.get('release_year', 0)) if pd.notna(movie.get('release_year')) else None
        )
        
        # Enrich with TMDB
        if tmdb_service:
            tmdb_data = tmdb_service.enrich_movie(
                movie_id=request.target_movie_id,
                title=movie['title'],
                year=movie_response.year
            )
            movie_response.poster_url = tmdb_data.get('poster_url')
            movie_response.backdrop_url = tmdb_data.get('backdrop_url')
            movie_response.overview = tmdb_data.get('overview')
            movie_response.vote_average = tmdb_data.get('vote_average')
        
        return RatingPredictionResponse(
            predicted_rating=predicted_rating,
            confidence=confidence,
            movie=movie_response,
            model_used=request.model
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting rating: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error predicting rating: {str(e)}")

# Movie search endpoint
@app.get("/api/movies/search")
async def search_movies(query: str = Query(..., min_length=1, description="Search query")):
    """Search movies by title"""
    if not model_service or model_service.movies_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    try:
        # Search in movies dataframe
        matching_movies = model_service.movies_df[
            model_service.movies_df['title'].str.contains(query, case=False, na=False)
        ].head(20)  # Limit to 20 results
        
        results = []
        for _, movie in matching_movies.iterrows():
            movie_data = {
                'movieId': int(movie['movieId']),
                'title': movie['title'],
                'genres': movie['genres'],
                'year': int(movie.get('release_year', 0)) if pd.notna(movie.get('release_year')) else None
            }
            
            # Enrich with TMDB
            if tmdb_service:
                tmdb_data = tmdb_service.enrich_movie(
                    movie_id=movie['movieId'],
                    title=movie['title'],
                    year=movie_data['year']
                )
                movie_data['poster_url'] = tmdb_data.get('poster_url')
                movie_data['overview'] = tmdb_data.get('overview')
            
            results.append(movie_data)
        
        return {"results": results, "count": len(results)}
        
    except Exception as e:
        logger.error(f"Error searching movies: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error searching movies: {str(e)}")

# Get movie details endpoint
@app.get("/api/movies/{movie_id}")
async def get_movie_details(movie_id: int):
    """Get detailed information about a specific movie"""
    if not model_service or model_service.movies_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    try:
        movie_row = model_service.movies_df[
            model_service.movies_df['movieId'] == movie_id
        ]
        
        if movie_row.empty:
            raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found")
        
        movie = movie_row.iloc[0]
        movie_data = {
            'movieId': int(movie_id),
            'title': movie['title'],
            'genres': movie['genres'],
            'year': int(movie.get('release_year', 0)) if pd.notna(movie.get('release_year')) else None
        }
        
        # Add movie features
        movie_features = model_service._get_movie_features(movie_id)
        movie_data['features'] = {
            'avg_rating': movie_features.get('movie_avg_rating'),
            'num_ratings': movie_features.get('movie_num_ratings'),
            'rating_momentum': movie_features.get('movie_rating_momentum')
        }
        
        # Enrich with TMDB
        if tmdb_service:
            tmdb_data = tmdb_service.enrich_movie(
                movie_id=movie_id,
                title=movie['title'],
                year=movie_data['year']
            )
            movie_data['poster_url'] = tmdb_data.get('poster_url')
            movie_data['backdrop_url'] = tmdb_data.get('backdrop_url')
            movie_data['overview'] = tmdb_data.get('overview')
            movie_data['vote_average'] = tmdb_data.get('vote_average')
            movie_data['tmdb_id'] = tmdb_data.get('tmdb_id')
        
        return movie_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting movie details: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting movie details: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Movie Recommendation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

