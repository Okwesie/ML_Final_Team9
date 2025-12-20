from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import logging
import re

from model_service import ModelService
from tmdb_service import TMDBService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model_service = None
tmdb_service = None

@app.on_event("startup")
async def startup():
    global model_service, tmdb_service
    model_service = ModelService()
    tmdb_service = TMDBService()
    logger.info("Backend Services Started")

# --- MODELS ---
class RecRequest(BaseModel):
    movie_ids: List[int]
    top_n: int = 10
    model: str = "hybrid"

class SimilarRequest(BaseModel):
    movie_id: int
    top_n: int = 5

# --- FIXED ENDPOINTS ---

@app.get("/health")
async def health():
    """FIXED: Added missing health endpoint"""
    return {
        "status": "healthy",
        "model_service": model_service.is_ready() if model_service else False,
        "movies_count": len(model_service.movies_df) if model_service.movies_df is not None else 0
    }

@app.get("/api/movies/search")
async def search(query: str):
    if model_service.movies_df is None: return {"results": []}
    
    results = model_service.movies_df[
        model_service.movies_df['title'].str.contains(query, case=False, na=False, regex=False)
    ].head(10)
    
    movies = []
    for _, r in results.iterrows():
        # Clean title: "Toy Story (1995)" -> "Toy Story"
        clean_title = re.sub(r'\s*\(\d{4}\)', '', r['title']).strip()
        movies.append({
            "movieId": int(r['movieId']),
            "title": r['title'],
            "clean_title": clean_title,
            "year": int(r['release_year']) if pd.notna(r['release_year']) else None
        })
    
    return {"results": tmdb_service.batch_enrich(movies)}

@app.get("/api/movies/{movie_id}")
async def get_details(movie_id: int):
    row = model_service.movies_df[model_service.movies_df['movieId'] == movie_id]
    if row.empty: raise HTTPException(404, "Movie not found")
    
    m = row.iloc[0]
    full_title = m['title']
    clean_title = re.sub(r'\s*\(\d{4}\)', '', full_title).strip()
    year = int(m['release_year']) if pd.notna(m['release_year']) else None
    
    enriched = tmdb_service.enrich_movie(movie_id, clean_title, year)
    return {
        "movieId": movie_id,
        "title": full_title,
        "year": year,
        **enriched
    }

@app.post("/api/recommend/personalized")
async def recommend(req: RecRequest):
    recs = model_service.get_recommendations(req.movie_ids, req.top_n)
    # Clean titles for TMDB before enriching
    for r in recs:
        r['title'] = re.sub(r'\s*\(\d{4}\)', '', r['title']).strip()
    return {"recommendations": tmdb_service.batch_enrich(recs)}

@app.post("/api/recommend/similar")
async def similar_movies(req: SimilarRequest):
    """FIXED: Added missing similar endpoint"""
    # Logic to find similar movies (placeholder or model call)
    similar = model_service.get_recommendations([req.movie_id], req.top_n)
    return {"similar_movies": tmdb_service.batch_enrich(similar)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)