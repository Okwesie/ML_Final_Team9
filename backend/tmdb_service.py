import os
import requests
import pandas as pd
import logging
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

class TMDBService:
    BASE_URL = "https://api.themoviedb.org/3"
    IMAGE_BASE_URL = "https://image.tmdb.org/t/p"

    def __init__(self, access_token: str = None):
        # Using your provided token
        self.access_token = access_token or 'eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyOTg3ZmY3OGQwNzg2OGRjZDlhNzAxODhmYzViMmFmZSIsIm5iZiI6MTc2NjA4MzE1MS4wOTc5OTk4LCJzdWIiOiI2OTQ0NGE0Zjg5MzdhYmVjM2FkZWM3MDgiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.lANfnGrBUtANyW9MjgtPdQdV1Y2YxEM5vW-GmSQ5Db4'
        self.headers = {"Authorization": f"Bearer {self.access_token}", "Content-Type": "application/json;charset=utf-8"}
        self.cache_df = pd.DataFrame() 
        logger.info("TMDB Service Initialized")

    def get_poster_url(self, path: str, size: str = 'w342') -> Optional[str]:
        return f"{self.IMAGE_BASE_URL}/{size}{path}" if path else None

    def search_movie(self, title: str, year: Optional[int] = None) -> Optional[Dict]:
        params = {"query": title}
        if year: params["primary_release_year"] = year
        try:
            res = requests.get(f"{self.BASE_URL}/search/movie", params=params, headers=self.headers)
            results = res.json().get("results", [])
            return results[0] if results else None
        except Exception as e:
            logger.error(f"TMDB Search Error: {e}")
            return None

    def enrich_movie(self, movie_id: int, title: str, year: Optional[int] = None) -> Dict:
        """FIXED: Takes explicit arguments to match main.py calls"""
        tmdb_data = self.search_movie(title, year)
        if tmdb_data:
            return {
                "poster_url": self.get_poster_url(tmdb_data.get("poster_path")),
                "backdrop_url": self.get_poster_url(tmdb_data.get("backdrop_path"), "w780"),
                "overview": tmdb_data.get("overview"),
                "vote_average": tmdb_data.get("vote_average"),
                "tmdb_id": tmdb_data.get("id")
            }
        return {"poster_url": None, "overview": "No description available."}

    def batch_enrich(self, movies: List[Dict]) -> List[Dict]:
        """Processes a list of movies and adds TMDB metadata"""
        enriched = []
        for movie in movies:
            mid = movie.get("movieId")
            title = movie.get("title", "")
            year = movie.get("year")
            # Logic call
            tmdb_info = self.enrich_movie(mid, title, year)
            enriched.append({**movie, **tmdb_info})
        return enriched