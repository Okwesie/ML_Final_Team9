"""
TMDB API Service
Handles movie metadata fetching, poster URLs, caching and common endpoints
Supports API key or Bearer access token authentication
"""
import os
import requests
import pandas as pd
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class TMDBService:
    BASE_URL = "https://api.themoviedb.org/3"
    IMAGE_BASE_URL = "https://image.tmdb.org/t/p"

    POSTER_SIZES = {
        'small': 'w185',
        'medium': 'w342',
        'large': 'w500',
        'original': 'original'
    }

    BACKDROP_SIZES = {
        'small': 'w300',
        'medium': 'w780',
        'large': 'w1280',
        'original': 'original'
    }

    def __init__(self, api_key: Optional[str] = None, access_token: Optional[str] = None):
        # Support API key or Bearer token
        self.api_key = api_key or '2987ff78d07868dcd9a70188fc5b2afe'
        self.access_token = access_token or 'eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyOTg3ZmY3OGQwNzg2OGRjZDlhNzAxODhmYzViMmFmZSIsIm5iZiI6MTc2NjA4MzE1MS4wOTc5OTk4LCJzdWIiOiI2OTQ0NGE0Zjg5MzdhYmVjM2FkZWM3MDgiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.lANfnGrBUtANyW9MjgtPdQdV1Y2YxEM5vW-GmSQ5Db4'

        self.use_bearer = bool(self.access_token)

        if not self.api_key and not self.access_token:
            logger.warning("TMDB credentials not found. TMDB calls will return None.")
        elif self.use_bearer:
            logger.info("TMDB: using Bearer token auth")
        else:
            logger.info("TMDB: using API key auth")

        # cache
        cache_dir = Path(__file__).parent / 'data'
        cache_dir.mkdir(exist_ok=True)
        self.cache_file = cache_dir / 'movies_with_tmdb.csv'
        self.cache_df = None
        self._load_cache()

        # rate limiting
        self.last_request_time = 0.0
        self.min_request_interval = 0.25  # seconds

    def _load_cache(self):
        if self.cache_file.exists():
            try:
                self.cache_df = pd.read_csv(self.cache_file)
                logger.info(f"Loaded TMDB cache ({len(self.cache_df)} rows)")
            except Exception as e:
                logger.error("Failed to load TMDB cache: %s", e)
                self.cache_df = pd.DataFrame()
        else:
            self.cache_df = pd.DataFrame()

    def _save_cache(self):
        try:
            self.cache_df.to_csv(self.cache_file, index=False)
            logger.info("Saved TMDB cache")
        except Exception as e:
            logger.error("Failed to save TMDB cache: %s", e)

    def _rate_limit(self):
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict]:
        if not self.api_key and not self.access_token:
            logger.debug("No TMDB credentials, skipping request to %s", endpoint)
            return None

        self._rate_limit()
        url = f"{self.BASE_URL}{endpoint}"
        params = params.copy() if params else {}
        headers = {"Accept": "application/json"}

        if self.use_bearer:
            headers["Authorization"] = f"Bearer {self.access_token}"
        else:
            params["api_key"] = self.api_key

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=8)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as he:
            status = getattr(he.response, "status_code", None)
            text = getattr(he.response, "text", "")
            logger.error("TMDB HTTP error %s: %s", status, text)
            return None
        except requests.exceptions.RequestException as re:
            logger.error("TMDB request error: %s", re)
            return None

    # --- Common endpoints -------------------------------------------------
    def search_movie(self, title: str, year: Optional[int] = None, page: int = 1) -> Optional[Dict]:
        params = {"query": title, "page": page}
        if year:
            params["year"] = year
        data = self._make_request("/search/movie", params)
        if data and data.get("results"):
            return data["results"][0]
        return None

    def get_movie_details(self, tmdb_id: int, append_to_response: Optional[str] = None) -> Optional[Dict]:
        params = {}
        if append_to_response:
            params["append_to_response"] = append_to_response
        return self._make_request(f"/movie/{tmdb_id}", params)

    def get_movie_by_title(self, title: str, year: Optional[int] = None) -> Optional[Dict]:
        # check cache first
        if not self.cache_df.empty and "title" in self.cache_df.columns:
            cached = self.cache_df[self.cache_df["title"].str.contains(title, case=False, na=False)]
            if not cached.empty and "tmdb_id" in cached.columns:
                tmdb_id = cached.iloc[0]["tmdb_id"]
                if pd.notna(tmdb_id):
                    return self.get_movie_details(int(tmdb_id))

        # fallback to search
        result = self.search_movie(title, year)
        if result:
            self._cache_movie(result, title)
            return self.get_movie_details(result.get("id"))
        return None

    def _cache_movie(self, tmdb_data: Dict, original_title: str):
        if self.cache_df is None or self.cache_df.empty:
            self.cache_df = pd.DataFrame(columns=[
                "title", "tmdb_id", "poster_path", "backdrop_path", "overview", "vote_average"
            ])
        new_row = {
            "title": original_title,
            "tmdb_id": tmdb_data.get("id"),
            "poster_path": tmdb_data.get("poster_path"),
            "backdrop_path": tmdb_data.get("backdrop_path"),
            "overview": tmdb_data.get("overview", ""),
            "vote_average": tmdb_data.get("vote_average", 0)
        }
        self.cache_df = pd.concat([self.cache_df, pd.DataFrame([new_row])], ignore_index=True)
        self._save_cache()

    # --- Helpers ----------------------------------------------------------
    def get_poster_url(self, poster_path: Optional[str], size: str = 'medium') -> Optional[str]:
        if not poster_path:
            return None
        size_code = self.POSTER_SIZES.get(size, 'w342')
        return f"{self.IMAGE_BASE_URL}/{size_code}{poster_path}"

    def get_backdrop_url(self, backdrop_path: Optional[str], size: str = 'large') -> Optional[str]:
        if not backdrop_path:
            return None
        size_code = self.BACKDROP_SIZES.get(size, 'w1280')
        return f"{self.IMAGE_BASE_URL}/{size_code}{backdrop_path}"

    # --- Convenience / extended endpoints --------------------------------
    def enrich_movie(self, movie_id: int, title: str, year: Optional[int] = None) -> Dict:
        """Return a dictionary with movie metadata + poster/backdrop/overview/vote_average"""
        result = {
            "movieId": movie_id,
            "title": title,
            "poster_url": None,
            "backdrop_url": None,
            "overview": None,
            "vote_average": None,
            "tmdb_id": None
        }

        if not self.cache_df.empty:
            cached = self.cache_df[self.cache_df["title"].str.contains(title, case=False, na=False)]
            if not cached.empty:
                row = cached.iloc[0]
                result["poster_url"] = self.get_poster_url(row.get("poster_path"))
                result["backdrop_url"] = self.get_backdrop_url(row.get("backdrop_path"))
                result["overview"] = row.get("overview")
                result["vote_average"] = row.get("vote_average")
                result["tmdb_id"] = row.get("tmdb_id")
                return result

        tmdb_data = self.get_movie_by_title(title, year)
        if tmdb_data:
            result["poster_url"] = self.get_poster_url(tmdb_data.get("poster_path"))
            result["backdrop_url"] = self.get_backdrop_url(tmdb_data.get("backdrop_path"))
            result["overview"] = tmdb_data.get("overview", "")
            result["vote_average"] = tmdb_data.get("vote_average", 0)
            result["tmdb_id"] = tmdb_data.get("id")
        return result

    # alias matching earlier examples
    def enrich_movie_data(self, title: str, year: Optional[int] = None) -> Optional[Dict]:
        movie = self.get_movie_by_title(title, year)
        if not movie:
            return None
        return self.get_movie_details(movie.get("id"))

    def get_similar_movies(self, tmdb_id: int, page: int = 1) -> Optional[Dict]:
        return self._make_request(f"/movie/{tmdb_id}/similar", {"page": page})

    def get_trending_movies(self, time_window: str = 'week', media_type: str = 'movie', page: int = 1) -> Optional[Dict]:
        # time_window: 'day' or 'week'; media_type: 'movie', 'all', 'tv', 'person'
        return self._make_request(f"/trending/{media_type}/{time_window}", {"page": page})

    def discover_movies(self,
                        with_genres: Optional[Any] = None,
                        vote_average_gte: Optional[float] = None,
                        sort_by: str = 'popularity.desc',
                        page: int = 1) -> Optional[Dict]:
        params: Dict[str, Any] = {"sort_by": sort_by, "page": page}
        if with_genres is not None:
            # accept int, list[int] or comma-separated str
            if isinstance(with_genres, (list, tuple)):
                params["with_genres"] = ",".join(map(str, with_genres))
            else:
                params["with_genres"] = str(with_genres)
        if vote_average_gte is not None:
            params["vote_average.gte"] = vote_average_gte
        return self._make_request("/discover/movie", params)

    def batch_enrich(self, movies: List[Dict], delay: float = 0.25) -> List[Dict]:
        enriched = []
        for movie in movies:
            title = movie.get("title", "")
            year = movie.get("year")
            movie_id = movie.get("movieId")
            enriched_movie = self.enrich_movie(movie_id, title, year)
            enriched.append({**movie, **enriched_movie})
            time.sleep(delay)
        return enriched

