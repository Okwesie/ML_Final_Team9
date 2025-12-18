"""
TMDB API Service
Handles movie metadata fetching, poster URLs, and caching
"""
import os
import requests
import pandas as pd
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

class TMDBService:
    """Service for interacting with The Movie Database (TMDB) API"""
    
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
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('TMDB_API_KEY')
        if not self.api_key:
            logger.warning("TMDB API key not found. TMDB features will be disabled.")
        
        # Cache file in backend/data directory
        cache_dir = Path(__file__).parent / 'data'
        cache_dir.mkdir(exist_ok=True)
        self.cache_file = cache_dir / 'movies_with_tmdb.csv'
        self.cache_df = None
        self._load_cache()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.25  # 250ms between requests
    
    def _load_cache(self):
        """Load cached TMDB data if available"""
        if self.cache_file.exists():
            try:
                self.cache_df = pd.read_csv(self.cache_file)
                logger.info(f"Loaded TMDB cache with {len(self.cache_df)} movies")
            except Exception as e:
                logger.error(f"Error loading TMDB cache: {e}")
                self.cache_df = pd.DataFrame()
        else:
            self.cache_df = pd.DataFrame()
    
    def _save_cache(self):
        """Save TMDB data to cache file"""
        try:
            self.cache_df.to_csv(self.cache_file, index=False)
            logger.info(f"Saved TMDB cache with {len(self.cache_df)} movies")
        except Exception as e:
            logger.error(f"Error saving TMDB cache: {e}")
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request with error handling"""
        if not self.api_key:
            return None
        
        self._rate_limit()
        
        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}
        params['api_key'] = self.api_key
        
        try:
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"TMDB API request failed: {e}")
            return None
    
    def search_movie(self, title: str, year: Optional[int] = None) -> Optional[Dict]:
        """Search for a movie by title and optional year"""
        params = {'query': title}
        if year:
            params['year'] = year
        
        data = self._make_request('/search/movie', params)
        if data and data.get('results'):
            return data['results'][0]  # Return first match
        return None
    
    def get_movie_details(self, tmdb_id: int) -> Optional[Dict]:
        """Get detailed movie information by TMDB ID"""
        return self._make_request(f'/movie/{tmdb_id}')
    
    def get_movie_by_title(self, title: str, year: Optional[int] = None) -> Optional[Dict]:
        """Get movie by title, checking cache first"""
        # Check cache first
        if not self.cache_df.empty and 'title' in self.cache_df.columns:
            cached = self.cache_df[
                self.cache_df['title'].str.contains(title, case=False, na=False)
            ]
            if not cached.empty and 'tmdb_id' in cached.columns:
                tmdb_id = cached.iloc[0]['tmdb_id']
                if pd.notna(tmdb_id):
                    return self.get_movie_details(int(tmdb_id))
        
        # Search API
        result = self.search_movie(title, year)
        if result:
            # Cache the result
            self._cache_movie(result, title)
            return self.get_movie_details(result['id'])
        return None
    
    def _cache_movie(self, tmdb_data: Dict, original_title: str):
        """Cache movie data"""
        if self.cache_df.empty:
            self.cache_df = pd.DataFrame(columns=['title', 'tmdb_id', 'poster_path', 
                                                   'backdrop_path', 'overview', 'vote_average'])
        
        new_row = {
            'title': original_title,
            'tmdb_id': tmdb_data.get('id'),
            'poster_path': tmdb_data.get('poster_path'),
            'backdrop_path': tmdb_data.get('backdrop_path'),
            'overview': tmdb_data.get('overview', ''),
            'vote_average': tmdb_data.get('vote_average', 0)
        }
        
        self.cache_df = pd.concat([self.cache_df, pd.DataFrame([new_row])], ignore_index=True)
        self._save_cache()
    
    def get_poster_url(self, poster_path: Optional[str], size: str = 'medium') -> Optional[str]:
        """Construct poster URL from poster_path"""
        if not poster_path:
            return None
        
        size_code = self.POSTER_SIZES.get(size, 'w342')
        return f"{self.IMAGE_BASE_URL}/{size_code}{poster_path}"
    
    def get_backdrop_url(self, backdrop_path: Optional[str], size: str = 'large') -> Optional[str]:
        """Construct backdrop URL from backdrop_path"""
        if not backdrop_path:
            return None
        
        size_code = self.BACKDROP_SIZES.get(size, 'w1280')
        return f"{self.IMAGE_BASE_URL}/{size_code}{backdrop_path}"
    
    def enrich_movie(self, movie_id: int, title: str, year: Optional[int] = None) -> Dict:
        """Enrich movie data with TMDB information"""
        result = {
            'movieId': movie_id,
            'title': title,
            'poster_url': None,
            'backdrop_url': None,
            'overview': None,
            'vote_average': None,
            'tmdb_id': None
        }
        
        # Check cache first
        if not self.cache_df.empty:
            cached = self.cache_df[self.cache_df['title'].str.contains(title, case=False, na=False)]
            if not cached.empty:
                row = cached.iloc[0]
                result['poster_url'] = self.get_poster_url(row.get('poster_path'))
                result['backdrop_url'] = self.get_backdrop_url(row.get('backdrop_path'))
                result['overview'] = row.get('overview')
                result['vote_average'] = row.get('vote_average')
                result['tmdb_id'] = row.get('tmdb_id')
                return result
        
        # Fetch from API
        tmdb_data = self.get_movie_by_title(title, year)
        if tmdb_data:
            result['poster_url'] = self.get_poster_url(tmdb_data.get('poster_path'))
            result['backdrop_url'] = self.get_backdrop_url(tmdb_data.get('backdrop_path'))
            result['overview'] = tmdb_data.get('overview', '')
            result['vote_average'] = tmdb_data.get('vote_average', 0)
            result['tmdb_id'] = tmdb_data.get('id')
        
        return result
    
    def batch_enrich(self, movies: List[Dict], delay: float = 0.25) -> List[Dict]:
        """Enrich multiple movies with TMDB data"""
        enriched = []
        for movie in movies:
            title = movie.get('title', '')
            year = movie.get('year')
            movie_id = movie.get('movieId')
            
            enriched_movie = self.enrich_movie(movie_id, title, year)
            enriched.append({**movie, **enriched_movie})
            
            time.sleep(delay)  # Rate limiting
        
        return enriched

