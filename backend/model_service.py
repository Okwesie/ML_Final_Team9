import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self):
        self.movies_df = None
        self.models = {}
        self._load_data()
        self._load_mock_models() # Placeholders to prevent crashes if files missing
        logger.info("✓ ModelService initialized")

    def _load_data(self):
        # Path adjustment for your local structure
        path = Path(__file__).parent / "movies.csv"
        if path.exists():
            self.movies_df = pd.read_csv(path)
            # Fast extraction of years
            self.movies_df['release_year'] = self.movies_df['title'].str.extract(r'\((\d{4})\)').astype(float)
            logger.info(f"✓ Loaded {len(self.movies_df)} movies")

    def _load_mock_models(self):
        # In a real scenario, use joblib.load here
        self.models['hybrid'] = True 

    def get_recommendations(self, user_movie_ids: List[int], top_n: int = 10, model_type: str = 'hybrid') -> List[Dict]:
        if self.movies_df is None: return []
        # Return top N movies not in the seed list
        available = self.movies_df[~self.movies_df['movieId'].isin(user_movie_ids)].head(top_n)
        return [{
            "movieId": int(row['movieId']),
            "title": row['title'],
            "genres": row['genres'],
            "year": int(row['release_year']) if pd.notna(row['release_year']) else None,
            "predicted_rating": 4.5
        } for _, row in available.iterrows()]

    def is_ready(self) -> bool:
        return self.movies_df is not None

    def get_available_models(self):
        return ["hybrid", "content", "svd"]