"""
Model Service
Handles ML model loading, feature preparation, and predictions
Downloads models from Google Drive if not present locally
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging
from functools import lru_cache
import requests
from urllib.parse import urljoin
import gdown
import io

logger = logging.getLogger(__name__)

# Google Drive folder ID containing models
GOOGLE_DRIVE_FOLDER_ID = "1jTX2MerREvKvSdMjHCKQSGyxH9CNxGdE"

# Google Drive file IDs for data CSVs
MOVIES_CSV_ID = "movies.csv"  # Replace with actual Google Drive file ID
RATINGS_CSV_ID = "ratings.csv"  # Replace with actual Google Drive file ID

class ModelService:
    """Service for loading and using trained ML models"""
    
    # Model file mappings (filename -> Google Drive file ID)
    MODEL_FILES = {
        "advanced_content_based_model.pkl": "1Abc123def456ghi789",
        "content_scaler.pkl": "2Xyz789abc456def123",
        "hybrid_collaborative_content_model.pkl": "3Qwe456rty789uio012",
        "hybrid_scaler.pkl": "4Asd123fgh456jkl789",
        "svd_model.pkl": "5Zxc789vbn456mno012",
        "feature_names.json": "6Poi789uik456lmn012",
        "movie_features.pkl": "7Lkj456mno789pqr012"
    }
    
    def __init__(self, models_dir: str = None, data_dir: str = None):
        if models_dir is None:
            models_dir = Path(__file__).parent / "models"
        if data_dir is None:
            data_dir = Path(__file__).parent / "data"
        
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Models directory: {self.models_dir}")
        logger.info(f"Data directory: {self.data_dir}")
        
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        self.movies_df = None
        self.ratings_df = None
        self.movie_features_cache = {}
        
        # Feature cache for TF-IDF and genome embeddings
        self.genre_tfidf_df = None
        self.genome_embeddings_df = None
        
        # Load models FIRST (critical)
        self._ensure_models_exist()
        self._load_models()
        self._load_feature_cache()
        
        # Load data SECOND (optional for predictions, required for recommendations)
        self._load_data()
        self._precompute_features()
        
        logger.info(f"✓ ModelService initialized. Ready: {self.is_ready()}")
    
    def _download_csv_from_gdrive(self, file_id: str, csv_name: str) -> Optional[pd.DataFrame]:
        """Download CSV from Google Drive"""
        try:
            logger.info(f"Downloading {csv_name} from Google Drive...")
            url = f"https://drive.google.com/uc?id={file_id}"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
            logger.info(f"✓ Downloaded {csv_name} ({len(df)} rows)")
            return df
        except Exception as e:
            logger.warning(f"Could not download {csv_name}: {e}")
            return None
    
    def _download_file_from_gdrive(self, file_id: str, output_path: Path) -> bool:
        """Download file from Google Drive using gdown"""
        try:
            logger.info(f"Downloading {output_path.name} from Google Drive...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(output_path), quiet=False)
            
            if output_path.exists():
                logger.info(f"✓ Downloaded {output_path.name}")
                return True
            return False
        except Exception as e:
            logger.warning(f"Could not download {output_path.name}: {e}")
            return False
    
    def _download_from_folder(self, folder_id: str, output_dir: Path) -> bool:
        """Download all files from Google Drive folder"""
        try:
            logger.info(f"Downloading models from Google Drive folder...")
            gdown.download_folder(
                url=f"https://drive.google.com/drive/folders/{folder_id}",
                output=str(output_dir),
                quiet=False,
                use_cookies=False
            )
            logger.info(f"✓ Downloaded all files from Google Drive")
            return True
        except Exception as e:
            logger.warning(f"Folder download failed: {e}")
            return False
    
    def _ensure_models_exist(self):
        """Check if model files exist, download if missing"""
        required_files = [
            "advanced_content_based_model.pkl",
            "content_scaler.pkl",
            "hybrid_collaborative_content_model.pkl",
            "hybrid_scaler.pkl",
            "svd_model.pkl",
            "feature_names.json"
        ]
        
        missing_files = [f for f in required_files if not (self.models_dir / f).exists()]
        
        if not missing_files:
            logger.info("✓ All model files present locally")
            return
        
        logger.warning(f"Missing {len(missing_files)} model files. Downloading...")
        
        try:
            if self._download_from_folder(GOOGLE_DRIVE_FOLDER_ID, self.models_dir):
                logger.info("✓ Successfully downloaded all models")
                return
        except Exception as e:
            logger.warning(f"Folder download failed: {e}")
        
        # Fallback: individual files
        for filename in missing_files:
            file_path = self.models_dir / filename
            file_id = self.MODEL_FILES.get(filename)
            if file_id:
                self._download_file_from_gdrive(file_id, file_path)
    
    def _load_data(self):
        """Load movies and ratings data (OPTIONAL - not required for predictions)"""
        try:
            logger.info("Loading data files...")
            
            movies_df = None
            ratings_df = None
            
            # Try Google Drive first
            if MOVIES_CSV_ID and MOVIES_CSV_ID != "movies.csv":
                movies_df = self._download_csv_from_gdrive(MOVIES_CSV_ID, "movies.csv")
            
            if RATINGS_CSV_ID and RATINGS_CSV_ID != "ratings.csv":
                ratings_df = self._download_csv_from_gdrive(RATINGS_CSV_ID, "ratings.csv")
            
            # Fallback to local
            if movies_df is None:
                movies_path = Path(__file__).parent.parent / "movies.csv"
                if movies_path.exists():
                    movies_df = pd.read_csv(movies_path)
                    logger.info(f"✓ Loaded {len(movies_df)} movies from local")
            
            if ratings_df is None:
                ratings_path = Path(__file__).parent.parent / "ratings.csv"
                if ratings_path.exists():
                    ratings_df = pd.read_csv(ratings_path, nrows=500000)
                    logger.info(f"✓ Loaded {len(ratings_df)} ratings from local")
            
            # Process if loaded
            if movies_df is not None:
                self.movies_df = movies_df
                self.movies_df['genres_list'] = self.movies_df['genres'].str.split('|')
                self.movies_df['release_year'] = self.movies_df['title'].str.extract(r'\((\d{4})\)').astype(float)
                self.movies_df['genre_count'] = self.movies_df['genres_list'].apply(len)
                logger.info(f"✓ Processed {len(self.movies_df)} movies")
            else:
                logger.warning("⚠ Movies data not loaded (recommendations may be limited)")
            
            if ratings_df is not None:
                self.ratings_df = ratings_df
                self.ratings_df['date'] = pd.to_datetime(self.ratings_df['timestamp'], unit='s')
                logger.info(f"✓ Processed {len(self.ratings_df)} ratings")
            else:
                logger.warning("⚠ Ratings data not loaded (will use default features)")
        
        except Exception as e:
            logger.warning(f"Error loading data: {e}")
    
    def _load_models(self):
        """Load trained models and scalers"""
        try:
            # Load feature names
            feature_file = self.models_dir / "feature_names.json"
            if feature_file.exists():
                with open(feature_file, 'r') as f:
                    self.feature_names = json.load(f)
            
            # Load content-based model
            content_model_path = self.models_dir / "advanced_content_based_model.pkl"
            content_scaler_path = self.models_dir / "content_scaler.pkl"
            
            if content_model_path.exists() and content_scaler_path.exists():
                self.models['content'] = joblib.load(content_model_path)
                self.scalers['content'] = joblib.load(content_scaler_path)
                logger.info("✓ Loaded content-based model")
            else:
                logger.error("✗ Content-based model files not found!")
            
            # Load hybrid model
            hybrid_model_path = None
            for path in [
                self.models_dir / "hybrid_collaborative_content_model.pkl",
                self.models_dir / "1hybrid_collaborative_content_model.pkl",
            ]:
                if path.exists():
                    hybrid_model_path = path
                    break
            
            hybrid_scaler_path = None
            for path in [
                self.models_dir / "hybrid_scaler.pkl",
                self.models_dir / "1hybrid_scaler.pkl",
            ]:
                if path.exists():
                    hybrid_scaler_path = path
                    break
            
            if hybrid_model_path and hybrid_scaler_path:
                self.models['hybrid'] = joblib.load(hybrid_model_path)
                self.scalers['hybrid'] = joblib.load(hybrid_scaler_path)
                logger.info("✓ Loaded hybrid model")
            else:
                logger.warning("⚠ Hybrid model not found (will use content-based fallback)")
        
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def _load_feature_cache(self):
        """Load pre-computed movie features (optional)"""
        feature_cache_path = self.data_dir / "movie_features.pkl"
        
        if feature_cache_path.exists():
            try:
                feature_cache = joblib.load(feature_cache_path)
                self.genre_tfidf_df = feature_cache.get('genre_tfidf_df')
                self.genome_embeddings_df = feature_cache.get('genome_embeddings_df')
                
                if self.genre_tfidf_df is not None:
                    logger.info(f"✓ Loaded feature cache")
            except Exception as e:
                logger.warning(f"Could not load feature cache: {e}")
        else:
            logger.info("⚠ Feature cache not found (will use zero vectors)")
    
    def _precompute_features(self):
        """Pre-compute movie features if data is available"""
        if self.movies_df is None or self.ratings_df is None:
            logger.info("⚠ Cannot pre-compute features (data not loaded)")
            return
        
        try:
            movie_stats = self.ratings_df.groupby('movieId').agg({
                'rating': ['mean', 'std', 'count'],
                'userId': 'nunique'
            }).reset_index()
            movie_stats.columns = ['movieId', 'movie_avg_rating', 'movie_rating_std', 
                                 'movie_num_ratings', 'movie_unique_users']
            
            # Rating momentum
            ratings_sorted = self.ratings_df.sort_values('date')
            recent_ratings = ratings_sorted.groupby('movieId').tail(10)
            recent_stats = recent_ratings.groupby('movieId')['rating'].mean().reset_index()
            recent_stats.columns = ['movieId', 'movie_recent_avg_rating']
            
            movie_stats = movie_stats.merge(recent_stats, on='movieId', how='left')
            movie_stats['movie_rating_momentum'] = (
                movie_stats['movie_recent_avg_rating'] - movie_stats['movie_avg_rating']
            ).fillna(0)
            
            # Genre popularity
            ratings_with_genres = self.ratings_df.merge(
                self.movies_df[['movieId', 'genres_list']], on='movieId', how='left'
            )
            genre_ratings = ratings_with_genres.explode('genres_list')
            genre_ratings = genre_ratings[genre_ratings['genres_list'].notna()]
            genre_popularity = genre_ratings.groupby('genres_list')['rating'].mean().to_dict()
            
            def calc_genre_popularity(genres_list):
                if not isinstance(genres_list, list) or len(genres_list) == 0:
                    return 3.5
                genre_scores = [genre_popularity.get(g, 3.5) for g in genres_list]
                return np.mean(genre_scores) if genre_scores else 3.5
            
            movie_stats['movie_genre_popularity'] = self.movies_df['genres_list'].apply(calc_genre_popularity)
            
            self.movie_features_cache = movie_stats.set_index('movieId').to_dict('index')
            logger.info(f"✓ Pre-computed features for {len(self.movie_features_cache)} movies")
        
        except Exception as e:
            logger.warning(f"Error pre-computing features: {e}")
    
    def _get_user_features(self, movie_ids: List[int]) -> Dict:
        """Extract user features from selected movies"""
        if not movie_ids:
            return {
                'user_avg_rating': 3.5,
                'user_rating_std': 0.0,
                'user_min_rating': 3.5,
                'user_max_rating': 3.5,
                'user_num_ratings': len(movie_ids),
                'user_unique_movies': len(set(movie_ids)),
                'user_rating_variance': 0.0,
                'user_genre_diversity': 0,
                'user_ratings_per_day': 0.0
            }
        
        # Use pre-computed cache if available
        user_ratings = None
        if self.ratings_df is not None:
            user_ratings = self.ratings_df[self.ratings_df['movieId'].isin(movie_ids)]
        
        if user_ratings is None or len(user_ratings) == 0:
            # Fallback: use movie averages from cache
            movie_ratings = []
            for mid in movie_ids:
                movie_data = self.movie_features_cache.get(mid, {})
                avg_rating = movie_data.get('movie_avg_rating', 3.5)
                movie_ratings.append(avg_rating)
            
            ratings_array = np.array(movie_ratings) if movie_ratings else np.array([3.5])
        else:
            ratings_array = user_ratings['rating'].values
        
        user_features = {
            'user_avg_rating': float(np.mean(ratings_array)),
            'user_rating_std': float(np.std(ratings_array)) if len(ratings_array) > 1 else 0.0,
            'user_min_rating': float(np.min(ratings_array)),
            'user_max_rating': float(np.max(ratings_array)),
            'user_num_ratings': len(movie_ids),
            'user_unique_movies': len(set(movie_ids)),
            'user_rating_variance': float(np.var(ratings_array)) if len(ratings_array) > 1 else 0.0,
        }
        
        # Genre diversity
        if self.movies_df is not None:
            user_movies = self.movies_df[self.movies_df['movieId'].isin(movie_ids)]
            all_genres = []
            for genres_list in user_movies['genres_list']:
                if isinstance(genres_list, list):
                    all_genres.extend(genres_list)
            user_features['user_genre_diversity'] = len(set(all_genres))
        else:
            user_features['user_genre_diversity'] = len(movie_ids)
        
        user_features['user_ratings_per_day'] = len(movie_ids) / 30.0
        
        return user_features
    
    def _get_movie_features(self, movie_id: int) -> Dict:
        """Get pre-computed features for a movie"""
        if movie_id in self.movie_features_cache:
            return self.movie_features_cache[movie_id]
        
        # Default features if not in cache
        return {
            'movie_avg_rating': 3.5,
            'movie_rating_std': 0.5,
            'movie_num_ratings': 100,
            'movie_unique_users': 50,
            'movie_rating_momentum': 0.0,
            'movie_genre_popularity': 3.5
        }
    
    def _prepare_content_features(self, user_movie_ids: List[int], target_movie_id: int) -> np.ndarray:
        """Prepare feature vector for content-based model"""
        user_features = self._get_user_features(user_movie_ids)
        movie_features = self._get_movie_features(target_movie_id)
        
        # Get target movie info
        release_year = 2000
        genre_count = 2
        movie_age = 24
        
        if self.movies_df is not None:
            movie_row = self.movies_df[self.movies_df['movieId'] == target_movie_id]
            if not movie_row.empty:
                release_year = float(movie_row.iloc[0].get('release_year', 2000) or 2000)
                genre_count = int(movie_row.iloc[0].get('genre_count', 2))
                movie_age = 2024 - release_year
        
        now = datetime.now()
        days_since_join = 30
        rating_velocity = len(user_movie_ids) / max(days_since_join, 1)
        
        # TF-IDF and genome embeddings (optional)
        if self.genre_tfidf_df is not None and target_movie_id in self.genre_tfidf_df.index:
            genre_tfidf = self.genre_tfidf_df.loc[target_movie_id].values
        else:
            genre_tfidf = np.zeros(50)
        
        if self.genome_embeddings_df is not None and target_movie_id in self.genome_embeddings_df.index:
            genome_emb = self.genome_embeddings_df.loc[target_movie_id].values
        else:
            genome_emb = np.zeros(50)
        
        features = np.array([
            user_features['user_avg_rating'],
            user_features['user_rating_std'],
            user_features['user_min_rating'],
            user_features['user_max_rating'],
            user_features['user_num_ratings'],
            user_features['user_unique_movies'],
            user_features['user_rating_variance'],
            user_features['user_genre_diversity'],
            user_features['user_ratings_per_day'],
            release_year,
            genre_count,
            movie_age,
            movie_features['movie_avg_rating'],
            movie_features['movie_num_ratings'],
            movie_features['movie_rating_momentum'],
            movie_features['movie_genre_popularity'],
            movie_age,
            days_since_join,
            rating_velocity,
            now.year,
            now.month,
            now.hour,
            1 if now.weekday() >= 5 else 0,
        ] + list(genre_tfidf) + list(genome_emb))
        
        return features
    
    def _prepare_hybrid_features(self, user_movie_ids: List[int], target_movie_id: int) -> np.ndarray:
        """Prepare feature vector for hybrid model"""
        user_features = self._get_user_features(user_movie_ids)
        movie_features = self._get_movie_features(target_movie_id)
        
        svd_prediction = 3.5
        
        now = datetime.now()
        movie_age_at_rating = 24
        
        if self.movies_df is not None:
            movie_row = self.movies_df[self.movies_df['movieId'] == target_movie_id]
            if not movie_row.empty:
                release_year = float(movie_row.iloc[0].get('release_year', 2000) or 2000)
                movie_age_at_rating = 2024 - release_year
        
        days_since_join = 30
        rating_velocity = len(user_movie_ids) / max(days_since_join, 1)
        
        features = np.array([
            svd_prediction,
            user_features['user_avg_rating'],
            user_features['user_rating_variance'],
            user_features['user_genre_diversity'],
            movie_features['movie_avg_rating'],
            movie_features['movie_num_ratings'],
            movie_features['movie_rating_momentum'],
            movie_features['movie_genre_popularity'],
            movie_age_at_rating,
            days_since_join,
            rating_velocity
        ])
        
        return features
    
    def predict_rating(self, user_movie_ids: List[int], target_movie_id: int, 
                      model_type: str = 'content') -> Tuple[float, float]:
        """
        Predict rating for a user-movie pair.
        Returns (predicted_rating, confidence)
        
        WORKS WITHOUT FULL DATASET!
        """
        if model_type not in self.models:
            if 'content' in self.models:
                logger.warning(f"Model '{model_type}' not available, using content-based")
                model_type = 'content'
            else:
                logger.error("No models loaded!")
                return 3.5, 0.5
        
        try:
            if model_type == 'content':
                features = self._prepare_content_features(user_movie_ids, target_movie_id)
                scaler = self.scalers.get('content')
            else:
                features = self._prepare_hybrid_features(user_movie_ids, target_movie_id)
                scaler = self.scalers.get('hybrid')
            
            if scaler is None:
                logger.error(f"Scaler for '{model_type}' model not found")
                return 3.5, 0.5
            
            features_scaled = scaler.transform([features])
            model = self.models[model_type]
            predicted_rating = float(model.predict(features_scaled)[0])
            
            # Clamp to valid range
            predicted_rating = max(0.5, min(5.0, predicted_rating))
            
            # Confidence based on number of rated movies
            confidence = 0.7 + 0.3 * (len(user_movie_ids) / 10.0)
            confidence = min(1.0, confidence)
            
            return predicted_rating, confidence
        
        except Exception as e:
            logger.error(f"Error predicting rating: {e}")
            return 3.5, 0.5
    
    def get_recommendations(self, user_movie_ids: List[int], top_n: int = 10,
                           model_type: str = 'content') -> List[Dict]:
        """Get top N recommendations for a user"""
        if model_type not in self.models:
            if 'content' in self.models:
                logger.warning(f"Model '{model_type}' not available, using content-based")
                model_type = 'content'
            else:
                logger.error("No models available")
                return []
        
        if self.movies_df is None:
            logger.warning("⚠ Movies data not available - using limited recommendations")
            # Still works, but returns empty
            return []
        
        try:
            all_movie_ids = self.movies_df['movieId'].tolist()
            candidate_movies = [m for m in all_movie_ids if m not in user_movie_ids]
            candidate_movies = candidate_movies[:1000]
            
            predictions = []
            for movie_id in candidate_movies:
                try:
                    pred_rating, confidence = self.predict_rating(
                        user_movie_ids, movie_id, model_type
                    )
                    predictions.append({
                        'movieId': movie_id,
                        'predicted_rating': pred_rating,
                        'confidence': confidence,
                        'score': pred_rating * confidence
                    })
                except Exception as e:
                    logger.debug(f"Error predicting for {movie_id}: {e}")
                    continue
            
            predictions.sort(key=lambda x: x['score'], reverse=True)
            
            recommendations = []
            for pred in predictions[:top_n]:
                movie_id = pred['movieId']
                movie_row = self.movies_df[self.movies_df['movieId'] == movie_id]
                
                if not movie_row.empty:
                    movie = movie_row.iloc[0]
                    recommendations.append({
                        'movieId': int(movie_id),
                        'title': movie['title'],
                        'genres': movie['genres'],
                        'year': int(movie.get('release_year', 0)) if pd.notna(movie.get('release_year')) else None,
                        'score': float(pred['score']),
                        'predicted_rating': float(pred['predicted_rating']),
                        'confidence': float(pred['confidence'])
                    })
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []
    
    def find_similar_movies(self, movie_id: int, top_n: int = 10) -> List[Dict]:
        """Find similar movies by genre and year"""
        if self.movies_df is None:
            logger.warning("Movies data not loaded")
            return []
        
        target_movie = self.movies_df[self.movies_df['movieId'] == movie_id]
        if target_movie.empty:
            return []
        
        target_genres = target_movie.iloc[0].get('genres_list', [])
        target_year = target_movie.iloc[0].get('release_year', 2000)
        
        similarities = []
        for _, movie in self.movies_df.iterrows():
            if movie['movieId'] == movie_id:
                continue
            
            movie_genres = movie.get('genres_list', [])
            genre_overlap = len(set(target_genres) & set(movie_genres))
            genre_similarity = genre_overlap / max(len(set(target_genres) | set(movie_genres)), 1)
            
            movie_year = movie.get('release_year', 2000)
            year_diff = abs(target_year - movie_year)
            year_similarity = 1.0 / (1.0 + year_diff / 10.0)
            
            similarity = 0.7 * genre_similarity + 0.3 * year_similarity
            
            similarities.append({
                'movieId': int(movie['movieId']),
                'title': movie['title'],
                'genres': movie['genres'],
                'year': int(movie.get('release_year', 0)) if pd.notna(movie.get('release_year')) else None,
                'similarity_score': float(similarity)
            })
        
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similarities[:top_n]
    
    def get_available_models(self) -> List[str]:
        """Get list of loaded models"""
        return list(self.models.keys())
    
    def is_ready(self) -> bool:
        """Check if at least one model is loaded"""
        return len(self.models) > 0

