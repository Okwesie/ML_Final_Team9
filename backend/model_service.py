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
        "advanced_content_based_model.pkl": "1Abc123def456ghi789",  # Replace with actual IDs
        "content_scaler.pkl": "2Xyz789abc456def123",
        "hybrid_collaborative_content_model.pkl": "3Qwe456rty789uio012",
        "hybrid_scaler.pkl": "4Asd123fgh456jkl789",
        "svd_model.pkl": "5Zxc789vbn456mno012",
        "feature_names.json": "6Poi789uik456lmn012",
        "movie_features.pkl": "7Lkj456mno789pqr012"
    }
    
    def __init__(self, models_dir: str = None, data_dir: str = None):
        # Load from backend/models and backend/data directories
        if models_dir is None:
            models_dir = Path(__file__).parent / "models"  # backend/models
        if data_dir is None:
            data_dir = Path(__file__).parent / "data"      # backend/data
        
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        
        # Create directories if they don't exist
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
        
        self._load_data()
        self._ensure_models_exist()  # Download if missing
        self._load_models()
        self._load_feature_cache()
        self._precompute_features()
    
    def _download_csv_from_gdrive(self, file_id: str, csv_name: str) -> Optional[pd.DataFrame]:
        """
        Download a CSV file directly from Google Drive and load as DataFrame
        
        Args:
            file_id: Google Drive file ID
            csv_name: Name of the CSV file (for logging)
            
        Returns:
            pandas DataFrame or None if download fails
        """
        try:
            logger.info(f"Downloading {csv_name} from Google Drive...")
            url = f"https://drive.google.com/uc?id={file_id}"
            
            # Download directly to memory
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Load CSV from bytes
            df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
            logger.info(f"✓ Downloaded {csv_name} ({len(df)} rows)")
            return df
            
        except Exception as e:
            logger.error(f"Error downloading {csv_name} from Google Drive: {e}")
            return None
    
    def _download_file_from_gdrive(self, file_id: str, output_path: Path) -> bool:
        """
        Download a file from Google Drive using gdown
        
        Args:
            file_id: Google Drive file ID
            output_path: Path to save the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Downloading {output_path.name} from Google Drive...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(output_path), quiet=False)
            
            if output_path.exists():
                logger.info(f"✓ Downloaded {output_path.name} ({output_path.stat().st_size / 1e6:.1f} MB)")
                return True
            else:
                logger.error(f"✗ Download failed: {output_path.name} not created")
                return False
        except Exception as e:
            logger.error(f"Error downloading {output_path.name}: {e}")
            return False
    
    def _download_from_folder(self, folder_id: str, output_dir: Path) -> bool:
        """
        Download all files from a Google Drive folder
        
        Args:
            folder_id: Google Drive folder ID
            output_dir: Directory to save files
            
        Returns:
            True if all files downloaded successfully
        """
        try:
            logger.info(f"Downloading models from Google Drive folder {folder_id}...")
            gdown.download_folder(
                url=f"https://drive.google.com/drive/folders/{folder_id}",
                output=str(output_dir),
                quiet=False,
                use_cookies=False
            )
            logger.info(f"✓ Downloaded all files from Google Drive folder")
            return True
        except Exception as e:
            logger.error(f"Error downloading from Google Drive folder: {e}")
            return False
    
    def _ensure_models_exist(self):
        """
        Check if all required model files exist locally.
        If not, download them from Google Drive.
        """
        required_files = [
            "advanced_content_based_model.pkl",
            "content_scaler.pkl",
            "hybrid_collaborative_content_model.pkl",
            "hybrid_scaler.pkl",
            "svd_model.pkl",
            "feature_names.json"
        ]
        
        missing_files = []
        for filename in required_files:
            file_path = self.models_dir / filename
            if not file_path.exists():
                missing_files.append(filename)
        
        if not missing_files:
            logger.info("✓ All model files present locally")
            return
        
        logger.warning(f"Missing {len(missing_files)} model files: {missing_files}")
        logger.info("Attempting to download from Google Drive...")
        
        # Try downloading entire folder first (faster)
        try:
            success = self._download_from_folder(GOOGLE_DRIVE_FOLDER_ID, self.models_dir)
            if success:
                logger.info("✓ Successfully downloaded all models from Google Drive")
                return
        except Exception as e:
            logger.warning(f"Folder download failed, trying individual files: {e}")
        
        # Fallback: download individual files
        for filename in missing_files:
            file_path = self.models_dir / filename
            file_id = self.MODEL_FILES.get(filename)
            
            if file_id:
                self._download_file_from_gdrive(file_id, file_path)
            else:
                logger.warning(f"No Google Drive ID configured for {filename}")
        
        # Check if critical files were downloaded
        critical_files = [
            self.models_dir / "advanced_content_based_model.pkl",
            self.models_dir / "content_scaler.pkl"
        ]
        
        missing_critical = [f for f in critical_files if not f.exists()]
        if missing_critical:
            logger.error(f"Critical model files still missing: {missing_critical}")
            logger.error("Models will not be available. Please manually download from:")
            logger.error(f"https://drive.google.com/drive/folders/{GOOGLE_DRIVE_FOLDER_ID}")
    
    def _load_data(self):
        """Load movies and ratings data from Google Drive or local fallback"""
        try:
            # Try loading from Google Drive first
            logger.info("Attempting to load data from Google Drive...")
            
            movies_df = None
            ratings_df = None
            
            # Download movies.csv from Google Drive
            if MOVIES_CSV_ID and MOVIES_CSV_ID != "1xyz123movies":
                movies_df = self._download_csv_from_gdrive(MOVIES_CSV_ID, "movies.csv")
            
            # Download ratings.csv from Google Drive
            if RATINGS_CSV_ID and RATINGS_CSV_ID != "1xyz123ratings":
                ratings_df = self._download_csv_from_gdrive(RATINGS_CSV_ID, "ratings.csv")
            
            # Fallback to local files if Google Drive download fails
            if movies_df is None:
                logger.info("Falling back to local movies.csv...")
                movies_path = Path(__file__).parent.parent / "movies.csv"
                if movies_path.exists():
                    movies_df = pd.read_csv(movies_path)
                    logger.info(f"Loaded {len(movies_df)} movies from {movies_path}")
                else:
                    logger.warning(f"Movies file not found at {movies_path}")
            
            if ratings_df is None:
                logger.info("Falling back to local ratings.csv...")
                ratings_path = Path(__file__).parent.parent / "ratings.csv"
                if ratings_path.exists():
                    ratings_df = pd.read_csv(ratings_path, nrows=500000)
                    logger.info(f"Loaded {len(ratings_df)} ratings from {ratings_path}")
                else:
                    logger.warning(f"Ratings file not found at {ratings_path}")
            
            # Process movies dataframe
            if movies_df is not None:
                self.movies_df = movies_df
                self.movies_df['genres_list'] = self.movies_df['genres'].str.split('|')
                self.movies_df['release_year'] = self.movies_df['title'].str.extract(r'\((\d{4})\)').astype(float)
                self.movies_df['genre_count'] = self.movies_df['genres_list'].apply(len)
                logger.info(f"✓ Processed {len(self.movies_df)} movies")
            
            # Process ratings dataframe
            if ratings_df is not None:
                self.ratings_df = ratings_df
                self.ratings_df['date'] = pd.to_datetime(self.ratings_df['timestamp'], unit='s')
                logger.info(f"✓ Processed {len(self.ratings_df)} ratings")
        
        except Exception as e:
            logger.error(f"Error loading data: {e}")
    
    def _load_models(self):
        """Load all trained models and scalers from backend/models directory"""
        try:
            # Load feature names
            feature_file = self.models_dir / "feature_names.json"
            if feature_file.exists():
                with open(feature_file, 'r') as f:
                    self.feature_names = json.load(f)
                logger.info("Loaded feature names")
            else:
                logger.warning(f"Feature names not found at {feature_file}")
            
            # Load Model 1: Content-Based
            content_model_path = self.models_dir / "advanced_content_based_model.pkl"
            content_scaler_path = self.models_dir / "content_scaler.pkl"
            
            if content_model_path.exists() and content_scaler_path.exists():
                try:
                    self.models['content'] = joblib.load(content_model_path)
                    self.scalers['content'] = joblib.load(content_scaler_path)
                    logger.info("✓ Loaded content-based model")
                except Exception as e:
                    logger.error(f"Error loading content-based model: {e}")
            else:
                logger.warning(f"Content-based model files not found")
                logger.warning(f"  Expected: {content_model_path}")
                logger.warning(f"  Expected: {content_scaler_path}")
            
            # Load Model 2: Hybrid (check for both naming patterns)
            hybrid_model_paths = [
                self.models_dir / "hybrid_collaborative_content_model.pkl",
                self.models_dir / "1hybrid_collaborative_content_model.pkl",
            ]
            hybrid_scaler_paths = [
                self.models_dir / "hybrid_scaler.pkl",
                self.models_dir / "1hybrid_scaler.pkl",
            ]
            svd_model_paths = [
                self.models_dir / "svd_model.pkl",
                self.models_dir / "1svd_model.pkl",
            ]
            
            hybrid_model_path = None
            hybrid_scaler_path = None
            svd_model_path = None
            
            # Find actual files
            for path in hybrid_model_paths:
                if path.exists():
                    hybrid_model_path = path
                    break
            
            for path in hybrid_scaler_paths:
                if path.exists():
                    hybrid_scaler_path = path
                    break
            
            for path in svd_model_paths:
                if path.exists():
                    svd_model_path = path
                    break
            
            if hybrid_model_path and hybrid_scaler_path:
                try:
                    self.models['hybrid'] = joblib.load(hybrid_model_path)
                    self.scalers['hybrid'] = joblib.load(hybrid_scaler_path)
                    logger.info("✓ Loaded hybrid model")
                    
                    if svd_model_path:
                        try:
                            self.models['svd'] = joblib.load(svd_model_path)
                            logger.info("✓ Loaded hybrid model with SVD")
                        except Exception as svd_error:
                            logger.warning(f"Could not load SVD model: {svd_error}")
                    else:
                        logger.warning("SVD model not found, hybrid model will work without SVD component")
                except (ModuleNotFoundError, AttributeError) as e:
                    if '_loss' in str(e) or 'sklearn' in str(e).lower():
                        logger.error(f"scikit-learn version mismatch: {e}")
                        logger.error("Solution: Re-save models with current scikit-learn v1.5.1")
                        logger.warning("Hybrid model will not be available. Using content-based fallback.")
                    else:
                        raise
                except Exception as e:
                    logger.error(f"Error loading hybrid model: {e}")
                    logger.warning("Hybrid model will not be available. Using content-based fallback.")
            else:
                logger.warning("Hybrid model files not found in backend/models/")
                logger.warning(f"  Checked for: hybrid_collaborative_content_model.pkl")
                logger.warning(f"  Checked for: hybrid_scaler.pkl")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def _load_feature_cache(self):
        """Load pre-computed TF-IDF and genome embeddings from backend/data directory"""
        feature_cache_path = self.data_dir / "movie_features.pkl"
        
        if feature_cache_path.exists():
            try:
                feature_cache = joblib.load(feature_cache_path)
                self.genre_tfidf_df = feature_cache.get('genre_tfidf_df')
                self.genome_embeddings_df = feature_cache.get('genome_embeddings_df')
                
                if self.genre_tfidf_df is not None and self.genome_embeddings_df is not None:
                    logger.info(f"✓ Loaded feature cache from {feature_cache_path}")
                    logger.info(f"  - Genre TF-IDF shape: {self.genre_tfidf_df.shape}")
                    logger.info(f"  - Genome embeddings shape: {self.genome_embeddings_df.shape}")
                else:
                    logger.warning("Feature cache loaded but missing TF-IDF or genome embeddings")
            except Exception as e:
                logger.warning(f"Could not load feature cache: {e}")
                logger.warning("Using simplified features (zeros) - predictions may be less accurate")
        else:
            logger.warning(f"Feature cache not found at {feature_cache_path}")
            logger.warning("Using simplified features (zeros) - predictions may be less accurate")
    
    def _precompute_features(self):
        """Pre-compute movie-level features for faster predictions"""
        if self.movies_df is None or self.ratings_df is None:
            logger.warning("Cannot pre-compute features: movies_df or ratings_df is None")
            return
        
        try:
            # Compute movie statistics
            movie_stats = self.ratings_df.groupby('movieId').agg({
                'rating': ['mean', 'std', 'count'],
                'userId': 'nunique'
            }).reset_index()
            movie_stats.columns = ['movieId', 'movie_avg_rating', 'movie_rating_std', 
                                 'movie_num_ratings', 'movie_unique_users']
            
            # Rating momentum (recent ratings)
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
                    return 0
                genre_scores = [genre_popularity.get(g, 0) for g in genres_list]
                return np.mean(genre_scores) if genre_scores else 0
            
            movie_stats['movie_genre_popularity'] = self.movies_df['genres_list'].apply(calc_genre_popularity)
            
            # Merge with movies
            self.movie_features_cache = movie_stats.set_index('movieId').to_dict('index')
            logger.info(f"Pre-computed features for {len(self.movie_features_cache)} movies")
            
        except Exception as e:
            logger.error(f"Error pre-computing features: {e}")
    
    def _get_user_features(self, movie_ids: List[int]) -> Dict:
        """Extract user features from selected movies"""
        if not movie_ids or self.ratings_df is None:
            return {
                'user_avg_rating': 3.5,
                'user_rating_std': 0.0,
                'user_min_rating': 3.5,
                'user_max_rating': 3.5,
                'user_num_ratings': 0,
                'user_unique_movies': 0,
                'user_rating_variance': 0.0,
                'user_genre_diversity': 0,
                'user_ratings_per_day': 0.0
            }
        
        user_ratings = self.ratings_df[self.ratings_df['movieId'].isin(movie_ids)]
        
        if len(user_ratings) == 0:
            movie_ratings = []
            for mid in movie_ids:
                movie_data = self.movie_features_cache.get(mid, {})
                avg_rating = movie_data.get('movie_avg_rating', 3.5)
                movie_ratings.append(avg_rating)
            
            if movie_ratings:
                ratings_array = np.array(movie_ratings)
            else:
                ratings_array = np.array([3.5])
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
        
        if self.movies_df is not None:
            user_movies = self.movies_df[self.movies_df['movieId'].isin(movie_ids)]
            all_genres = []
            for genres_list in user_movies['genres_list']:
                if isinstance(genres_list, list):
                    all_genres.extend(genres_list)
            user_features['user_genre_diversity'] = len(set(all_genres))
        else:
            user_features['user_genre_diversity'] = 0
        
        user_features['user_ratings_per_day'] = len(movie_ids) / 30.0
        
        return user_features
    
    def _get_movie_features(self, movie_id: int) -> Dict:
        """Get pre-computed features for a movie"""
        if movie_id in self.movie_features_cache:
            return self.movie_features_cache[movie_id]
        
        return {
            'movie_avg_rating': 3.5,
            'movie_rating_std': 0.5,
            'movie_num_ratings': 10,
            'movie_unique_users': 5,
            'movie_rating_momentum': 0.0,
            'movie_genre_popularity': 3.5
        }
    
    def _prepare_content_features(self, user_movie_ids: List[int], target_movie_id: int) -> np.ndarray:
        """Prepare feature vector for content-based model"""
        user_features = self._get_user_features(user_movie_ids)
        movie_features = self._get_movie_features(target_movie_id)
        
        movie_row = self.movies_df[self.movies_df['movieId'] == target_movie_id]
        if movie_row.empty:
            release_year = 2000
            genre_count = 2
            movie_age = 24
        else:
            release_year = float(movie_row.iloc[0].get('release_year', 2000) or 2000)
            genre_count = int(movie_row.iloc[0].get('genre_count', 2))
            movie_age = 2024 - release_year
        
        now = datetime.now()
        movie_age_at_rating = movie_age
        days_since_join = 30
        rating_velocity = len(user_movie_ids) / max(days_since_join, 1)
        
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
            movie_age_at_rating,
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
        if 'svd' in self.models and user_movie_ids:
            try:
                svd_prediction = movie_features.get('movie_avg_rating', 3.5)
            except:
                pass
        
        now = datetime.now()
        movie_row = self.movies_df[self.movies_df['movieId'] == target_movie_id]
        if not movie_row.empty:
            release_year = float(movie_row.iloc[0].get('release_year', 2000) or 2000)
            movie_age_at_rating = 2024 - release_year
        else:
            movie_age_at_rating = 24
        
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
        """Predict rating for a user-movie pair. Returns (predicted_rating, confidence)"""
        if model_type not in self.models:
            if model_type == 'hybrid' and 'content' in self.models:
                logger.warning(f"Hybrid model not available, falling back to content-based model")
                model_type = 'content'
            elif model_type == 'content' and 'content' not in self.models:
                logger.error("Content-based model not available")
                return 3.5, 0.5
        
        if model_type not in self.models:
            return 3.5, 0.5
        
        try:
            if model_type == 'content':
                features = self._prepare_content_features(user_movie_ids, target_movie_id)
                scaler = self.scalers['content']
            else:
                features = self._prepare_hybrid_features(user_movie_ids, target_movie_id)
                scaler = self.scalers['hybrid']
            
            features_scaled = scaler.transform([features])
            model = self.models[model_type]
            predicted_rating = float(model.predict(features_scaled)[0])
            
            predicted_rating = max(0.5, min(5.0, predicted_rating))
            
            confidence = 0.7 + 0.3 * (len(user_movie_ids) / 10.0)
            confidence = min(1.0, confidence)
            
            return predicted_rating, confidence
            
        except Exception as e:
            logger.error(f"Error predicting rating: {e}")
            return 3.5, 0.5
    
    def get_recommendations(self, user_movie_ids: List[int], top_n: int = 10,
                           model_type: str = 'content') -> List[Dict]:
        """Get top N recommendations for a user based on selected movies"""
        if self.movies_df is None:
            logger.error("Movies data not loaded")
            return []
        
        if model_type not in self.models:
            if model_type == 'hybrid' and 'content' in self.models:
                logger.warning(f"Hybrid model not available, using content-based model")
                model_type = 'content'
            elif model_type not in self.models:
                logger.error(f"Model {model_type} not available")
                return []
        
        all_movie_ids = self.movies_df['movieId'].tolist()
        candidate_movies = [mid for mid in all_movie_ids if mid not in user_movie_ids]
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
                logger.error(f"Error predicting for movie {movie_id}: {e}")
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
                    'score': pred['score'],
                    'predicted_rating': pred['predicted_rating'],
                    'confidence': pred['confidence']
                })
        
        return recommendations
    
    def find_similar_movies(self, movie_id: int, top_n: int = 10) -> List[Dict]:
        """Find movies similar to a given movie using content-based features"""
        if self.movies_df is None:
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
                'similarity_score': similarity
            })
        
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similarities[:top_n]
    
    def get_available_models(self) -> List[str]:
        """Get list of loaded models"""
        return list(self.models.keys())
    
    def is_ready(self) -> bool:
        """Check if models are loaded and ready"""
        return len(self.models) > 0 and self.movies_df is not None

