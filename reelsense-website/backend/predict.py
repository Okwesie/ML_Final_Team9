import sys
import json
import pandas as pd
import joblib
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import warnings
warnings.filterwarnings('ignore')
class NonPersonalizedRecommender:
    def __init__(self, min_ratings: int = 50, percentile: int = 90):
        self.min_ratings = min_ratings
        self.percentile = percentile
        self.movie_scores = None
        
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        movie_stats = ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        movie_stats.columns = ['movieId', 'avg_rating', 'num_ratings']
        C = ratings_df['rating'].mean()
        m = movie_stats['num_ratings'].quantile(self.percentile / 100)
        movie_stats['weighted_score'] = (
            (movie_stats['num_ratings'] / (movie_stats['num_ratings'] + m)) * movie_stats['avg_rating'] +
            (m / (movie_stats['num_ratings'] + m)) * C
        )
        self.movie_scores = movie_stats.merge(
            movies_df[['movieId', 'title', 'genres', 'release_year']], 
            on='movieId'
        )
        self.movie_scores = self.movie_scores.sort_values('weighted_score', ascending=False)
        return self
    
    def recommend(self, n: int = 10, genre: str = None) -> pd.DataFrame:
        df = self.movie_scores.copy()
        if genre:
            df = df[df['genres'].str.contains(genre, case=False, na=False)]
        return df.head(n)[['movieId', 'title', 'genres', 'avg_rating', 
                          'num_ratings', 'weighted_score']]

models = {}
movies_df = None
ratings_df = None

def load_models():
    global models, movies_df, ratings_df
    
    if models:
        return
    
    models_dir = Path(__file__).parent.parent.parent / 'models'
    data_dir = Path(__file__).parent.parent.parent
    
    movies_path = data_dir / 'movies.csv'
    if movies_path.exists():
        movies_df = pd.read_csv(movies_path)
        movies_df['genres_list'] = movies_df['genres'].str.split('|')
        movies_df['genre_count'] = movies_df['genres_list'].apply(len)
        movies_df['release_year'] = movies_df['title'].str.extract(r'\((\d{4})\)').astype(float)
    
    ratings_path = data_dir / 'ratings.csv'
    if ratings_path.exists():
        ratings_df = pd.read_csv(ratings_path, nrows=100000)
    
    non_pers_path = models_dir / 'non_personalized_model.pkl'
    if non_pers_path.exists():
        models['non_personalized'] = joblib.load(non_pers_path)
    
    content_path = models_dir / 'content_based_best_model.pkl'
    if content_path.exists():
        try:
            models['content_based'] = joblib.load(content_path)
        except:
            pass

def handle_recommendations(n=10, genre=None):
    load_models()
    
    if 'non_personalized' not in models:
        return {'error': 'Model not found'}
    
    try:
        recommendations = models['non_personalized'].recommend(n=int(n), genre=genre)
        result = recommendations.to_dict('records')
        return {'success': True, 'data': result}
    except Exception as e:
        return {'error': str(e)}

def handle_search(query):
    load_models()
    
    if movies_df is None:
        return {'error': 'Movies data not found'}
    
    try:
        matching = movies_df[
            movies_df['title'].str.contains(query, case=False, na=False)
        ].head(20)
        
        result = matching[['movieId', 'title', 'genres', 'release_year']].to_dict('records')
        return {'success': True, 'data': result}
    except Exception as e:
        return {'error': str(e)}

def handle_similar(movie_id, n=10):
    load_models()
    
    if 'content_based' not in models:
        if movies_df is None:
            return {'error': 'Movies data not found'}
        
        try:
            target_movie = movies_df[movies_df['movieId'] == int(movie_id)]
            if len(target_movie) == 0:
                return {'error': 'Movie not found'}
            
            target_genres = set(target_movie.iloc[0]['genres_list'] if isinstance(target_movie.iloc[0]['genres_list'], list) else [])
            
            def genre_similarity(row):
                if row['movieId'] == int(movie_id):
                    return -1
                movie_genres = set(row['genres_list'] if isinstance(row['genres_list'], list) else [])
                intersection = len(target_genres & movie_genres)
                union = len(target_genres | movie_genres)
                return intersection / union if union > 0 else 0
            
            movies_df['similarity'] = movies_df.apply(genre_similarity, axis=1)
            similar = movies_df.nlargest(int(n) + 1, 'similarity').tail(int(n))
            result = similar[['movieId', 'title', 'genres', 'release_year']].to_dict('records')
            return {'success': True, 'data': result}
        except Exception as e:
            return {'error': str(e)}
    
    if movies_df is None:
        return {'error': 'Movies data not found'}
    
    try:
        similar = models['content_based'].find_similar_movies(int(movie_id), n=int(n))
        result = similar.to_dict('records')
        return {'success': True, 'data': result}
    except Exception as e:
        try:
            target_movie = movies_df[movies_df['movieId'] == int(movie_id)]
            if len(target_movie) == 0:
                return {'error': 'Movie not found'}
            
            target_genres = set(target_movie.iloc[0]['genres_list'] if isinstance(target_movie.iloc[0]['genres_list'], list) else [])
            
            def genre_similarity(row):
                if row['movieId'] == int(movie_id):
                    return -1
                movie_genres = set(row['genres_list'] if isinstance(row['genres_list'], list) else [])
                intersection = len(target_genres & movie_genres)
                union = len(target_genres | movie_genres)
                return intersection / union if union > 0 else 0
            
            movies_df['similarity'] = movies_df.apply(genre_similarity, axis=1)
            similar = movies_df.nlargest(int(n) + 1, 'similarity').tail(int(n))
            result = similar[['movieId', 'title', 'genres', 'release_year']].to_dict('records')
            return {'success': True, 'data': result}
        except Exception as e2:
            return {'error': str(e2)}

def get_movie_details(movie_id):
    """Get movie details by ID"""
    load_models()
    
    if movies_df is None:
        return {'error': 'Movies data not found'}
    
    try:
        movie = movies_df[movies_df['movieId'] == int(movie_id)]
        if len(movie) == 0:
            return {'error': 'Movie not found'}
        
        movie_data = movie.iloc[0]
        
        # Get rating stats if ratings available
        rating_info = {}
        if ratings_df is not None:
            movie_ratings = ratings_df[ratings_df['movieId'] == int(movie_id)]
            if len(movie_ratings) > 0:
                rating_info = {
                    'avg_rating': float(movie_ratings['rating'].mean()),
                    'num_ratings': int(len(movie_ratings))
                }
        
        result = {
            'movieId': int(movie_data['movieId']),
            'title': str(movie_data['title']),
            'genres': str(movie_data['genres']),
            'release_year': float(movie_data['release_year']) if pd.notna(movie_data['release_year']) else None,
            **rating_info
        }
        return {'success': True, 'data': result}
    except Exception as e:
        return {'error': str(e)}

# Main execution
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'No action specified'}))
        sys.exit(1)
    
    action = sys.argv[1]
    
    try:
        if action == 'recommend':
            n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            genre = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != 'null' else None
            result = handle_recommendations(n=n, genre=genre)
        
        elif action == 'search':
            query = sys.argv[2] if len(sys.argv) > 2 else ''
            result = handle_search(query)
        
        elif action == 'similar':
            movie_id = sys.argv[2] if len(sys.argv) > 2 else None
            n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
            if movie_id:
                result = handle_similar(movie_id, n=n)
            else:
                result = {'error': 'Movie ID required'}
        
        elif action == 'details':
            movie_id = sys.argv[2] if len(sys.argv) > 2 else None
            if movie_id:
                result = get_movie_details(movie_id)
            else:
                result = {'error': 'Movie ID required'}
        
        else:
            result = {'error': f'Unknown action: {action}'}
        
        print(json.dumps(result))
    
    except Exception as e:
        print(json.dumps({'error': str(e)}))

