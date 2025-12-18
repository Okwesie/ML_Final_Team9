"""
Script to fix model version issues and improve feature preparation
Run this after the notebook to ensure models work with current scikit-learn version
"""
import sys
import os
from pathlib import Path

# Add parent directory to path to import from notebook context
sys.path.insert(0, str(Path(__file__).parent.parent))

def fix_models():
    """Re-save models with current scikit-learn version and save feature data"""
    print("=" * 70)
    print("FIXING MODELS AND IMPROVING FEATURE PREPARATION")
    print("=" * 70)
    
    try:
        import joblib
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler
        import sklearn
        print(f"Current scikit-learn version: {sklearn.__version__}")
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Install with: pip install scikit-learn joblib pandas numpy")
        return
    
    models_dir = Path("../models")
    data_dir = Path("../data")
    data_dir.mkdir(exist_ok=True)
    
    # Check if models exist
    hybrid_model_path = models_dir / "hybrid_collaborative_content_model.pkl"
    if not hybrid_model_path.exists():
        print("ERROR: Hybrid model not found. Run the notebook first.")
        return
    
    print("\n1. Loading existing hybrid model...")
    try:
        # Try to load with compatibility mode
        model = joblib.load(hybrid_model_path)
        print("   Model loaded successfully")
        
        # Re-save with current version
        print("\n2. Re-saving model with current scikit-learn version...")
        joblib.dump(model, hybrid_model_path)
        print(f"   ✓ Re-saved: {hybrid_model_path}")
        
    except Exception as e:
        print(f"   ERROR: Could not load model: {e}")
        print("\n   Solution: Re-run the notebook cell that trains model2_hybrid")
        print("   Then re-save it with: joblib.dump(model2_hybrid, 'models/hybrid_collaborative_content_model.pkl')")
        return
    
    # Check scaler
    hybrid_scaler_path = models_dir / "hybrid_scaler.pkl"
    if hybrid_scaler_path.exists():
        try:
            scaler = joblib.load(hybrid_scaler_path)
            joblib.dump(scaler, hybrid_scaler_path)
            print(f"   ✓ Re-saved scaler: {hybrid_scaler_path}")
        except Exception as e:
            print(f"   WARNING: Could not re-save scaler: {e}")
    
    print("\n3. Checking for feature data to improve accuracy...")
    
    # Check if we can load movies and create feature cache
    movies_path = Path("../movies.csv")
    if movies_path.exists():
        print("   Movies file found - can improve feature preparation")
        print("   See DEPLOYMENT_NOTES.md for instructions on saving TF-IDF/genome features")
    else:
        print("   WARNING: movies.csv not found in parent directory")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS FOR FULL ACCURACY:")
    print("=" * 70)
    print("1. In the notebook, after feature engineering, save TF-IDF and genome features:")
    print("""
   import joblib
   
   # Save feature data for backend
   feature_cache = {
       'genre_tfidf': genre_tfidf_df,
       'genome_embeddings': genome_embeddings_df,
       'movies_index': movies_df.set_index('movieId')
   }
   joblib.dump(feature_cache, 'data/movie_features.pkl')
   print("✓ Saved feature cache for backend")
""")
    print("\n2. Restart the backend server")
    print("\n3. The backend will automatically use these features for accurate predictions")

if __name__ == "__main__":
    fix_models()

