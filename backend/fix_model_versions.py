"""
Script to check and fix scikit-learn version compatibility issues
Run this to diagnose model loading problems
"""
import sys
import joblib
from pathlib import Path

def check_model_compatibility():
    """Check if models can be loaded with current scikit-learn version"""
    print("=" * 70)
    print("MODEL COMPATIBILITY CHECK")
    print("=" * 70)
    
    try:
        import sklearn
        print(f"Current scikit-learn version: {sklearn.__version__}")
    except ImportError:
        print("ERROR: scikit-learn not installed")
        return
    
    models_dir = Path("../models")
    
    models_to_check = [
        ("Content-Based Model", "advanced_content_based_model.pkl"),
        ("Hybrid Model", "hybrid_collaborative_content_model.pkl"),
        ("SVD Model", "svd_model.pkl"),
        ("Content Scaler", "content_scaler.pkl"),
        ("Hybrid Scaler", "hybrid_scaler.pkl"),
    ]
    
    print("\nChecking models:")
    print("-" * 70)
    
    for name, filename in models_to_check:
        model_path = models_dir / filename
        if not model_path.exists():
            print(f"❌ {name}: File not found")
            continue
        
        try:
            model = joblib.load(model_path)
            print(f"✅ {name}: Loaded successfully")
            
            # Try to get model type
            if hasattr(model, '__class__'):
                print(f"   Type: {model.__class__.__name__}")
        except ModuleNotFoundError as e:
            if '_loss' in str(e):
                print(f"❌ {name}: scikit-learn version mismatch")
                print(f"   Error: {e}")
                print(f"   Solution: Re-save model with current scikit-learn version")
            else:
                print(f"❌ {name}: Missing module - {e}")
        except Exception as e:
            print(f"❌ {name}: Error loading - {e}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS:")
    print("=" * 70)
    print("1. If you see '_loss' errors, the model was saved with a different")
    print("   scikit-learn version. You need to:")
    print("   a) Match scikit-learn versions, OR")
    print("   b) Re-save the model in the notebook with current version")
    print("\n2. The backend will automatically fall back to content-based model")
    print("   if hybrid model fails to load.")
    print("\n3. To fix: Run the notebook cell that saves the hybrid model again")

if __name__ == "__main__":
    check_model_compatibility()

