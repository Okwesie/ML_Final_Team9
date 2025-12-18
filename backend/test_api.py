"""
Test script for Movie Recommendation API
Run this after starting the server to verify all endpoints work
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint"""
    print("=" * 70)
    print("Testing /health endpoint")
    print("=" * 70)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_search():
    """Test movie search endpoint"""
    print("=" * 70)
    print("Testing /api/movies/search endpoint")
    print("=" * 70)
    
    response = requests.get(f"{BASE_URL}/api/movies/search", params={"query": "toy story"})
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Found {data.get('count', 0)} movies")
    if data.get('results'):
        print(f"First result: {data['results'][0]['title']}")
    print()

def test_movie_details():
    """Test get movie details endpoint"""
    print("=" * 70)
    print("Testing /api/movies/{movie_id} endpoint")
    print("=" * 70)
    
    response = requests.get(f"{BASE_URL}/api/movies/1")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Movie: {data.get('title')}")
        print(f"Poster URL: {data.get('poster_url', 'N/A')}")
    print()

def test_personalized_recommendations():
    """Test personalized recommendations endpoint"""
    print("=" * 70)
    print("Testing /api/recommend/personalized endpoint")
    print("=" * 70)
    
    payload = {
        "movie_ids": [1, 2, 3],
        "model": "content",
        "top_n": 5
    }
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/api/recommend/personalized", json=payload)
    elapsed = (time.time() - start_time) * 1000
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Model used: {data.get('model_used')}")
        print(f"Computation time: {data.get('computation_time_ms', 0):.2f}ms")
        print(f"Actual request time: {elapsed:.2f}ms")
        print(f"Recommendations: {len(data.get('recommendations', []))}")
        
        if data.get('recommendations'):
            first_rec = data['recommendations'][0]
            print(f"Top recommendation: {first_rec.get('title')}")
            print(f"  Predicted rating: {first_rec.get('predicted_rating', 'N/A')}")
            print(f"  Poster URL: {first_rec.get('poster_url', 'N/A')}")
    else:
        print(f"Error: {response.text}")
    print()

def test_similar_movies():
    """Test similar movies endpoint"""
    print("=" * 70)
    print("Testing /api/recommend/similar endpoint")
    print("=" * 70)
    
    payload = {
        "movie_id": 1,
        "model": "content",
        "top_n": 5
    }
    
    response = requests.post(f"{BASE_URL}/api/recommend/similar", json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Target movie: {data.get('target_movie', {}).get('title')}")
        print(f"Similar movies: {len(data.get('similar_movies', []))}")
        
        if data.get('similar_movies'):
            first_similar = data['similar_movies'][0]
            print(f"Most similar: {first_similar.get('title')}")
            print(f"  Similarity score: {first_similar.get('similarity_score', 'N/A')}")
    else:
        print(f"Error: {response.text}")
    print()

def test_rating_prediction():
    """Test rating prediction endpoint"""
    print("=" * 70)
    print("Testing /api/predict/rating endpoint")
    print("=" * 70)
    
    payload = {
        "user_movie_ids": [1, 2, 3],
        "target_movie_id": 50,
        "model": "content"
    }
    
    response = requests.post(f"{BASE_URL}/api/predict/rating", json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Movie: {data.get('movie', {}).get('title')}")
        print(f"Predicted rating: {data.get('predicted_rating', 'N/A')}")
        print(f"Confidence: {data.get('confidence', 'N/A')}")
    else:
        print(f"Error: {response.text}")
    print()

def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("MOVIE RECOMMENDATION API TEST SUITE")
    print("=" * 70)
    print(f"Testing API at: {BASE_URL}")
    print("Make sure the server is running: uvicorn main:app --reload")
    print()
    
    try:
        test_health()
        test_search()
        test_movie_details()
        test_personalized_recommendations()
        test_similar_movies()
        test_rating_prediction()
        
        print("=" * 70)
        print("ALL TESTS COMPLETED")
        print("=" * 70)
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to server.")
        print("Make sure the server is running:")
        print("  cd backend")
        print("  uvicorn main:app --reload")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()

