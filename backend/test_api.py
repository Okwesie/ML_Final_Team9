"""
Test script for Movie Recommendation API
Aligned with synchronized main.py
"""
import requests
import json
import time

# Ensure this matches your running server port
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint"""
    print("\n" + "=" * 70)
    print("1. Testing /health endpoint")
    print("=" * 70)
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Connection Error: {e}")

def test_search():
    """Test movie search endpoint"""
    print("\n" + "=" * 70)
    print("2. Testing /api/movies/search")
    print("=" * 70)
    response = requests.get(f"{BASE_URL}/api/movies/search", params={"query": "Toy Story"})
    print(f"Status: {response.status_code}")
    data = response.json()
    results = data.get('results', [])
    print(f"Found {len(results)} movies")
    if results:
        print(f"First result: {results[0]['title']} ({results[0].get('year')})")
        print(f"Poster URL: {results[0].get('poster_url')}")

def test_movie_details():
    """Test get movie details (Verify the TypeError fix)"""
    print("\n" + "=" * 70)
    print("3. Testing /api/movies/1 (Detail Enrichment)")
    print("=" * 70)
    response = requests.get(f"{BASE_URL}/api/movies/1")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Movie: {data.get('title')}")
        print(f"Poster: {data.get('poster_url')}")
        print(f"Overview: {data.get('overview')[:100]}...")

def test_personalized_recommendations():
    """Test personalized recommendations"""
    print("\n" + "=" * 70)
    print("4. Testing /api/recommend/personalized")
    print("=" * 70)
    
    # Using 'movie_ids' to match the RecRequest Pydantic model
    payload = {
        "movie_ids": [1, 2, 480], # Toy Story, Jumanji, Jurassic Park
        "model": "hybrid",
        "top_n": 5
    }
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/api/recommend/personalized", json=payload)
    elapsed = (time.time() - start_time) * 1000
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        recs = data.get('recommendations', [])
        print(f"Received {len(recs)} recommendations in {elapsed:.2f}ms")
        if recs:
            print(f"Top Rec: {recs[0]['title']} - Predicted: {recs[0].get('predicted_rating')}")
    else:
        print(f"Error: {response.text}")

def test_similar_movies():
    """Test similarity endpoint"""
    print("\n" + "=" * 70)
    print("5. Testing /api/recommend/similar")
    print("=" * 70)
    
    payload = {
        "movie_id": 1,
        "top_n": 5
    }
    
    response = requests.post(f"{BASE_URL}/api/recommend/similar", json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        similar = data.get('similar_movies', [])
        print(f"Found {len(similar)} similar movies")
        if similar:
            print(f"Most similar: {similar[0]['title']}")
    else:
        print(f"Error: {response.text}")

def main():
    print("\nAPI TEST SUITE STARTING...")
    test_health()
    test_search()
    test_movie_details()
    test_personalized_recommendations()
    test_similar_movies()
    print("\n" + "=" * 70)
    print("TEST SUITE COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()