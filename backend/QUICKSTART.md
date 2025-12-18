# Quick Start Guide

## 1. Prerequisites

- Python 3.8+
- Trained models from notebook (run `Team_9_Final_Project.ipynb` first)
- TMDB API key (optional but recommended)

## 2. Installation

```bash
cd backend
pip install -r requirements.txt
```

## 3. Configuration

Create `.env` file:
```bash
TMDB_API_KEY=your_key_here
ENVIRONMENT=development
PORT=8000
```

## 4. Start Server

```bash
uvicorn main:app --reload
```

## 5. Test

In another terminal:
```bash
python test_api.py
```

## 6. Access API

- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

## Example Request

```bash
curl -X POST "http://localhost:8000/api/recommend/personalized" \
  -H "Content-Type: application/json" \
  -d '{
    "movie_ids": [1, 2, 3],
    "model": "content",
    "top_n": 10
  }'
```

