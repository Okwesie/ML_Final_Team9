# 🎬 ReelSense - Intelligent Movie Recommendation Website

A beautiful, modern movie recommendation website powered by machine learning models.

## 🚀 Quick Start

### Prerequisites
- Node.js 16+ and npm
- Python 3.8+
- ML models in `../../models/` folder
- `movies.csv` and `ratings.csv` in `../../` folder

### Installation

#### 1. Backend Setup
```bash
cd backend
npm install
```

#### 2. Frontend Setup
```bash
cd frontend
npm install
```

#### 3. Environment Variables

**Backend** (`backend/.env`):
```
PORT=5000
```

**Frontend** - No .env file needed! The app uses beautiful gradient placeholders instead of movie posters.

### Running the Application

#### Terminal 1 - Backend
```bash
cd backend
npm start
```

#### Terminal 2 - Frontend
```bash
cd frontend
npm run dev
```

The website will be available at `http://localhost:3000`

## 📁 Project Structure

```
reelsense-website/
├── backend/
│   ├── server.js          # Express API server
│   ├── predict.py         # Python script for ML models
│   ├── package.json
│   └── .env
│
└── frontend/
    ├── src/
    │   ├── components/
    │   │   ├── MovieCard.jsx
    │   │   └── SearchBar.jsx
    │   ├── App.jsx         # Main app with all pages
    │   ├── main.jsx
    │   └── index.css
    ├── index.html
    ├── package.json
    └── tailwind.config.js
```

## 🎨 Features

- **Homepage**: Animated hero section with search bar and trending movies carousel
- **Search**: Real-time movie search with beautiful grid layout
- **Movie Details**: Detailed movie information with similar movies recommendations
- **Animations**: Smooth Framer Motion animations throughout
- **Responsive**: Mobile-first design that works on all devices
- **Dark Theme**: Netflix-inspired dark theme with purple/red accents

## 🔌 API Endpoints

- `GET /api/recommendations?n=10&genre=Action` - Get top N recommendations
- `GET /api/search?query=inception` - Search for movies
- `GET /api/similar/:movieId?n=10` - Get similar movies
- `GET /api/movie/:movieId` - Get movie details
- `GET /api/health` - Health check

## 🛠️ Technology Stack

- **Frontend**: React, Vite, TailwindCSS, Framer Motion
- **Backend**: Node.js, Express, python-shell
- **ML**: Python, pandas, scikit-learn, joblib
- **Design**: Beautiful gradient placeholders for movie cards (no external API needed)

## 📝 Notes

- Make sure your ML models are trained and saved in `../../models/`
- The Python script loads models on first request (may take a few seconds)
- Beautiful gradient placeholders are used for movie cards (no external API needed)
- All pages are in `App.jsx` for simplicity

## 🎯 Next Steps

1. Add your TMDB API key to `.env` files
2. Ensure models are in `../../models/` folder
3. Run backend and frontend servers
4. Open `http://localhost:3000` and enjoy!

---

Built with ❤️ for ML Final Project

