# 🚀 ReelSense Setup Guide

## Quick Setup (5 minutes)

### Step 1: Install Backend Dependencies
```bash
cd backend
npm install
```

### Step 2: Install Frontend Dependencies
```bash
cd ../frontend
npm install
```

### Step 3: Configure Environment Variables

**Backend** - Create `backend/.env`:
```
PORT=5000
TMDB_API_KEY=your_key_here
```

**Frontend** - Create `frontend/.env`:
```
VITE_TMDB_API_KEY=your_key_here
```

> Get TMDB API key: https://www.themoviedb.org/settings/api (free, no credit card)

### Step 4: Verify ML Models Exist

Make sure these files exist in `../../models/`:
- `non_personalized_model.pkl` ✅ Required
- `content_based_best_model.pkl` ⚠️ Optional (will use fallback if missing)

And these files in `../../`:
- `movies.csv` ✅ Required
- `ratings.csv` ✅ Required

### Step 5: Run the Application

**Terminal 1 - Backend:**
```bash
cd backend
npm start
```
Should see: `🚀 ReelSense backend running on http://localhost:5000`

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```
Should see: `Local: http://localhost:3000`

### Step 6: Open Browser

Navigate to: **http://localhost:3000**

## Troubleshooting

### Backend won't start
- Check Python is installed: `python3 --version`
- Check Node.js: `node --version` (need 16+)
- Verify models exist: `ls ../../models/`

### Frontend won't start
- Check Node.js version
- Delete `node_modules` and run `npm install` again
- Check port 3000 is available

### No movies showing
- Verify `movies.csv` exists in parent directory
- Check backend console for errors
- Verify models are loaded (first request may take 10-20 seconds)

### TMDB API errors
- API key is optional - app works without it (uses placeholders)
- If you want real posters, get free API key from TMDB

## File Structure Check

Your directory should look like:
```
reelsense-website/
├── backend/
│   ├── server.js ✅
│   ├── predict.py ✅
│   ├── package.json ✅
│   └── .env (create this)
│
└── frontend/
    ├── src/
    │   ├── components/ ✅
    │   ├── App.jsx ✅
    │   └── main.jsx ✅
    ├── package.json ✅
    └── .env (create this)
```

## Next Steps

1. ✅ Install dependencies
2. ✅ Create .env files
3. ✅ Verify models/data exist
4. ✅ Start backend
5. ✅ Start frontend
6. 🎉 Enjoy your beautiful movie recommendation website!

---

**Need help?** Check the main README.md for more details.

