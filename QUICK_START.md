# Quick Start Guide - Running the App

## Prerequisites

1. Python 3.8+ installed
2. MovieLens dataset files (ratings.csv, movies.csv, etc.)
3. Jupyter notebook environment (for running the notebook)

## Step-by-Step Setup

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install streamlit pandas numpy scikit-learn joblib plotly
```

### Step 2: Run the Notebook

**Important:** The app needs models and data from the notebook!

1. Open `Team_9_Final_Project.ipynb`
2. Run all cells from top to bottom
3. Wait for completion - this will create:
   - `models/` directory with trained models
   - `outputs/` directory with data files

**Expected Output:**
```
models/
├── non_personalized_model.pkl
└── content_based_best_model.pkl

outputs/
├── hidden_gems.csv
├── executive_summary.json
└── model_comparison.csv
```

### Step 3: Start the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Step 4: Load Models and Data

In the app sidebar:
1. Click **"🔄 Load Models"** button
2. Click **"📊 Load Data"** button
3. You should see success messages

### Step 5: Explore Features

Navigate using the sidebar:
- **🏠 Home**: Overview and setup instructions
- **🎯 Get Recommendations**: Top-N movie recommendations
- **🔍 Find Similar Movies**: Content-based movie discovery
- **📊 Business Insights**: Analytics dashboard
- **📈 Model Performance**: Compare ML models

## Troubleshooting

### App won't start

```bash
# Check if Streamlit is installed
pip install streamlit

# Try running with explicit Python
python -m streamlit run app.py
```

### "Models not found" error

**Solution:**
1. Make sure you ran the notebook completely
2. Check that `models/` directory exists in the same folder as `app.py`
3. Verify files exist: `ls models/` or `dir models\`

### "Data not found" error

**Solution:**
1. Run the notebook to generate outputs
2. Check that `outputs/` directory exists
3. Verify files: `ls outputs/` or `dir outputs\`

### Models load but recommendations don't work

**Solution:**
1. Make sure you used the same dataset in notebook
2. Check that model files aren't corrupted
3. Try re-running the notebook to regenerate models

### Port already in use

```bash
# Use a different port
streamlit run app.py --server.port 8502
```

## File Structure

Make sure your directory looks like this:

```
ML_Final_Group/
├── app.py                          # Streamlit app
├── Team_9_Final_Project.ipynb     # Jupyter notebook
├── requirements.txt                # Dependencies
├── data/                           # Dataset files
│   ├── movies.csv
│   ├── ratings.csv
│   └── ...
├── models/                         # Created by notebook
│   ├── non_personalized_model.pkl
│   └── content_based_best_model.pkl
└── outputs/                        # Created by notebook
    ├── hidden_gems.csv
    ├── executive_summary.json
    └── model_comparison.csv
```

## Common Workflows

### First Time Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run notebook (in Jupyter/Colab)
# Open Team_9_Final_Project.ipynb and run all cells

# 3. Start app
streamlit run app.py
```

### Daily Use (After Setup)
```bash
# Just start the app - models are already saved
streamlit run app.py
```

### After Modifying Notebook
```bash
# 1. Re-run notebook to regenerate models
# 2. Restart app (or just reload models in sidebar)
streamlit run app.py
```

## Features Overview

### 🎯 Get Recommendations
- Get top-N movie recommendations
- Filter by genre
- Based on IMDB-weighted score algorithm

### 🔍 Find Similar Movies
- Search for any movie
- Find similar movies using content-based filtering
- Uses Random Forest model

### 📊 Business Insights
- Executive summary metrics
- User segmentation analysis
- Genre performance data
- Hidden gems discovery

### 📈 Model Performance
- Compare all trained models
- View RMSE, MAE, Accuracy metrics
- See which model performs best

## Tips

1. **Always run notebook first** - The app needs the models it creates
2. **Check file paths** - Make sure data files are in the right location
3. **Use sidebar buttons** - Load models/data before using features
4. **Check console** - Error messages appear in terminal/console
5. **Reload if needed** - Use sidebar buttons to reload models/data

## Next Steps

- Explore all app features
- Try different recommendation counts
- Search for various movies
- Review business insights
- Compare model performance

---

**Need Help?** Check `APP_NOTEBOOK_CONNECTION.md` for detailed connection information.


