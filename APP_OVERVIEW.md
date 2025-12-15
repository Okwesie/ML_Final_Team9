# Streamlit App Overview

## What the App Does

The Streamlit app is a **production-ready web interface** that lets users interact with the recommendation system built in the notebook. It provides:

1. **Interactive Recommendations** - Get top-N movie recommendations with filters
2. **Similar Movie Discovery** - Find movies similar to favorites
3. **Business Intelligence Dashboard** - View insights and analytics
4. **Model Performance Comparison** - See how different models perform

## App Structure

### Pages

1. **🏠 Home**
   - Platform overview
   - Quick stats
   - Setup instructions
   - Connection status

2. **🎯 Get Recommendations**
   - Top-N movie recommendations
   - Genre filtering
   - IMDB-weighted score display
   - Rating and popularity metrics

3. **🔍 Find Similar Movies**
   - Movie search functionality
   - Content-based similarity
   - Similar movies list
   - Genre and year information

4. **📊 Business Insights**
   - Executive summary
   - User segmentation
   - Genre performance
   - Hidden gems table

5. **📈 Model Performance**
   - Model comparison table
   - Performance metrics (RMSE, MAE, Accuracy)
   - Best model recommendation
   - Detailed analysis

### Key Features

- **Modern Dark Theme** - Professional gradient design
- **Interactive UI** - Buttons, sliders, search boxes
- **Real-time Loading** - Load models/data on demand
- **Error Handling** - Clear error messages
- **Responsive Design** - Works on different screen sizes

## How It Works

```
User Opens App
    ↓
App Checks for Models/Data
    ↓
User Clicks "Load Models" → Loads .pkl files
    ↓
User Clicks "Load Data" → Loads CSV/JSON files
    ↓
User Navigates Pages → Uses loaded models/data
    ↓
App Displays Results → Interactive visualizations
```

## Connection to Notebook

```
┌─────────────────────┐
│   Jupyter Notebook  │
│  (Training Phase)   │
│                     │
│  • Trains models    │
│  • Generates data   │
│  • Saves outputs    │
└──────────┬──────────┘
           │
           │ Creates
           ↓
┌─────────────────────┐
│   Files Created:    │
│                     │
│  models/            │
│  ├── *.pkl          │
│                     │
│  outputs/           │
│  ├── *.csv          │
│  └── *.json         │
└──────────┬──────────┘
           │
           │ Loads
           ↓
┌─────────────────────┐
│   Streamlit App     │
│  (Deployment Phase) │
│                     │
│  • Loads models     │
│  • Displays data    │
│  • User interaction │
└─────────────────────┘
```

## Data Flow

### Recommendations Flow:
```
User Input (N, genre)
    ↓
App calls: model.recommend(n, genre)
    ↓
Model returns: DataFrame with movies
    ↓
App displays: Formatted movie cards
```

### Similar Movies Flow:
```
User searches: "Toy Story"
    ↓
App finds: movie_id from movies_df
    ↓
App calls: model.find_similar_movies(movie_id)
    ↓
Model returns: Similar movies DataFrame
    ↓
App displays: Similar movies list
```

### Insights Flow:
```
User clicks: "Business Insights"
    ↓
App loads: outputs/executive_summary.json
    ↓
App displays: Metrics and charts
```

## Technical Details

### Dependencies
- `streamlit` - Web framework
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - ML models (for loading)
- `joblib` - Model serialization
- `plotly` - Visualizations (if needed)

### Caching
- Models cached with `@st.cache_resource`
- Data cached with `@st.cache_data`
- Fast reloading after first load

### Error Handling
- File not found errors
- Model loading errors
- Data processing errors
- User-friendly error messages

## Usage Example

```python
# 1. User starts app
streamlit run app.py

# 2. User clicks "Load Models" in sidebar
# → App loads: models/non_personalized_model.pkl
# → App loads: models/content_based_best_model.pkl

# 3. User clicks "Load Data" in sidebar
# → App loads: outputs/hidden_gems.csv
# → App loads: data/movies.csv

# 4. User navigates to "Get Recommendations"
# → User sets: N=10, genre="Action"
# → User clicks: "Get Recommendations"
# → App calls: model.recommend(10, "Action")
# → App displays: 10 action movie recommendations

# 5. User navigates to "Find Similar Movies"
# → User searches: "Matrix"
# → User clicks: "Find Similar Movies"
# → App finds: movie_id for "Matrix"
# → App calls: model.find_similar_movies(movie_id)
# → App displays: Similar sci-fi/action movies
```

## Benefits

1. **User-Friendly** - No coding required
2. **Interactive** - Real-time recommendations
3. **Professional** - Modern, polished interface
4. **Shareable** - Easy to demo to stakeholders
5. **Extensible** - Easy to add new features

## Future Enhancements

- User authentication
- Personalization based on user history
- Real-time model updates
- A/B testing interface
- Export recommendations to CSV
- Share recommendations via link

---

**The app transforms your notebook's models into an interactive, user-friendly interface!**


