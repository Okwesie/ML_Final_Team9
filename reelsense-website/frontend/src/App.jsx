import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import SearchBar from './components/SearchBar';
import MovieCard from './components/MovieCard';

const API_BASE = '/api';

function App() {
  const [currentPage, setCurrentPage] = useState('home'); // 'home', 'search', 'details'
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [trendingMovies, setTrendingMovies] = useState([]);
  const [selectedMovie, setSelectedMovie] = useState(null);
  const [similarMovies, setSimilarMovies] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Load trending movies on mount
  useEffect(() => {
    loadTrendingMovies();
  }, []);

  const loadTrendingMovies = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE}/recommendations?n=20`);
      if (response.data.success) {
        setTrendingMovies(response.data.data);
      }
    } catch (err) {
      console.error('Error loading trending movies:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async (query) => {
    if (!query.trim()) return;
    
    setSearchQuery(query);
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.get(`${API_BASE}/search?query=${encodeURIComponent(query)}`);
      if (response.data.success) {
        setSearchResults(response.data.data);
        setCurrentPage('search');
      } else {
        setError(response.data.error || 'Search failed');
      }
    } catch (err) {
      setError(err.message || 'Failed to search movies');
      console.error('Search error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleMovieClick = async (movie) => {
    setLoading(true);
    setError(null);
    
    try {
      // Get movie details
      const detailsResponse = await axios.get(`${API_BASE}/movie/${movie.movieId}`);
      if (detailsResponse.data.success) {
        setSelectedMovie(detailsResponse.data.data);
        
        // Get similar movies
        try {
          const similarResponse = await axios.get(`${API_BASE}/similar/${movie.movieId}?n=10`);
          if (similarResponse.data.success) {
            setSimilarMovies(similarResponse.data.data);
          }
        } catch (err) {
          console.error('Error loading similar movies:', err);
        }
        
        setCurrentPage('details');
      } else {
        setError('Movie not found');
      }
    } catch (err) {
      setError(err.message || 'Failed to load movie details');
      console.error('Movie details error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleBack = () => {
    if (currentPage === 'details') {
      setCurrentPage('search');
    } else {
      setCurrentPage('home');
      setSearchQuery('');
      setSearchResults([]);
    }
  };

  // Home Page
  const HomePage = () => (
    <div className="min-h-screen">
      {/* Hero Section */}
      <div className="relative h-screen flex items-center justify-center overflow-hidden">
        {/* Animated Gradient Background */}
        <motion.div
          className="absolute inset-0"
          animate={{
            background: [
              'linear-gradient(-45deg, #0a0a0a, #1a0f2e, #0f1c2e, #1a0a1f)',
              'linear-gradient(-45deg, #1a0a1f, #0a0a0a, #1a0f2e, #0f1c2e)',
              'linear-gradient(-45deg, #0f1c2e, #1a0a1f, #0a0a0a, #1a0f2e)',
              'linear-gradient(-45deg, #1a0f2e, #0f1c2e, #1a0a1f, #0a0a0a)',
            ],
          }}
          transition={{
            duration: 10,
            repeat: Infinity,
            repeatType: 'reverse',
          }}
          style={{
            backgroundSize: '400% 400%',
          }}
        />
        
        {/* Content */}
        <div className="relative z-10 w-full max-w-6xl mx-auto px-4 text-center">
          <motion.h1
            initial={{ opacity: 0, y: -30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-6xl md:text-8xl font-bold mb-6 bg-gradient-to-r from-purple-accent via-netflix-red to-purple-accent bg-clip-text text-transparent"
          >
            ReelSense
          </motion.h1>
          
          <motion.p
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="text-xl md:text-2xl text-gray-300 mb-12 font-light"
          >
            Discover Your Next Favorite Movie
          </motion.p>
          
          {/* Search Bar */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <SearchBar onSearch={handleSearch} />
          </motion.div>
        </div>
      </div>

      {/* Trending Now Section */}
      <div className="py-20 px-4">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="max-w-7xl mx-auto"
        >
          <h2 className="text-4xl font-bold mb-8 text-center bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
            Trending Now
          </h2>
          
          {loading && trendingMovies.length === 0 ? (
            <div className="flex justify-center items-center py-20">
              <div className="w-16 h-16 border-4 border-purple-accent border-t-transparent rounded-full animate-spin"></div>
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-6">
              {trendingMovies.map((movie, index) => (
                <MovieCard
                  key={movie.movieId}
                  movie={movie}
                  onClick={handleMovieClick}
                  index={index}
                />
              ))}
            </div>
          )}
        </motion.div>
      </div>
    </div>
  );

  // Search Results Page
  const SearchPage = () => (
    <div className="min-h-screen py-20 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <motion.button
            onClick={handleBack}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="mb-6 px-6 py-2 bg-gradient-to-r from-purple-accent to-netflix-red rounded-full text-white font-semibold hover:glow transition-all duration-200"
          >
            ← Back to Home
          </motion.button>
          
          <h2 className="text-4xl font-bold mb-4">
            Search Results for "{searchQuery}"
          </h2>
          <p className="text-gray-400">
            {searchResults.length} movies found
          </p>
        </div>

        {/* Search Bar */}
        <div className="mb-8">
          <SearchBar onSearch={handleSearch} />
        </div>

        {/* Results */}
        {loading ? (
          <div className="flex justify-center items-center py-20">
            <div className="w-16 h-16 border-4 border-purple-accent border-t-transparent rounded-full animate-spin"></div>
          </div>
        ) : error ? (
          <div className="text-center py-20">
            <p className="text-red-400 text-xl mb-4">Error: {error}</p>
            <button
              onClick={() => handleSearch(searchQuery)}
              className="px-6 py-2 bg-netflix-red rounded-full text-white font-semibold hover:glow transition-all"
            >
              Try Again
            </button>
          </div>
        ) : searchResults.length === 0 ? (
          <div className="text-center py-20">
            <p className="text-gray-400 text-xl">No movies found. Try a different search.</p>
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-6">
            {searchResults.map((movie, index) => (
              <MovieCard
                key={movie.movieId}
                movie={movie}
                onClick={handleMovieClick}
                index={index}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );

  // Movie Details Page
  const DetailsPage = () => {
    if (!selectedMovie) return null;

    const year = selectedMovie.release_year || selectedMovie.title?.match(/\((\d{4})\)/)?.[1] || 'N/A';
    const title = selectedMovie.title?.replace(/\s*\(\d{4}\)\s*$/, '') || 'Unknown';
    const genres = selectedMovie.genres?.split('|') || [];

    return (
      <div className="min-h-screen py-20 px-4">
        <div className="max-w-7xl mx-auto">
          {/* Back Button */}
          <motion.button
            onClick={handleBack}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="mb-8 px-6 py-2 bg-gradient-to-r from-purple-accent to-netflix-red rounded-full text-white font-semibold hover:glow transition-all duration-200"
          >
            ← Back
          </motion.button>

          {/* Movie Details */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass rounded-2xl p-8 mb-12"
          >
            <div className="flex flex-col md:flex-row gap-8">
              {/* Poster with gradient */}
              <div className="w-full md:w-64 flex-shrink-0">
                <div 
                  className="aspect-[2/3] rounded-lg flex flex-col items-center justify-center p-6 text-center"
                  style={{
                    background: `linear-gradient(135deg, #8b5cf6 0%, #e50914 100%)`,
                  }}
                >
                  <span className="text-6xl mb-4 opacity-90">🎬</span>
                  <h3 className="text-white font-bold text-lg line-clamp-2">{title}</h3>
                </div>
              </div>

              {/* Info */}
              <div className="flex-1">
                <h1 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
                  {title}
                </h1>
                
                <div className="flex flex-wrap items-center gap-4 mb-6">
                  {year !== 'N/A' && (
                    <span className="px-4 py-1 bg-netflix-red/20 border border-netflix-red rounded-full text-sm">
                      {year}
                    </span>
                  )}
                  {selectedMovie.avg_rating && (
                    <span className="px-4 py-1 bg-yellow-500/20 border border-yellow-500 rounded-full text-sm flex items-center gap-2">
                      <span>⭐</span>
                      {selectedMovie.avg_rating.toFixed(1)} / 5.0
                    </span>
                  )}
                  {selectedMovie.num_ratings && (
                    <span className="text-gray-400 text-sm">
                      {selectedMovie.num_ratings.toLocaleString()} ratings
                    </span>
                  )}
                </div>

                {/* Genres */}
                {genres.length > 0 && (
                  <div className="mb-6">
                    <h3 className="text-lg font-semibold mb-2">Genres</h3>
                    <div className="flex flex-wrap gap-2">
                      {genres.map((genre, idx) => (
                        <span
                          key={idx}
                          className="px-3 py-1 bg-purple-accent/20 border border-purple-accent rounded-full text-sm"
                        >
                          {genre}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                <div className="text-gray-300">
                  <p className="text-lg">
                    Movie ID: {selectedMovie.movieId}
                  </p>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Similar Movies */}
          {similarMovies.length > 0 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
            >
              <h2 className="text-3xl font-bold mb-6">Similar Movies</h2>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-6">
                {similarMovies.map((movie, index) => (
                  <MovieCard
                    key={movie.movieId}
                    movie={movie}
                    onClick={handleMovieClick}
                    index={index}
                  />
                ))}
              </div>
            </motion.div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-netflix-dark">
      <AnimatePresence mode="wait">
        {currentPage === 'home' && (
          <motion.div
            key="home"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <HomePage />
          </motion.div>
        )}
        
        {currentPage === 'search' && (
          <motion.div
            key="search"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
          >
            <SearchPage />
          </motion.div>
        )}
        
        {currentPage === 'details' && (
          <motion.div
            key="details"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
          >
            <DetailsPage />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;

