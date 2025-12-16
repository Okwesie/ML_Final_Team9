import { motion } from 'framer-motion';
import { useState } from 'react';

const MovieCard = ({ movie, onClick, index = 0 }) => {
  // Extract year from title or use release_year
  const year = movie.release_year || movie.title?.match(/\((\d{4})\)/)?.[1] || 'N/A';
  const title = movie.title?.replace(/\s*\(\d{4}\)\s*$/, '') || 'Unknown';

  // Generate a beautiful gradient placeholder based on movie title
  const getGradientColor = (str) => {
    // Create a hash from the title to get consistent colors
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = str.charCodeAt(i) + ((hash << 5) - hash);
    }
    
    // Generate colors from a palette
    const colors = [
      ['#e50914', '#b20710'], // Netflix red
      ['#8b5cf6', '#6d28d9'], // Purple
      ['#3b82f6', '#2563eb'], // Blue
      ['#10b981', '#059669'], // Green
      ['#f59e0b', '#d97706'], // Orange
      ['#ef4444', '#dc2626'], // Red
      ['#6366f1', '#4f46e5'], // Indigo
    ];
    
    const colorIndex = Math.abs(hash) % colors.length;
    return colors[colorIndex];
  };

  const [gradientColors] = useState(getGradientColor(title));

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.05 }}
      whileHover={{ scale: 1.05, y: -10 }}
      className="cursor-pointer"
      onClick={() => onClick && onClick(movie)}
    >
      <div className="relative group">
        {/* Poster with gradient background */}
        <div className="relative overflow-hidden rounded-lg aspect-[2/3] mb-3">
          <motion.div
            className="w-full h-full flex flex-col items-center justify-center p-4 text-center"
            style={{
              background: `linear-gradient(135deg, ${gradientColors[0]} 0%, ${gradientColors[1]} 100%)`,
            }}
            whileHover={{ scale: 1.05 }}
            transition={{ duration: 0.3 }}
          >
            {/* Movie Icon */}
            <div className="text-6xl mb-3 opacity-80">🎬</div>
            
            {/* Title on card */}
            <h3 className="text-white font-bold text-sm line-clamp-3 mb-2 drop-shadow-lg">
              {title}
            </h3>
            
            {/* Year badge */}
            {year !== 'N/A' && (
              <span className="px-3 py-1 bg-black/30 backdrop-blur-sm rounded-full text-white text-xs font-semibold">
                {year}
              </span>
            )}
          </motion.div>
          
          {/* Overlay on hover */}
          <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex flex-col items-center justify-end p-4">
            <p className="text-white text-sm font-semibold line-clamp-2 mb-2">{title}</p>
            {movie.avg_rating && (
              <div className="flex items-center gap-1">
                <span className="text-yellow-400">⭐</span>
                <span className="text-white text-sm font-semibold">{movie.avg_rating.toFixed(1)}</span>
              </div>
            )}
            {movie.num_ratings && (
              <span className="text-gray-300 text-xs mt-1">
                {movie.num_ratings.toLocaleString()} ratings
              </span>
            )}
          </div>
        </div>

        {/* Title and info below card */}
        <div className="text-center">
          <h3 className="text-white font-semibold text-sm mb-1 line-clamp-2">{title}</h3>
          <div className="flex items-center justify-center gap-3 text-xs text-gray-400">
            {year !== 'N/A' && <span>{year}</span>}
            {movie.avg_rating && (
              <span className="flex items-center gap-1">
                <span className="text-yellow-400">⭐</span>
                {movie.avg_rating.toFixed(1)}
              </span>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default MovieCard;

