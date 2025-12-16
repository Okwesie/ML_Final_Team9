import { motion } from 'framer-motion';
import { useState } from 'react';

const SearchBar = ({ onSearch, placeholder = "Search for movies..." }) => {
  const [query, setQuery] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query.trim());
    }
  };

  return (
    <motion.form
      onSubmit={handleSubmit}
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="w-full max-w-2xl mx-auto"
    >
      <div className="relative">
        <motion.input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={placeholder}
          className="w-full px-6 py-4 text-lg bg-black/40 backdrop-blur-md border-2 border-purple-accent/50 rounded-full text-white placeholder-gray-400 focus:outline-none focus:border-purple-accent focus:glow-purple transition-all duration-300"
          whileFocus={{ scale: 1.02 }}
        />
        <motion.button
          type="submit"
          className="absolute right-2 top-1/2 -translate-y-1/2 px-6 py-2 bg-gradient-to-r from-purple-accent to-netflix-red rounded-full text-white font-semibold hover:scale-105 transition-transform duration-200"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          Search
        </motion.button>
      </div>
    </motion.form>
  );
};

export default SearchBar;

