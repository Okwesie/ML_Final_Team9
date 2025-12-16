const express = require('express');
const cors = require('cors');
const { PythonShell } = require('python-shell');
const path = require('path');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 5001;

app.use(cors());
app.use(express.json());

function callPythonScript(action, args = []) {
  return new Promise((resolve, reject) => {
    const options = {
      mode: 'text',
      pythonPath: process.platform === 'win32' ? 'python' : 'python3',
      pythonOptions: ['-u'],
      scriptPath: __dirname,
      args: [action, ...args]
    };

    PythonShell.run('predict.py', options, (err, results) => {
      if (err) {
        reject(err);
        return;
      }
      
      try {
        const result = JSON.parse(results.join(''));
        resolve(result);
      } catch (parseErr) {
        reject(new Error('Failed to parse Python output'));
      }
    });
  });
}

app.get('/api/recommendations', async (req, res) => {
  try {
    const n = req.query.n || 10;
    const genre = req.query.genre || null;
    
    const result = await callPythonScript('recommend', [n, genre || 'null']);
    
    if (result.error) {
      return res.status(500).json(result);
    }
    
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/search', async (req, res) => {
  try {
    const query = req.query.query || '';
    
    if (!query) {
      return res.json({ success: true, data: [] });
    }
    
    const result = await callPythonScript('search', [query]);
    
    if (result.error) {
      return res.status(500).json(result);
    }
    
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/similar/:movieId', async (req, res) => {
  try {
    const movieId = req.params.movieId;
    const n = req.query.n || 10;
    
    const result = await callPythonScript('similar', [movieId, n]);
    
    if (result.error) {
      return res.status(500).json(result);
    }
    
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/movie/:movieId', async (req, res) => {
  try {
    const movieId = req.params.movieId;
    
    const result = await callPythonScript('details', [movieId]);
    
    if (result.error) {
      return res.status(404).json(result);
    }
    
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', message: 'ReelSense API is running' });
});

app.listen(PORT, () => {
  console.log(`🚀 ReelSense backend running on http://localhost:${PORT}`);
  console.log(`📊 API endpoints available at http://localhost:${PORT}/api`);
});

