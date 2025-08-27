# Adaptive Keyboard Project

A web-based adaptive keyboard prototype with machine learning-powered word prediction.

## Current Status

This is a work-in-progress prototype demonstrating:
- ML-based next-word prediction using n-gram language models
- React frontend with adaptive keyboard UI
- Python backend with Flask API
- Word completion with tab autofill
- Basic typing analytics
  
<img width="1391" height="722" alt="Screenshot 2025-08-27 at 4 10 17 AM" src="https://github.com/user-attachments/assets/8a35ec9a-35aa-4e54-ab1a-f718dfdb9e1e" />
<img width="852" height="363" alt="Screenshot 2025-08-27 at 4 10 28 AM" src="https://github.com/user-attachments/assets/93611fb3-452a-4893-8ea3-6268d13b4f21" />


## Architecture

### Backend (Python)
- `api_server.py` - Flask API server
- `language_model.py` - Real ML n-gram language model
- `keyboard_engine.py` - Core prediction logic
- `word_dictionary.py` - Dictionary and word frequency handling

### Frontend (React)
- `react-demo/src/AdaptiveKeyboard.jsx` - Main keyboard component
- Runs on `http://localhost:3003`

## Machine Learning

- N-gram statistical language model (trigrams)
- Trained on 611 text samples with 505 word vocabulary
- Context-aware predictions with Laplace smoothing
- Model persistence (saves/loads trained state)

## Running the Demo

### Backend
```bash
cd "keyboard proj"
python3 api_server.py
```

### Frontend
```bash
cd react-demo
npm run dev
```

Then visit: `http://localhost:3003`

