# Adaptive Keyboard Project

A web-based adaptive keyboard prototype with machine learning-powered word prediction.

## Current Status

This is a work-in-progress prototype demonstrating:
- ML-based next-word prediction using n-gram language models
- React frontend with adaptive keyboard UI
- Python backend with Flask API
- Word completion with tab autofill
- Basic typing analytics

## Architecture

### Backend (Python)
- `api_server.py` - Flask API server
- `language_model.py` - Real ML n-gram language model
- `keyboard_engine.py` - Core prediction logic
- `word_dictionary.py` - Dictionary and word frequency handling

### Frontend (React)
- `react-demo/src/AdaptiveKeyboard.jsx` - Main keyboard component
- Runs on `http://localhost:3003`

### API Endpoints
- `POST /predict` - Get word predictions
- `POST /analytics` - Update typing analytics
- `GET /health` - Health check

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

## Future Enhancements

- Swipe gesture recognition
- Multi-language support
- Cloud sync for typing patterns
- Advanced ML models (LSTM/Transformer)
- Voice-to-text integration
- Accessibility features

## License

MIT License
