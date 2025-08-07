import React, { useState, useEffect, useCallback } from 'react';
import './AdaptiveKeyboard.css';

const AdaptiveKeyboard = () => {
  const [inputText, setInputText] = useState('');
  const [predictions, setPredictions] = useState({});
  const [keyHeat, setKeyHeat] = useState({});
  const [wordFrequency, setWordFrequency] = useState({});
  const [personalizedInsights, setPersonalizedInsights] = useState({
    learnedPatterns: 0,
    dictionaryCoverage: 0,
    vocabularyDiversity: 0,
    topWords: [],
    uniqueStyle: 'Ready to learn your patterns...',
    sessionTime: 0
  });
  
  // Add state for word completion
  const [currentSuggestion, setCurrentSuggestion] = useState('');
  const [keystrokesSaved, setKeystrokesSaved] = useState(0);

  // Keyboard layout
  const keyboardRows = [
    ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
    ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
    ['z', 'x', 'c', 'v', 'b', 'n', 'm']
  ];

  // API configuration
  const API_BASE_URL = 'http://localhost:3000';
  
  // Loading and error states
  const [isLoading, setIsLoading] = useState(false);
  const [apiError, setApiError] = useState(null);
  const [backendConnected, setBackendConnected] = useState(false);

  // Helper function to make API calls with detailed error logging
  const callAPI = async (endpoint, data = null) => {
    const url = `${API_BASE_URL}${endpoint}`;
    const options = {
      method: data ? 'POST' : 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      ...(data && { body: JSON.stringify(data) })
    };

    try {
      console.log(`🔄 API Call: ${endpoint}`, data ? { data } : '');
      const response = await fetch(url, options);
      
      if (!response.ok) {
        const errorText = await response.text();
        let errorDetails;
        try {
          errorDetails = JSON.parse(errorText);
        } catch {
          errorDetails = { error: errorText };
        }
        console.error(`❌ API Error (${endpoint}):`, {
          status: response.status,
          statusText: response.statusText,
          details: errorDetails
        });
        throw new Error(`API call failed: ${response.status}`);
      }
      
      const result = await response.json();
      console.log(`✅ API Success (${endpoint}):`, result);
      return result;
    } catch (error) {
      console.error(`🚨 API Exception (${endpoint}):`, error);
      throw error;
    }
  };

  // Generate predictions using Python backend API
  const generatePredictions = useCallback(async (text) => {
    if (!backendConnected) {
      // Fallback predictions when backend is not available
      return { 't': 0.25, 'a': 0.20, 'i': 0.15, 'w': 0.12, 'y': 0.10, 's': 0.08, 'h': 0.10 };
    }

    try {
      setIsLoading(true);
      const result = await callAPI('/predict', { text: text || '' });
      
      if (result && result.predictions) {
        setCurrentSuggestion(result.word_suggestion || '');
        console.log('Word suggestion from predict:', result.word_suggestion);
        
        return result.predictions;
      }
      
      // Fallback if API call fails
      setCurrentSuggestion(''); // Clear suggestion on fallback
      return { 't': 0.25, 'a': 0.20, 'i': 0.15, 'w': 0.12, 'y': 0.10, 's': 0.08, 'h': 0.10 };
    } catch (error) {
      console.error('Prediction error:', error);
      setCurrentSuggestion(''); // Clear suggestion on error
      return { 't': 0.25, 'a': 0.20, 'i': 0.15, 'w': 0.12, 'y': 0.10, 's': 0.08, 'h': 0.10 };
    } finally {
      setIsLoading(false);
    }
  }, [backendConnected]);

  // Calculate key heat based on word frequency (moved to avoid dependency issues)
  const calculateKeyHeat = useCallback(() => {
    const heat = {};
    const totalWords = Object.values(wordFrequency).reduce((sum, freq) => sum + freq, 0);
    
    if (totalWords < 3) return {};
    

    // Calculate character frequency from word usage
    Object.entries(wordFrequency).forEach(([word, freq]) => {
      [...word].forEach(char => {
        heat[char] = (heat[char] || 0) + freq;
      });
    });

    // Normalize to 0-1 range
    const maxHeat = Math.max(...Object.values(heat));
    Object.keys(heat).forEach(char => {
      heat[char] = (heat[char] / maxHeat) * 0.6;
    });

    return heat;
  }, [wordFrequency]);

  // Handle key press
  const handleKeyPress = (key) => {
    if (key === 'space') {
      setInputText(prev => prev + ' ');
    } else if (key === 'backspace') {
      setInputText(prev => prev.slice(0, -1));
    } else {
      setInputText(prev => prev + key);
    }
  };
  
  // Handle tab completion
  const handleTabCompletion = () => {
    if (currentSuggestion) {
      const words = inputText.split(' ');
      const currentWord = words[words.length - 1];
      const completedWord = currentSuggestion;
      
      // keystrokes
      const saved = completedWord.length - currentWord.length;
      if (saved > 0) {
        setKeystrokesSaved(prev => prev + saved);
      }
      
      // suggestion replacement
      words[words.length - 1] = completedWord;
      setInputText(words.join(' ') + ' ');
      setCurrentSuggestion('');
    }
  };
  
  // Handle keyboard events
  const handleKeyDown = (e) => {
    if (e.key === 'Tab' && currentSuggestion) {
      e.preventDefault();
      handleTabCompletion();
    }
  };

  // Check backend connection on component mount
  useEffect(() => {
    const checkBackendConnection = async () => {
      try {
        const result = await callAPI('/health');
        if (result && result.status === 'healthy') {
          setBackendConnected(true);
          console.log('Backend connected successfully!');
        }
      } catch (error) {
        console.log('Backend not available, using fallback mode');
        setBackendConnected(false);
      }
    };
    
    checkBackendConnection();
  }, []); 

  useEffect(() => {
    let isMounted = true;
    
    const words = inputText.split(' ');
    const currentWord = words[words.length - 1];
    if (inputText.endsWith(' ') || inputText.endsWith('.') || inputText.endsWith(',') || currentWord.length < 2) {
      setCurrentSuggestion('');
    }
    
    const updatePredictionsAsync = async () => {
      if (!isMounted) return;
      try {
        const newPredictions = await generatePredictions(inputText || '');
        if (isMounted && newPredictions) {
          setPredictions(newPredictions);
        }
      } catch (error) {
        console.error('Prediction update failed:', error);
        if (isMounted) {
          setPredictions({ 't': 0.25, 'a': 0.20, 'i': 0.15, 'w': 0.12, 'y': 0.10, 's': 0.08, 'h': 0.10 });
        }
      }
    };
    
    const updateInsightsAsync = async () => {
      if (!isMounted) return;
      
      if (inputText.endsWith(' ') || inputText.endsWith('.') || inputText.endsWith(',')) {
        setCurrentSuggestion('');
      }
      
      if (backendConnected && inputText) {
        try {
          const result = await callAPI('/insights', { text: inputText });
          if (result && isMounted) {
            setPersonalizedInsights(prev => ({
              ...prev,
              learnedPatterns: result.learned_patterns || prev.learnedPatterns,
              dictionaryCoverage: result.dictionary_coverage || prev.dictionaryCoverage,
              vocabularyDiversity: result.vocabulary_diversity || prev.vocabularyDiversity,
              topWords: result.top_words || prev.topWords,
              uniqueStyle: result.unique_style || prev.uniqueStyle
            }));
            
          }
        } catch (error) {
          console.error('Insights update failed:', error);
          setCurrentSuggestion(''); // Clear suggestion on error
        }
      } else {
        const words = inputText.trim().split(' ').filter(w => w.length > 0);
        const uniqueWords = new Set(words).size;
        const vocabDiversity = words.length > 0 ? Math.round((uniqueWords / words.length) * 100) : 0;
        if (isMounted) {
          setPersonalizedInsights(prev => ({
            ...prev,
            learnedPatterns: Math.max(uniqueWords - 1, 0),
            dictionaryCoverage: Math.min(uniqueWords * 20, 100),
            vocabularyDiversity: vocabDiversity,
            uniqueStyle: uniqueWords > 5 ? 'Diverse vocabulary detected' : 'Building patterns...'
          }));
        }
      }
    };
    
    const updateHeatAsync = async () => {
      if (!isMounted) return;
      if (backendConnected) {
        try {
          const result = await callAPI('/heat', { text: inputText });
          if (result && result.heat_map && isMounted) {
            setKeyHeat(result.heat_map);
          }
        } catch (error) {
          console.error('Heat update failed:', error);
          if (isMounted) {
            setKeyHeat(calculateKeyHeat());
          }
        }
      } else {
        if (isMounted) {
          setKeyHeat(calculateKeyHeat());
        }
      }
    };
    
    const updateWordFrequency = () => {
      if (!inputText) return;
      
      const lastChar = inputText[inputText.length - 1];
      if (lastChar === ' ' || lastChar === '.' || lastChar === ',' || lastChar === '!' || lastChar === '?') {
        const words = inputText.toLowerCase().match(/\b[a-z]{2,}\b/g) || [];
        const currentWordCount = {};
        
        words.forEach(word => {
          currentWordCount[word] = (currentWordCount[word] || 0) + 1;
        });
        
        if (isMounted) {
          setWordFrequency(currentWordCount);
        }
      }
    };
    
    updateWordFrequency();
    
    updatePredictionsAsync();
    updateInsightsAsync();
    updateHeatAsync();
    
    return () => {
      isMounted = false;
    };
  }, [inputText, backendConnected]); 

  useEffect(() => {
    const timer = setInterval(() => {
      setPersonalizedInsights(prev => ({
        ...prev,
        sessionTime: prev.sessionTime + 1
      }));
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  const getKeyStyle = (key) => {
    const prediction = predictions[key] || 0;
    const heat = keyHeat[key] || 0;
    
    return {
      backgroundColor: `rgba(59, 130, 246, ${heat})`,
      borderColor: prediction > 0.1 ? '#3b82f6' : '#e5e7eb',
      borderWidth: prediction > 0.1 ? '2px' : '1px',
      transform: prediction > 0.2 ? 'scale(1.05)' : 'scale(1)',
      boxShadow: prediction > 0.1 ? '0 4px 12px rgba(59, 130, 246, 0.3)' : 'none'
    };
  };

  return (
    <div className="adaptive-keyboard-container">
      <div className="header">
        <h1 className="title">Adaptive Mobile Keyboard</h1>
        <p className="subtitle">AI-Powered Predictive Typing with Real-time Learning</p>
        
        {/* Backend Status Indicator */}
        <div className={`backend-status ${backendConnected ? 'connected' : 'disconnected'}`}>
          <div className="status-indicator">
            {backendConnected ? '🟢' : '🔴'}
            <span className="status-text">
              {backendConnected ? 'Backend Connected (370K+ Words)' : 'Backend Offline (Fallback Mode)'}
            </span>
            {isLoading && <span className="loading-spinner">⏳</span>}
          </div>
          {apiError && (
            <div className="error-message">
              ⚠️ {apiError}
            </div>
          )}
        </div>
      </div>

      <div className="demo-section">
        <div className="input-area">
          <label className="input-label">Type here to see the adaptive predictions:</label>
          <div className="input-container">
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Start typing to see the keyboard adapt to your patterns..."
              className="text-input"
              rows={4}
            />
            {currentSuggestion && (
              <div className="word-suggestion">
                <span className="suggestion-text">{currentSuggestion}</span>
                <span className="suggestion-hint">Press Tab to complete</span>
              </div>
            )}
          </div>
        </div>

        <div className="insights-panel">
          <h3>🧠 Adaptive Intelligence</h3>
          
          <div className="insight-grid">
            <div className="insight-card primary">
              <div className="insight-icon">📚</div>
              <div className="insight-content">
                <span className="insight-value">{personalizedInsights.dictionaryCoverage}%</span>
                <span className="insight-label">Dictionary Coverage</span>
                <span className="insight-explanation">Words found in 370K+ dictionary</span>
              </div>
            </div>
            
            <div className="insight-card">
              <div className="insight-icon">🧠</div>
              <div className="insight-content">
                <span className="insight-value">{personalizedInsights.learnedPatterns}</span>
                <span className="insight-label">Word Transitions</span>
                <span className="insight-explanation">Learned word-to-word patterns</span>
              </div>
            </div>
            
            <div className="insight-card">
              <div className="insight-icon">🌈</div>
              <div className="insight-content">
                <span className="insight-value">{personalizedInsights.vocabularyDiversity}%</span>
                <span className="insight-label">Vocabulary Diversity</span>
                <span className="insight-explanation">Unique words / total words</span>
              </div>
            </div>
            
            <div className="insight-card">
              <div className="insight-icon">⚡</div>
              <div className="insight-content">
                <span className="insight-value">{keystrokesSaved}</span>
                <span className="insight-label">Keystrokes Saved</span>
                <span className="insight-explanation">Via tab word completions</span>
              </div>
            </div>
          </div>
          
          <div className="unique-style-card">
            <h4>✨ Your Typing Style</h4>
            <p className="style-description">{personalizedInsights.uniqueStyle}</p>
          </div>
          
          <div className="personal-dictionary">
            <h4>📚 Your Personal Dictionary</h4>
            <div className="word-insights">
              {Object.entries(wordFrequency)
                .sort(([,a], [,b]) => b - a)
                .slice(0, 6)
                .map(([word, freq]) => (
                  <div key={word} className="word-insight-item">
                    <span className="word-text">{word}</span>
                    <div className="word-stats">
                      <span className="frequency-badge">{freq}x</span>
                      <div className="usage-bar">
                        <div 
                          className="usage-fill" 
                          style={{ width: `${Math.min((freq / Math.max(...Object.values(wordFrequency))) * 100, 100)}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                ))}
            </div>
          </div>
          
          <div className="session-info">
            <span className="session-time">Session: {Math.floor(personalizedInsights.sessionTime / 60)}:{(personalizedInsights.sessionTime % 60).toString().padStart(2, '0')}</span>
          </div>
          
          {/* Debug Panel - Remove this in production */}
          <div className="debug-panel">
            <h4>🔍 Debug Info</h4>
            <div className="debug-stats">
              <div className="debug-item">
                <span className="debug-label">Input Length:</span>
                <span className="debug-value">{inputText.length} chars</span>
              </div>
              <div className="debug-item">
                <span className="debug-label">Word Frequency Updates:</span>
                <span className="debug-value">{Object.keys(wordFrequency).length} unique words</span>
              </div>
              <div className="debug-item">
                <span className="debug-label">Backend Connected:</span>
                <span className="debug-value">{backendConnected ? '✅ Yes' : '❌ No'}</span>
              </div>
              <div className="debug-item">
                <span className="debug-label">Last Character:</span>
                <span className="debug-value">'{inputText[inputText.length - 1] || 'none'}'</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="keyboard-section">
        <div className="keyboard">
          {keyboardRows.map((row, rowIndex) => (
            <div key={rowIndex} className="keyboard-row">
              {row.map(key => (
                <button
                  key={key}
                  className="keyboard-key"
                  style={getKeyStyle(key)}
                  onClick={() => handleKeyPress(key)}
                >
                  <span className="key-char">{key.toUpperCase()}</span>
                  {predictions[key] && (
                    <span className="prediction-percent">
                      {Math.round(predictions[key] * 100)}%
                    </span>
                  )}
                </button>
              ))}
            </div>
          ))}
          <div className="keyboard-row">
            <button
              className="keyboard-key space-key"
              onClick={() => handleKeyPress('space')}
            >
              SPACE
            </button>
            <button
              className="keyboard-key backspace-key"
              onClick={() => handleKeyPress('backspace')}
            >
              ⌫
            </button>
          </div>
        </div>
      </div>

      <div className="features-section">
        <h2 className="features-title">🎯 Key Features</h2>
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">🧠</div>
            <h3>ML Language Model</h3>
            <p>Real n-gram statistical model trained on 611 samples with 505 word vocabulary</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">📊</div>
            <h3>Basic Analytics</h3>
            <p>Dictionary coverage, vocabulary diversity, and word frequency tracking</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">⚡</div>
            <h3>Tab Completion</h3>
            <p>Word autofill with real keystroke savings counter</p>
          </div>
        </div>
      </div>

      <div className="tech-section">
        <h2 className="tech-title">🛠️ Technology Stack</h2>
        <div className="tech-grid">
          <div className="tech-item">Python + Flask</div>
          <div className="tech-item">React + Vite</div>
          <div className="tech-item">N-gram ML Model</div>
        </div>
      </div>
    </div>
  );
};

export default AdaptiveKeyboard;
