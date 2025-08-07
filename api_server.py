#!/usr/bin/env python3
"""
Flask API Server for Adaptive Keyboard Backend
Exposes the sophisticated prediction engine to the React frontend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from keyboard_engine import AdaptiveKeyboardEngine
from language_model import EnhancedMLLanguageModel, create_enhanced_training_corpus
import traceback
import sys
import os

# add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from keyboard_engine import AdaptiveKeyboardEngine
from analytics import TypingAnalytics

app = Flask(__name__)
CORS(app)  # cors for react

# globals
keyboard_engine = None
analytics = None
ml_model = None

def initialize_engine():
    """Initialize the keyboard engine and analytics"""
    global keyboard_engine, analytics, ml_model
    if keyboard_engine is None:
        print("Initializing Adaptive Keyboard Engine...")
        keyboard_engine = AdaptiveKeyboardEngine()
        analytics = TypingAnalytics()
        print("Engine initialized successfully!")

        # init ml model
        print("Initializing Enhanced ML Language Model...")
        ml_model = EnhancedMLLanguageModel(max_n=5)  # 5-grams

        # load or train model
        if not ml_model.load_model('enhanced_ml_language_model.json'):
            print("Training new enhanced ML model...")
            corpus = create_enhanced_training_corpus()
            ml_model.train_on_corpus(corpus)
            ml_model.save_model('enhanced_ml_language_model.json')
            print("Enhanced ML model trained and saved!")
        else:
            print("Pre-trained enhanced ML model loaded successfully!")

        print(f"ML Model Stats: {ml_model.get_model_stats()}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Adaptive Keyboard API is running',
        'engine_loaded': keyboard_engine is not None
    })

@app.route('/predict', methods=['POST'])
def get_predictions():
    """Get next character predictions AND word suggestions based on current text"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text parameter'}), 400
        
        text = data['text']
        
        # Update predictions using our sophisticated backend
        keyboard_engine.update_predictions(text)
        predictions = keyboard_engine.get_predictions()
        
        # Get top predictions with probabilities
        top_predictions = keyboard_engine.get_top_predictions(10)
        
        # Format character predictions for frontend
        formatted_predictions = {}
        for char, prob in top_predictions:
            if char.isalpha() or char == ' ':
                formatted_predictions[char] = float(prob)
        
        # ADDED: Get word suggestions using the same logic as analytics endpoint
        word_suggestion = None
        if text:  # Always suggest if there's any text
            words = text.split()
            
            if text.endswith(' ') or text.endswith('\n'):
                # word prediction
                context = text.strip()  
                current_partial = ""  
                
            elif words:  # word completion
                current_partial = words[-1].lower()
                context = ' '.join(words[:-1]) if len(words) > 1 else ""
                
            else:
                # Fallback for edge cases
                context = ""
                current_partial = ""
            
            try:
                ml_predictions = ml_model.predict_next_words(
                    context=context, 
                    partial_word=current_partial, 
                    top_k=5  
                )
                
                if ml_predictions:
                    word_suggestion = ml_predictions[0][0]  
                    print(f"Word suggestion: '{word_suggestion}' (confidence: {ml_predictions[0][1]:.3f})")
                
            except Exception as e:
                print(f"ML model error in predict endpoint: {e}")
                word_suggestion = None
        
        return jsonify({
            'predictions': formatted_predictions,
            'total_predictions': len(predictions),
            'text_length': len(text),
            'word_suggestion': word_suggestion 
        })
    
    except Exception as e:
        print(f"Error in get_predictions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/insights', methods=['POST'])
def get_personalized_insights():
    """Get personalized adaptive intelligence insights"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text parameter'}), 400
        
        text = data['text']
        
        if text:
            keyboard_engine.update_predictions(text)
        
        words = text.lower().split() if text else []
        unique_words = set(words)
        total_words = len(words)
        
        print(f"Calculating insights for text: '{text[:50]}...'")
        print(f"Total words: {total_words}, Unique words: {len(unique_words)}")
        
        word_freq = keyboard_engine.word_frequency if hasattr(keyboard_engine, 'word_frequency') else {}
        word_transitions = keyboard_engine.word_transitions if hasattr(keyboard_engine, 'word_transitions') else {}
        
        learned_patterns = len(word_transitions)
        print(f"Learned word transitions: {learned_patterns}")
        
        words_in_dict = 0
        if unique_words:
            common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs', 'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'yourselves', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now', 'hello', 'world', 'test', 'example', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog'}
            words_in_dict = len([w for w in unique_words if w in common_words or len(w) > 2])
        
        dictionary_coverage = int((words_in_dict / max(len(unique_words), 1)) * 100) if unique_words else 0
        print(f"Dictionary coverage: {words_in_dict}/{len(unique_words)} words = {dictionary_coverage}%")
        
        learned_patterns = len(word_transitions)
        pattern_examples = list(word_transitions.keys())[:3] if word_transitions else []
        print(f"Learned word patterns: {learned_patterns} (e.g., {pattern_examples})")
        
        vocab_diversity = round((len(unique_words) / max(total_words, 1)) * 100, 1) if total_words > 0 else 0
        print(f"Vocabulary diversity: {len(unique_words)}/{total_words} = {vocab_diversity}%")
        
        word_suggestion = None
        if text:  
            words = text.split()
            
            if text.endswith(' ') or text.endswith('\n'):
                context = text.strip()  
                current_partial = ""  
                print(f"Next-word prediction after space. Context: '{context}'")
                
            elif words:  
                current_partial = words[-1].lower()
                context = ' '.join(words[:-1]) if len(words) > 1 else ""
                print(f"Word completion for partial: '{current_partial}', Context: '{context}'")
                
            else:
                context = ""
                current_partial = ""
            
            if True: 
                
                user_word_candidates = []
                if word_freq and current_partial: 
                    user_word_candidates = [
                        word for word in word_freq.keys() 
                        if word.startswith(current_partial) and word != current_partial
                    ]
                    
                    user_word_candidates.sort(key=lambda w: word_freq[w], reverse=True)
                
                try:
                    ml_predictions = ml_model.predict_next_words(
                        context=context, 
                        partial_word=current_partial, 
                        top_k=10
                    )
                    
                    dict_candidates = []
                    for word, probability in ml_predictions:
                        ml_score = probability * 1000
                        dict_candidates.append((word, ml_score))
                    
                    dict_candidates.sort(key=lambda x: x[1], reverse=True)
                    dict_candidates = [word for word, score in dict_candidates]
                    
                    print(f"ML predictions for context '{context}' + partial '{current_partial}':")
                    for i, (word, prob) in enumerate(ml_predictions[:3]):
                        print(f"   {i+1}. {word} (ML probability: {prob:.4f})")
                    
                except Exception as e:
                    print(f"ML model error: {e}, falling back to dictionary search")
                    if not hasattr(app, 'full_dictionary'):
                        print("Loading dictionary for fallback...")
                        with open('words.txt', 'r', encoding='utf-8') as f:
                            app.full_dictionary = set(line.strip().lower() for line in f if line.strip())
                    
                    dict_candidates = [
                        word for word in app.full_dictionary 
                        if word.startswith(current_partial) and word != current_partial and len(word) <= 15
                    ][:10]
                    
                    all_candidates = user_word_candidates + dict_candidates
                    if all_candidates:
                        word_suggestion = all_candidates[0]  
                        print(f"ML suggestion for '{current_partial}': '{word_suggestion}' (from {len(all_candidates)} candidates)")
        
        if len(unique_words) == 0:
            unique_style = "Ready to learn your patterns..."
        elif len(unique_words) < 3:
            unique_style = "Building initial patterns"
        elif len(unique_words) < 8:
            unique_style = "Developing personalized predictions"
        elif avg_word_length > 6:
            unique_style = "Sophisticated vocabulary detected"
        elif len([w for w in words if len(w) <= 3]) > len(words) * 0.6:
            unique_style = "Concise, efficient typing style"
        else:
            unique_style = "Balanced, adaptive writing style"
        
        top_words = []
        if word_freq:
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            top_words = [{'word': word, 'frequency': freq} for word, freq in sorted_words]
        
        insights = {
            'learned_patterns': learned_patterns,
            'dictionary_coverage': dictionary_coverage,
            'vocabulary_diversity': vocab_diversity,
            'unique_style': unique_style,
            'top_words': top_words,
            'total_unique_words': len(unique_words),
            'pattern_examples': pattern_examples,
            'word_suggestion': word_suggestion
        }
        
        return jsonify(insights)
    
    except Exception as e:
        import traceback
        error_details = {
            'error': str(e),
            'type': type(e).__name__,
            'traceback': traceback.format_exc()
        }
        print(f"Error in get_personalized_insights: {e}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        return jsonify(error_details), 500

@app.route('/heat', methods=['POST'])
def get_key_heat():
    """Get heat values for all keys based on usage patterns"""
    try:
        data = request.get_json()
        text = data.get('text', '') if data else ''
        
        if text:
            keyboard_engine.update_predictions(text)
        
        keys = 'abcdefghijklmnopqrstuvwxyz '
        heat_map = {}
        
        for key in keys:
            heat_map[key] = keyboard_engine.get_key_heat(key)
        
        return jsonify({
            'heat_map': heat_map,
            'max_heat': max(heat_map.values()) if heat_map.values() else 0
        })
    
    except Exception as e:
        print(f"Error in get_key_heat: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/word-frequency', methods=['GET'])
def get_word_frequency():
    """Get user's word frequency data for personalization insights"""
    try:
        word_freq = keyboard_engine.user_word_frequency
        
        # Get top 20 most frequent words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        
        return jsonify({
            'word_frequency': dict(sorted_words),
            'total_unique_words': len(word_freq),
            'total_word_instances': sum(word_freq.values())
        })
    
    except Exception as e:
        print(f"Error in get_word_frequency: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset_engine():
    """Reset the keyboard engine and analytics"""
    try:
        global keyboard_engine, analytics
        keyboard_engine = AdaptiveKeyboardEngine()
        analytics.update_typing_data(text, wpm, accuracy)
        
        # Learn from user input for personalization
        ml_model.learn_from_user_input(text)
        
        # Get analytics data
        analytics_data = analytics.get_analytics()
        return jsonify({
            'message': 'Engine and analytics reset successfully',
            'status': 'success',
            'analytics': analytics_data
        })
    
    except Exception as e:
        print(f"Error in reset_engine: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/dictionary-info', methods=['GET'])
def get_dictionary_info():
    """Get information about the loaded dictionary"""
    try:
        word_dict = keyboard_engine.word_dict
        
        return jsonify({
            'total_words': len(word_dict.words),
            'dictionary_loaded': len(word_dict.words) > 0,
            'sample_words': list(word_dict.words)[:10] if word_dict.words else [],
            'word_frequencies_sample': dict(list(word_dict.word_frequencies.items())[:10])
        })
    
    except Exception as e:
        print(f"Error in get_dictionary_info: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting server...")
    
    # Initialize the engine
    initialize_engine()
    
    print("Server ready!")
    print("API endpoints available:")
    print("   POST /predict - Get character predictions")
    print("   POST /analytics - Update and get typing analytics")
    print("   POST /heat - Get key heat map")
    print("   GET /word-frequency - Get word frequency data")
    print("   GET /dictionary-info - Get dictionary information")
    print("   POST /reset - Reset engine state")
    print("   GET /health - Health check")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=3000,
        debug=True,
        threaded=True
    )
