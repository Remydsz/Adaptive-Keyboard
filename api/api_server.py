#!/usr/bin/env python3
"""
Adaptive Keyboard API Server
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import sys
import os
import time
from functools import lru_cache
import hashlib

# Add parent directory to sys.path so we can import from engines and models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines.keyboard_engine import AdaptiveKeyboardEngine
from models.optimized_language_model import OptimizedLanguageModel
from engines.smart_context_engine import SmartContextEngine
from engines.predictive_completion_engine import PredictiveCompletionEngine
from analytics import TypingAnalytics

app = Flask(__name__)
CORS(app)  # cors for react

keyboard_engine = None
analytics = None
ml_model = None
smart_context = None
predictive_engine = None

prediction_cache = {}
CACHE_MAX_SIZE = 1000
CACHE_EXPIRY_SECONDS = 300

def initialize_engine():
    """Initialize the keyboard engine and analytics"""
    global keyboard_engine, analytics, ml_model, smart_context, predictive_engine
    if keyboard_engine is None:
        print("Initializing Adaptive Keyboard Engine...")
        ml_model = OptimizedLanguageModel(max_n=3)
        keyboard_engine = AdaptiveKeyboardEngine()
        smart_context = SmartContextEngine()
        predictive_engine = PredictiveCompletionEngine()
        analytics = TypingAnalytics()
        print("Engine initialized successfully!")

        print("Initializing Optimized Language Model...")
        if not ml_model.load_model('optimized_ml_model.json'):
            print("Training new optimized ML model...")
            from models.optimized_language_model import create_optimized_training_corpus
            corpus = create_optimized_training_corpus()
            ml_model.train_on_corpus(corpus)
            ml_model.save_model('optimized_ml_model.json')
            print("Optimized ML model trained and saved!")
        else:
            print("Pre-trained optimized ML model loaded successfully!")

        print(f"ML Model Stats: Vocab={len(ml_model.vocab)}, Total Words={ml_model.total_words}")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'engine_loaded': keyboard_engine is not None
    })

def _generate_cache_key(text: str) -> str:
    """Generate a cache key for the given text"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def _clean_cache():
    """Clean expired cache entries"""
    global prediction_cache
    current_time = time.time()
    expired_keys = []
    
    for key, (result, timestamp) in prediction_cache.items():
        if current_time - timestamp > CACHE_EXPIRY_SECONDS:
            expired_keys.append(key)
    
    for key in expired_keys:
        del prediction_cache[key]
    
    # Also limit cache size
    if len(prediction_cache) > CACHE_MAX_SIZE:
        # Remove oldest entries
        sorted_cache = sorted(prediction_cache.items(), key=lambda x: x[1][1])
        items_to_remove = len(prediction_cache) - CACHE_MAX_SIZE + 100  # Remove extra
        for i in range(items_to_remove):
            del prediction_cache[sorted_cache[i][0]]

@app.route('/predict', methods=['POST'])
def get_predictions():
    """Optimized prediction endpoint with caching and reduced model redundancy"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text parameter'}), 400
        
        text = data['text']
        
        cache_key = _generate_cache_key(text)
        current_time = time.time()
        
        if cache_key in prediction_cache:
            result, timestamp = prediction_cache[cache_key]
            if current_time - timestamp <= CACHE_EXPIRY_SECONDS:
                return jsonify(result)
        
        if len(prediction_cache) % 50 == 0:
            _clean_cache()
        
        word_suggestion = None
        character_predictions = {}
        
        if text:
            words = text.split()
            
            if text.endswith(' ') or text.endswith('\n'):
                context = text.strip()
                current_partial = ""
            elif words:
                current_partial = words[-1].lower()
                context = ' '.join(words[:-1]) if len(words) > 1 else ""
            else:
                context = ""
                current_partial = ""
            
            try:
                ml_predictions = ml_model.predict_next_words(
                    context=context,
                    partial_word=current_partial,
                    top_k=3
                )
                
                if ml_predictions:
                    word_suggestion = ml_predictions[0][0]
                
                if current_partial:
                    for word, confidence in ml_predictions[:3]:
                        if word.lower().startswith(current_partial) and len(word) > len(current_partial):
                            next_char = word[len(current_partial)].lower()
                            if next_char.isalpha():
                                character_predictions[next_char] = character_predictions.get(next_char, 0) + confidence
                else:
                    common_starts = {'t': 0.15, 'a': 0.12, 'i': 0.10, 'w': 0.08, 's': 0.07, 'h': 0.06, 'o': 0.05}
                    character_predictions.update(common_starts)
            
            except Exception as e:
                print(f"ML model error in predict endpoint: {e}")
                character_predictions = {'t': 0.15, 'a': 0.12, 'i': 0.10, 'w': 0.08, 's': 0.07}
        else:
            character_predictions = {'t': 0.15, 'a': 0.12, 'i': 0.10, 'w': 0.08, 's': 0.07}
        
        if character_predictions:
            total = sum(character_predictions.values())
            if total > 0:
                character_predictions = {k: v/total for k, v in character_predictions.items()}
        
        result = {
            'predictions': character_predictions,
            'total_predictions': len(character_predictions),
            'text_length': len(text),
            'word_suggestion': word_suggestion,
            'cached': False
        }
        
        prediction_cache[cache_key] = (result.copy(), current_time)
        result['cached'] = False  # This response is not from cache
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in get_predictions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/insights', methods=['POST'])
def get_personalized_insights():
    """Get adaptive keyboard personalization insights"""
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
        
        # Get user patterns
        word_freq = keyboard_engine.word_frequency if hasattr(keyboard_engine, 'word_frequency') else {}
        word_transitions = keyboard_engine.word_transitions if hasattr(keyboard_engine, 'word_transitions') else {}
        learned_patterns = len(word_transitions)
        
        word_suggestion = None
        if text:
            try:
                if text.endswith(' '):
                    suggestions = ml_model.predict_next_words(context=text.strip(), top_k=1)
                elif text.split():
                    words = text.split()
                    partial = words[-1]
                    suggestions = ml_model.predict_next_words(partial_word=partial, top_k=1)
                else:
                    suggestions = []
                
                if suggestions:
                    word_suggestion = suggestions[0][0]
            except Exception:
                word_suggestion = None
        
        if learned_patterns == 0:
            adaptation_status = "Learning your patterns"
        elif learned_patterns < 10:
            adaptation_status = "Building predictions"
        else:
            adaptation_status = "Fully adapted"
        
        # Top user words for personalization display
        top_words = []
        if word_freq:
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:3]
            top_words = [{'word': word, 'count': freq} for word, freq in sorted_words]
        
        insights = {
            'learned_patterns': learned_patterns,
            'adaptation_status': adaptation_status,
            'top_words': top_words,
            'total_unique_words': len(unique_words),
            'total_words': total_words,
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

@app.route('/smart-completion', methods=['POST'])
def get_smart_completion():
    """Get intelligent sentence completion suggestions"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'completions': []})
        
        # Use predictive completion engine for sentence-level intelligence
        completions = predictive_engine.predict_sentence_completion(text)
        
        return jsonify({
            'completions': [{'text': comp, 'confidence': conf} for comp, conf in completions],
            'original_text': text
        })
        
    except Exception as e:
        print(f"Smart completion error: {e}")
        return jsonify({'error': 'Smart completion failed', 'completions': []}), 500

@app.route('/context-analysis', methods=['POST'])
def analyze_context():
    """Analyze text context and provide intelligent insights"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'domain': 'general', 'suggestions': []})
        
        # Detect domain and get smart suggestions
        domain = smart_context.detect_domain(text)
        suggestions = smart_context.get_intelligent_suggestions(text, num_suggestions=5)
        
        return jsonify({
            'domain': domain,
            'suggestions': suggestions,
            'text_length': len(text),
            'word_count': len(text.split())
        })
        
    except Exception as e:
        print(f"Context analysis error: {e}")
        return jsonify({'error': 'Context analysis failed', 'domain': 'general'}), 500

@app.route('/smart-autocorrect', methods=['POST'])
def smart_autocorrect():
    """Get intelligent autocorrect suggestions for potentially misspelled words"""
    try:
        data = request.get_json()
        word = data.get('word', '')
        context = data.get('context', '')
        
        if not word:
            return jsonify({'corrections': []})
        
        # Get intelligent autocorrect suggestions
        corrections = predictive_engine.get_intelligent_autocorrect(word)
        
        # If we have context, enhance with smart predictions
        if context and corrections:
            enhanced_corrections = smart_context.get_smart_predictions(
                context, word, corrections
            )
            corrections = enhanced_corrections
        
        return jsonify({
            'corrections': [{'word': corr, 'confidence': conf} for corr, conf in corrections],
            'original_word': word
        })
        
    except Exception as e:
        print(f"Smart autocorrect error: {e}")
        return jsonify({'error': 'Smart autocorrect failed', 'corrections': []}), 500

@app.route('/next-word-intelligence', methods=['POST'])
def get_next_word_intelligence():
    """Get highly intelligent next word suggestions based on advanced context analysis"""
    try:
        data = request.get_json()
        context = data.get('text', '')
        
        if not context:
            return jsonify({'suggestions': []})
        
        # Get intelligent next word suggestions
        suggestions = predictive_engine.get_next_word_suggestions(context, num_suggestions=8)
        
        # Apply smart context enhancement
        if suggestions:
            enhanced_suggestions = smart_context.get_smart_predictions(
                context, '', suggestions
            )
            suggestions = enhanced_suggestions
        
        return jsonify({
            'suggestions': [{'word': word, 'confidence': conf} for word, conf in suggestions],
            'context_domain': smart_context.detect_domain(context)
        })
        
    except Exception as e:
        print(f"Next word intelligence error: {e}")
        return jsonify({'error': 'Next word intelligence failed', 'suggestions': []}), 500

@app.route('/learn-feedback', methods=['POST'])
def learn_from_feedback():
    """Learn from user feedback to improve future predictions"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        completion = data.get('completion', '')
        was_accepted = data.get('accepted', False)
        typing_speed = data.get('typing_speed', 0)
        
        if text and completion:
            # Let the predictive engine learn from the completion choice
            predictive_engine.learn_from_completion(text, completion, was_accepted)
            
            # Let the smart context engine learn from the interaction
            smart_context.learn_from_user(text, typing_speed)
        
        return jsonify({
            'status': 'learned',
            'message': 'Feedback processed successfully'
        })
        
    except Exception as e:
        print(f"Learning feedback error: {e}")
        return jsonify({'error': 'Learning feedback failed'}), 500

@app.route('/learn-word', methods=['POST'])
def learn_word():
    """Manually learn a specific word and add it to dynamic vocabulary"""
    try:
        data = request.get_json()
        word = data.get('word', '').strip()
        context = data.get('context', '')
        
        if not word:
            return jsonify({'error': 'No word provided'}), 400
        
        # Use the ML model's dynamic learning engine to learn the word
        success = ml_model.dynamic_learning.force_learn_word(word, context)
        
        if success:
            # Update the main vocabulary too
            ml_model.vocab.add(word.lower())
            ml_model.word_frequencies[word.lower()] = ml_model.dynamic_learning.unknown_word_frequencies[word.lower()]
            
            return jsonify({
                'status': 'success',
                'message': f"Successfully learned word: '{word}'",
                'word': word
            })
        else:
            return jsonify({
                'status': 'failed',
                'message': f"Could not learn word: '{word}' (may not meet quality criteria)"
            }), 400
            
    except Exception as e:
        print(f"Learn word error: {e}")
        return jsonify({'error': 'Failed to learn word'}), 500

@app.route('/dynamic-vocab-stats', methods=['GET'])
def get_dynamic_vocab_stats():
    """Get statistics about dynamically learned vocabulary"""
    try:
        # Get learning stats from the ML model
        stats = ml_model.dynamic_learning.get_learning_stats()
        
        return jsonify({
            'dynamic_learning_stats': stats,
            'total_vocab_size': len(ml_model.vocab),
            'original_vocab_size': len(ml_model.vocab) - len(stats['learned_words']),
            'expansion_percentage': len(stats['learned_words']) / max(len(ml_model.vocab), 1) * 100
        })
        
    except Exception as e:
        print(f"Dynamic vocab stats error: {e}")
        return jsonify({'error': 'Failed to get dynamic vocabulary stats'}), 500

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
    print("   POST /smart-completion - Get intelligent sentence completions")
    print("   POST /context-analysis - Analyze text context and domain")
    print("   POST /smart-autocorrect - Get intelligent autocorrect suggestions")
    print("   POST /next-word-intelligence - Get smart next word predictions")
    print("   POST /learn-feedback - Provide learning feedback")
    print("   POST /learn-word - Manually learn a specific word")
    print("   GET /dynamic-vocab-stats - Get dynamic vocabulary statistics")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=3000,
        debug=True,
        threaded=True
    )
