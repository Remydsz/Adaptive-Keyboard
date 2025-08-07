"""
Adaptive Keyboard Engine - Core logic for prediction and adaptation
"""

import json
import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import re
from word_dictionary import WordDictionary


class AdaptiveKeyboardEngine:
    def __init__(self):
        # Word-based personalization (replacing character-based n-grams)
        self.user_word_frequency = {}  # Track words user types most often
        self.word_transitions = {}     # Track which words typically follow others
        self.predictions = {}
        self.data_file = "keyboard_data.json"
        
        # Initialize word dictionary for modern predictive text
        self.word_dict = WordDictionary()
        
        # Initialize with iOS/Android-like baseline predictions
        self._initialize_mobile_baseline()
        self.load_data()
    
    def _initialize_mobile_baseline(self):
        """Initialize with word-based baseline - no longer needed with comprehensive dictionary"""
        # The word dictionary now provides the mobile baseline behavior
        # No initialization needed since we use the comprehensive dictionary
        # with proper frequency weighting for mobile-like predictions
        pass
    
    def _initialize_baseline_patterns(self):
        """Initialize with common English language patterns for better predictions"""
        # Common English bigrams with realistic frequencies
        common_bigrams = {
            't': {'h': 50, 'e': 30, 'i': 20, 'o': 15, 'r': 10},
            'h': {'e': 40, 'a': 25, 'i': 20, 'o': 15, 'r': 10},
            'e': {'r': 35, 's': 30, 'd': 25, 'n': 20, 't': 15},
            'a': {'n': 40, 'r': 30, 't': 25, 's': 20, 'l': 15},
            'i': {'n': 45, 't': 30, 's': 25, 'o': 20, 'e': 15},
            'o': {'n': 40, 'r': 30, 'u': 25, 'f': 20, 't': 15},
            'n': {'t': 40, 'g': 30, 'd': 25, 'e': 20, 's': 15},
            'r': {'e': 35, 'i': 30, 'o': 25, 's': 20, 't': 15},
            's': {'t': 40, 'e': 30, 'h': 25, 'i': 20, 'o': 15},
            'l': {'e': 40, 'l': 30, 'i': 25, 'o': 20, 'y': 15},
            'c': {'h': 40, 'e': 30, 'o': 25, 'a': 20, 'k': 15},
            'u': {'r': 35, 's': 30, 't': 25, 'n': 20, 'l': 15},
            'd': {'e': 40, 'i': 30, 'a': 25, 'o': 20, 's': 15},
            'p': {'e': 35, 'r': 30, 'o': 25, 'a': 20, 'l': 15},
            'm': {'e': 40, 'a': 30, 'i': 25, 'o': 20, 'y': 15},
            'f': {'o': 40, 'e': 30, 'i': 25, 'r': 20, 'a': 15},
            'g': {'h': 35, 'e': 30, 'o': 25, 'r': 20, 'a': 15},
            'w': {'h': 40, 'e': 30, 'i': 25, 'o': 20, 'a': 15},
            'y': {'e': 35, 'o': 30, 'a': 25, 's': 20, 'i': 15},
            'b': {'e': 40, 'a': 30, 'o': 25, 'r': 20, 'l': 15},
            'v': {'e': 40, 'i': 30, 'a': 25, 'o': 20, 'r': 15},
            'k': {'e': 35, 'i': 30, 'a': 25, 'o': 20, 's': 15},
            ' ': {'t': 50, 'a': 40, 'i': 35, 'o': 30, 'w': 25, 's': 20, 'h': 15, 'b': 10}
        }
        
        # Initialize bigram counts with baseline patterns
        for first_char, next_chars in common_bigrams.items():
            for next_char, count in next_chars.items():
                self.bigram_counts[first_char][next_char] = count
        
        # Common English letter frequencies
        common_frequencies = {
            'e': 120, 't': 90, 'a': 80, 'o': 75, 'i': 70, 'n': 67, 's': 63, 'h': 61,
            'r': 60, 'd': 43, 'l': 40, 'c': 28, 'u': 28, 'm': 24, 'w': 24, 'f': 22,
            'g': 20, 'y': 20, 'p': 19, 'b': 13, 'v': 10, 'k': 8, 'j': 2, 'x': 2,
            'q': 1, 'z': 1, ' ': 200
        }
        
        # Initialize key frequency with baseline
        for char, freq in common_frequencies.items():
            self.key_frequency[char] = freq
        
    def load_data(self):
        """Load existing word-based personalization data"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.user_word_frequency = data.get('user_word_frequency', {})
                    self.word_transitions = data.get('word_transitions', {})
            except Exception as e:
                print(f"Error loading data: {e}")
    
    def save_data(self):
        """Save word-based personalization data to file"""
        try:
            data = {
                'user_word_frequency': dict(self.user_word_frequency),
                'word_transitions': dict(self.word_transitions)
            }
            with open(self.data_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def update_predictions(self, current_text: str):
        """Update predictions based on current text input"""
        # Clean and normalize text
        text = current_text.lower().strip() if current_text else ""
        
        # Record word usage patterns for personalization
        if text:
            self._record_word_usage(text)
        
        # Generate word-based predictions
        self.predictions = self._generate_word_based_predictions(text)
        
        # Save data periodically
        if len(current_text) % 10 == 0:  # Save every 10 characters
            self.save_data()
    
    def _record_word_usage(self, text: str):
        """Record word usage patterns for personalization"""
        # Extract completed words from text
        import re
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Record word frequency (only meaningful words, not single letters)
        for word in words:
            if len(word) >= 2:  # Only record words with 2+ characters
                if word not in self.user_word_frequency:
                    self.user_word_frequency[word] = 0
                self.user_word_frequency[word] += 1
        
        # Record word transitions (what word typically follows another)
        for i in range(len(words) - 1):
            if len(words[i]) >= 2 and len(words[i+1]) >= 2:  # Both words must be meaningful
                current_word = words[i]
                next_word = words[i+1]
                
                if current_word not in self.word_transitions:
                    self.word_transitions[current_word] = {}
                if next_word not in self.word_transitions[current_word]:
                    self.word_transitions[current_word][next_word] = 0
                self.word_transitions[current_word][next_word] += 1
    
    def _generate_word_based_predictions(self, text: str) -> Dict[str, float]:
        """Generate predictions using word dictionary like modern keyboards"""
        if not text:
            # Show common starting characters when no text
            return self.word_dict.get_next_char_predictions("")
        
        # Find the current word being typed
        words = text.split()
        if not words:
            return self.word_dict.get_next_char_predictions("")
        
        current_word = words[-1]
        
        # If the last character is a space, we're starting a new word
        if text.endswith(' '):
            # Predict common starting characters for new words
            return self.word_dict.get_next_char_predictions("")
        
        # Get word-based predictions for the current partial word
        word_predictions = self.word_dict.get_next_char_predictions(current_word)
        
        # If no character predictions (word is complete with no extensions), suggest space/punctuation
        if not word_predictions and current_word and self.word_dict.is_valid_word(current_word):
            # Suggest space and common punctuation when word is complete
            return {' ': 0.6, '.': 0.15, ',': 0.10, '!': 0.05, '?': 0.05, ':': 0.05}
        
        # Apply word-based personalization boost (much more moderate than old system)
        if len(text) > 10 and current_word:  # Only after substantial typing
            personalization_boost = self._get_word_personalization_boost(current_word)
            
            # Apply moderate personalization boost to dictionary predictions
            if personalization_boost:
                for char, boost in personalization_boost.items():
                    if char in word_predictions:
                        # Boost existing predictions moderately (max 30% increase)
                        word_predictions[char] = min(word_predictions[char] + boost, 1.0)
        
        return word_predictions
    
    def _get_word_personalization_boost(self, current_word: str) -> Dict[str, float]:
        """Get personalization boost based on user's word usage patterns"""
        boost_scores = {}
        
        if not current_word:
            return boost_scores
        
        # Find words user types frequently that start with current_word
        for user_word, frequency in self.user_word_frequency.items():
            if user_word.startswith(current_word.lower()) and len(user_word) > len(current_word):
                # Get next character from this frequently used word
                next_char = user_word[len(current_word)].lower()
                
                # Boost score based on how often user types this word
                # Use moderate boost (not aggressive like old system)
                boost = min(frequency * 0.1, 0.3)  # Cap boost at 30%
                boost_scores[next_char] = boost_scores.get(next_char, 0) + boost
        
        return boost_scores
    
    def _generate_predictions(self, text: str) -> Dict[str, float]:
        """Generate probability predictions for next characters"""
        predictions = {}
        
        # Like iOS/Android, show predictions immediately from first character
        if len(text) == 0:
            # Show common starting letters like mobile keyboards do
            return {'t': 0.15, 'a': 0.12, 'i': 0.10, 'w': 0.08, 's': 0.07}
        
        # Get last character for bigram prediction
        last_char = text[-1].lower()
        
        # Get last two characters for trigram prediction
        last_two = text[-2:].lower() if len(text) >= 2 else ""
        
        # Start with bigram predictions (primary source)
        if last_char in self.bigram_counts and self.bigram_counts[last_char]:
            total_bigrams = sum(self.bigram_counts[last_char].values())
            if total_bigrams > 0:
                for char, count in self.bigram_counts[last_char].items():
                    predictions[char] = (count / total_bigrams) * 0.8
        
        # Add trigram predictions if available (higher precision)
        if last_two in self.trigram_counts and self.trigram_counts[last_two]:
            total_trigrams = sum(self.trigram_counts[last_two].values())
            if total_trigrams > 0:
                for char, count in self.trigram_counts[last_two].items():
                    trigram_prob = (count / total_trigrams) * 0.6
                    predictions[char] = predictions.get(char, 0) + trigram_prob
        
        # Normalize predictions to prevent values > 1.0
        if predictions:
            max_pred = max(predictions.values())
            if max_pred > 1.0:
                predictions = {k: v / max_pred for k, v in predictions.items()}
        
        # Filter out very low probability predictions to reduce noise
        predictions = {k: v for k, v in predictions.items() if v > 0.05}
        
        # Limit to top 5 predictions to avoid overwhelming the UI
        if len(predictions) > 5:
            sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            predictions = dict(sorted_preds[:5])
        
        return predictions
    
    def get_predictions(self) -> Dict[str, float]:
        """Get current predictions"""
        return self.predictions.copy()
    
    def get_key_heat(self, key: str) -> float:
        """Get heat value (0-1) for a specific key based on word usage patterns"""
        if not self.user_word_frequency:
            return 0.0  # Completely neutral initially
        
        # Only show heat if user has typed at least 3 words
        total_words = sum(self.user_word_frequency.values())
        if total_words < 3:
            return 0.0  # Stay neutral until enough typing
        
        # Calculate heat based on how often this key appears in user's frequently typed words
        key_char = key.lower()
        char_frequency = 0
        
        # Count occurrences of this character in user's frequent words
        for word, freq in self.user_word_frequency.items():
            char_count_in_word = word.count(key_char)
            char_frequency += char_count_in_word * freq
        
        # Find max character frequency for normalization
        max_char_freq = 0
        for char in 'abcdefghijklmnopqrstuvwxyz ':
            char_freq = 0
            for word, freq in self.user_word_frequency.items():
                char_freq += word.count(char) * freq
            max_char_freq = max(max_char_freq, char_freq)
        
        # Normalize to 0.0-0.6 range (more subtle heat visualization)
        heat = (char_frequency / max_char_freq) * 0.6 if max_char_freq > 0 else 0.0
        return min(heat, 0.6)
    
    def get_top_predictions(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top N predictions sorted by probability"""
        return sorted(self.predictions.items(), key=lambda x: x[1], reverse=True)[:n]
