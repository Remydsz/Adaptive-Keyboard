"""
Adaptive Keyboard Engine
"""

import json
import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import re
from .word_dictionary import WordDictionary


class AdaptiveKeyboardEngine:
    def __init__(self):
        self.user_word_frequency = {}
        self.word_transitions = {}
        self.predictions = {}
        self.data_file = "keyboard_data.json"
        
        self.word_dict = WordDictionary()
        self.load_data()
    
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
        
        for first_char, next_chars in common_bigrams.items():
            for next_char, count in next_chars.items():
                self.bigram_counts[first_char][next_char] = count
        
        common_frequencies = {
            'e': 120, 't': 90, 'a': 80, 'o': 75, 'i': 70, 'n': 67, 's': 63, 'h': 61,
            'r': 60, 'd': 43, 'l': 40, 'c': 28, 'u': 28, 'm': 24, 'w': 24, 'f': 22,
            'g': 20, 'y': 20, 'p': 19, 'b': 13, 'v': 10, 'k': 8, 'j': 2, 'x': 2,
            'q': 1, 'z': 1, ' ': 200
        }
        
        for char, freq in common_frequencies.items():
            self.key_frequency[char] = freq
        
    def load_data(self):
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.user_word_frequency = data.get('user_word_frequency', {})
                    self.word_transitions = data.get('word_transitions', {})
            except Exception as e:
                print(f"Error loading data: {e}")
    
    def save_data(self):
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
        text = current_text.lower().strip() if current_text else ""
        
        if text:
            self._record_word_usage(text)
        
        self.predictions = self._generate_word_based_predictions(text)
        
        if len(current_text) % 10 == 0:  # Save every 10 characters
            self.save_data()
    
    def _record_word_usage(self, text: str):
        import re
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        for word in words:
            if len(word) >= 2:  # Only record words with 2+ characters
                if word not in self.user_word_frequency:
                    self.user_word_frequency[word] = 0
                self.user_word_frequency[word] += 1
        
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
        if not text:
            return self.word_dict.get_next_char_predictions("")
        
        words = text.split()
        if not words:
            return self.word_dict.get_next_char_predictions("")
        
        current_word = words[-1]
        
        if text.endswith(' '):
            return self.word_dict.get_next_char_predictions("")
        
        word_predictions = self.word_dict.get_next_char_predictions(current_word)
        
        if not word_predictions and current_word and self.word_dict.is_valid_word(current_word):
            return {' ': 0.6, '.': 0.15, ',': 0.10, '!': 0.05, '?': 0.05, ':': 0.05}
        
        if len(text) > 10 and current_word:  # Only after substantial typing
            personalization_boost = self._get_word_personalization_boost(current_word)
            
            if personalization_boost:
                for char, boost in personalization_boost.items():
                    if char in word_predictions:
                        word_predictions[char] = min(word_predictions[char] + boost, 1.0)
        
        return word_predictions
    
    def _get_word_personalization_boost(self, current_word: str) -> Dict[str, float]:
        boost_scores = {}
        
        if not current_word:
            return boost_scores
        
        for user_word, frequency in self.user_word_frequency.items():
            if user_word.startswith(current_word.lower()) and len(user_word) > len(current_word):
                next_char = user_word[len(current_word)].lower()
                
                boost = min(frequency * 0.1, 0.3)
                boost_scores[next_char] = boost_scores.get(next_char, 0) + boost
        
        return boost_scores
    
    def _generate_predictions(self, text: str) -> Dict[str, float]:
        predictions = {}
        
        if len(text) == 0:
            return {'t': 0.15, 'a': 0.12, 'i': 0.10, 'w': 0.08, 's': 0.07}
        
        last_char = text[-1].lower()
        
        last_two = text[-2:].lower() if len(text) >= 2 else ""
        
        if last_char in self.bigram_counts and self.bigram_counts[last_char]:
            total_bigrams = sum(self.bigram_counts[last_char].values())
            if total_bigrams > 0:
                for char, count in self.bigram_counts[last_char].items():
                    predictions[char] = (count / total_bigrams) * 0.8
        
        if last_two in self.trigram_counts and self.trigram_counts[last_two]:
            total_trigrams = sum(self.trigram_counts[last_two].values())
            if total_trigrams > 0:
                for char, count in self.trigram_counts[last_two].items():
                    trigram_prob = (count / total_trigrams) * 0.6
                    predictions[char] = predictions.get(char, 0) + trigram_prob
        
        if predictions:
            max_pred = max(predictions.values())
            if max_pred > 1.0:
                predictions = {k: v / max_pred for k, v in predictions.items()}
        
        predictions = {k: v for k, v in predictions.items() if v > 0.05}
        
        if len(predictions) > 5:
            sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            predictions = dict(sorted_preds[:5])
        
        return predictions
    
    def get_predictions(self) -> Dict[str, float]:
        return self.predictions.copy()
    
    def get_key_heat(self, key: str) -> float:
        if not self.user_word_frequency:
            return 0.0
        
        total_words = sum(self.user_word_frequency.values())
        if total_words < 3:
            return 0.0
        
        key_char = key.lower()
        char_frequency = 0
        
        for word, freq in self.user_word_frequency.items():
            char_count_in_word = word.count(key_char)
            char_frequency += char_count_in_word * freq
        
        max_char_freq = 0
        for char in 'abcdefghijklmnopqrstuvwxyz ':
            char_freq = 0
            for word, freq in self.user_word_frequency.items():
                char_freq += word.count(char) * freq
            max_char_freq = max(max_char_freq, char_freq)
        
        heat = (char_frequency / max_char_freq) * 0.6 if max_char_freq > 0 else 0.0
        return min(heat, 0.6)
    
    def get_top_predictions(self, n: int = 5) -> List[Tuple[str, float]]:
        return sorted(self.predictions.items(), key=lambda x: x[1], reverse=True)[:n]
