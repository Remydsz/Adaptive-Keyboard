"""
Dynamic Learning Engine - Real-time vocabulary expansion and unknown word handling
"""

import json
import time
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Set
from datetime import datetime, timedelta


class DynamicLearningEngine:
    """Engine for dynamic vocabulary expansion and unknown word learning"""
    
    def __init__(self):
        self.unknown_word_frequencies = Counter()
        self.unknown_word_patterns = defaultdict(Counter)  # Context patterns for unknown words
        self.learned_words = set()  # Words we've added to vocabulary
        self.word_first_seen = {}  # When we first encountered each word
        self.word_contexts = defaultdict(list)  # Store contexts where words appear
        
        # Learning thresholds
        self.min_frequency_to_learn = 3  # Learn after seeing word 3+ times
        self.min_context_diversity = 2   # Need to see word in 2+ different contexts
        self.learning_cooldown = 60      # Seconds between learning sessions
        self.last_learning_time = 0
        
        # Dynamic vocabulary file
        self.dynamic_vocab_file = "dynamic_vocabulary.json"
        self.load_dynamic_vocabulary()
    
    def detect_unknown_words(self, text: str, known_vocab: Set[str]) -> List[str]:
        """Detect words that aren't in the current vocabulary"""
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        unknown_words = []
        
        for word in words:
            if len(word) > 1 and word not in known_vocab and word.isalpha():
                unknown_words.append(word)
        
        return unknown_words
    
    def learn_from_text(self, text: str, known_vocab: Set[str]) -> Dict:
        """Learn from user text input - track unknown words and their patterns"""
        unknown_words = self.detect_unknown_words(text, known_vocab)
        learning_stats = {
            'unknown_words_found': len(unknown_words),
            'new_words_learned': 0,
            'words_promoted': []
        }
        
        current_time = time.time()
        words_in_text = text.lower().split()
        
        # Track unknown words and their contexts
        for word in unknown_words:
            self.unknown_word_frequencies[word] += 1
            self.word_contexts[word].append(text[:100])  # Store context snippet
            
            if word not in self.word_first_seen:
                self.word_first_seen[word] = current_time
            
            # Learn contextual patterns for this unknown word
            word_index = None
            for i, w in enumerate(words_in_text):
                if w == word:
                    word_index = i
                    break
            
            if word_index is not None:
                # Learn bigram patterns - with proper Counter handling
                if word_index > 0:
                    prev_word = words_in_text[word_index - 1]
                    pattern_key = f"{prev_word} {word}"
                    self.unknown_word_patterns[pattern_key][pattern_key] = self.unknown_word_patterns[pattern_key].get(pattern_key, 0) + 1
                
                if word_index < len(words_in_text) - 1:
                    next_word = words_in_text[word_index + 1]
                    pattern_key = f"{word} {next_word}"
                    self.unknown_word_patterns[pattern_key][pattern_key] = self.unknown_word_patterns[pattern_key].get(pattern_key, 0) + 1
        
        # Check if we should promote any unknown words to learned vocabulary
        if current_time - self.last_learning_time > self.learning_cooldown:
            promoted_words = self._promote_frequent_words()
            learning_stats['new_words_learned'] = len(promoted_words)
            learning_stats['words_promoted'] = promoted_words
            
            if promoted_words:
                self.last_learning_time = current_time
                self.save_dynamic_vocabulary()
        
        return learning_stats
    
    def _promote_frequent_words(self) -> List[str]:
        """Promote frequently used unknown words to learned vocabulary"""
        promoted = []
        
        for word, frequency in self.unknown_word_frequencies.items():
            if (word not in self.learned_words and 
                frequency >= self.min_frequency_to_learn and
                len(set(self.word_contexts[word])) >= self.min_context_diversity):
                
                # Additional quality checks
                if self._is_valid_word_to_learn(word):
                    self.learned_words.add(word)
                    promoted.append(word)
                    print(f"📚 Learned new word: '{word}' (frequency: {frequency})")
        
        return promoted
    
    def _is_valid_word_to_learn(self, word: str) -> bool:
        """Check if a word is worth learning (quality control)"""
        # Basic quality checks
        if len(word) < 2 or len(word) > 20:
            return False
        
        # Avoid learning obvious typos or gibberish
        if word.count(word[0]) > len(word) * 0.6:  # Too many repeated letters
            return False
        
        # Avoid learning words that are just repeated characters
        if len(set(word)) < 2:
            return False
        
        # Check if it has reasonable letter patterns
        vowels = set('aeiou')
        has_vowel = any(c in vowels for c in word)
        
        return has_vowel or len(word) <= 3  # Short words might be acronyms
    
    def get_unknown_word_predictions(self, context: str, partial_word: str = "") -> List[Tuple[str, float]]:
        """Get predictions for unknown words based on learned patterns"""
        predictions = []
        context_words = context.lower().split()
        
        # Look for learned unknown words that fit the context
        if context_words:
            last_word = context_words[-1]
            
            # Find patterns where this context word precedes our learned words
            for pattern, frequency in self.unknown_word_patterns.items():
                if pattern.startswith(f"{last_word} "):
                    predicted_word = pattern.split()[1]
                    if predicted_word in self.learned_words:
                        if not partial_word or predicted_word.startswith(partial_word):
                            confidence = min(frequency / 10.0, 0.8)  # Scale confidence
                            predictions.append((predicted_word, confidence))
        
        # If we have a partial word, find matching learned words
        if partial_word:
            for learned_word in self.learned_words:
                if learned_word.startswith(partial_word):
                    frequency = self.unknown_word_frequencies.get(learned_word, 1)
                    confidence = min(frequency / 10.0, 0.7)
                    predictions.append((learned_word, confidence))
        
        # Sort by confidence and return top matches
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:5]
    
    def get_learned_words_for_completion(self, partial: str) -> List[Tuple[str, float]]:
        """Get learned words that match a partial input"""
        completions = []
        
        for word in self.learned_words:
            if word.startswith(partial.lower()):
                frequency = self.unknown_word_frequencies[word]
                confidence = min(frequency / 15.0, 0.9)  # Higher confidence for learned words
                completions.append((word, confidence))
        
        return sorted(completions, key=lambda x: x[1], reverse=True)
    
    def get_learning_stats(self) -> Dict:
        """Get statistics about the dynamic learning process"""
        return {
            'learned_words_count': len(self.learned_words),
            'learned_words': list(self.learned_words)[:10],  # Show top 10
            'unknown_words_tracked': len(self.unknown_word_frequencies),
            'top_unknown_words': [
                {'word': word, 'frequency': freq} 
                for word, freq in self.unknown_word_frequencies.most_common(5)
            ],
            'patterns_learned': len(self.unknown_word_patterns),
            'learning_thresholds': {
                'min_frequency': self.min_frequency_to_learn,
                'min_context_diversity': self.min_context_diversity
            }
        }
    
    def force_learn_word(self, word: str, context: str = "") -> bool:
        """Manually add a word to learned vocabulary (for user-requested words)"""
        word = word.lower().strip()
        
        if self._is_valid_word_to_learn(word):
            self.learned_words.add(word)
            self.unknown_word_frequencies[word] += 5  # Boost frequency
            
            if context:
                self.word_contexts[word].append(context)
                words = context.lower().split()
                
                # Add some basic patterns
                word_index = None
                for i, w in enumerate(words):
                    if w == word:
                        word_index = i
                        break
                
                if word_index is not None:
                    if word_index > 0:
                        prev_word = words[word_index - 1]
                        self.unknown_word_patterns[f"{prev_word} {word}"] += 2
                    
                    if word_index < len(words) - 1:
                        next_word = words[word_index + 1]
                        self.unknown_word_patterns[f"{word} {next_word}"] += 2
            
            self.save_dynamic_vocabulary()
            print(f"✅ Manually learned word: '{word}'")
            return True
        
        return False
    
    def save_dynamic_vocabulary(self):
        """Save the learned vocabulary to persistent storage"""
        data = {
            'learned_words': list(self.learned_words),
            'unknown_word_frequencies': dict(self.unknown_word_frequencies),
            'unknown_word_patterns': {k: dict(v) for k, v in self.unknown_word_patterns.items()},
            'word_first_seen': self.word_first_seen,
            'last_updated': time.time()
        }
        
        try:
            with open(self.dynamic_vocab_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving dynamic vocabulary: {e}")
    
    def load_dynamic_vocabulary(self):
        """Load previously learned vocabulary"""
        try:
            with open(self.dynamic_vocab_file, 'r') as f:
                data = json.load(f)
                
            self.learned_words = set(data.get('learned_words', []))
            self.unknown_word_frequencies = Counter(data.get('unknown_word_frequencies', {}))
            
            # Reconstruct patterns
            patterns_data = data.get('unknown_word_patterns', {})
            for pattern, count_dict in patterns_data.items():
                self.unknown_word_patterns[pattern] = Counter(count_dict)
            
            self.word_first_seen = data.get('word_first_seen', {})
            
            print(f"📖 Loaded {len(self.learned_words)} learned words from dynamic vocabulary")
            
        except FileNotFoundError:
            print("📝 No previous dynamic vocabulary found - starting fresh")
        except Exception as e:
            print(f"Error loading dynamic vocabulary: {e}")
    
    def reset_learning(self):
        """Reset all learning data (for testing or cleanup)"""
        self.unknown_word_frequencies.clear()
        self.unknown_word_patterns.clear()
        self.learned_words.clear()
        self.word_first_seen.clear()
        self.word_contexts.clear()
        
        try:
            import os
            if os.path.exists(self.dynamic_vocab_file):
                os.remove(self.dynamic_vocab_file)
        except Exception as e:
            print(f"Error cleaning up dynamic vocabulary file: {e}")
        
        print("🧹 Dynamic learning data reset")
