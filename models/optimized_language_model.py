"""
Optimized Language Model
"""

import json
import re
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional
import time
from datetime import datetime
from engines.smart_context_engine import SmartContextEngine
from engines.predictive_completion_engine import PredictiveCompletionEngine
from engines.dynamic_learning_engine import DynamicLearningEngine

class OptimizedLanguageModel:
    """Fast language model using 2-3 gram approach"""
    
    def __init__(self, max_n=3):  # Reduced from 5 to 3 for speed
        self.max_n = min(max_n, 3)
        self.bigram_counts = defaultdict(Counter)
        self.trigram_counts = defaultdict(Counter)
        self.word_frequencies = Counter()
        self.vocab = set()
        self.total_words = 0
        

        self.common_starters = {
            'the', 'a', 'an', 'i', 'you', 'we', 'they', 'it', 'this', 'that',
            'is', 'are', 'was', 'were', 'have', 'has', 'will', 'would', 'could',
            'can', 'may', 'might', 'should', 'must', 'do', 'does', 'did'
        }
        
        # Initialize advanced intelligence engines
        self.smart_context = SmartContextEngine()
        self.predictive_completion = PredictiveCompletionEngine()
        self.dynamic_learning = DynamicLearningEngine()
        
        # Robust fallback word suggestions for common contexts
        self.fallback_suggestions = {
            'please': ['help', 'let', 'provide', 'send', 'give', 'review', 'confirm'],
            'could': ['you', 'we', 'this', 'be', 'have', 'please'],
            'would': ['you', 'like', 'be', 'have', 'appreciate'],
            'should': ['we', 'you', 'be', 'have', 'consider'],
            'will': ['be', 'have', 'you', 'we', 'need'],
            'can': ['you', 'we', 'be', 'help', 'provide'],
            'and': ['the', 'we', 'I', 'you', 'it', 'then'],
            'or': ['you', 'we', 'the', 'maybe', 'perhaps'],
            'to': ['be', 'do', 'help', 'provide', 'discuss', 'review'],
            'for': ['the', 'your', 'this', 'our', 'me'],
            'with': ['the', 'you', 'this', 'our', 'me'],
            'on': ['the', 'this', 'our', 'your'],
            'in': ['the', 'this', 'our', 'order'],
            'at': ['the', 'this', 'your', 'our'],
            'by': ['the', 'you', 'me', 'us'],
            'me': ['know', 'help', 'understand'],
            'us': ['know', 'help', 'discuss'],
            'know': ['if', 'what', 'how', 'when', 'where'],
            'what': ['you', 'we', 'the', 'time'],
            'how': ['to', 'you', 'we', 'much'],
            'when': ['you', 'we', 'the', 'will'],
            'where': ['you', 'we', 'the', 'is'],
            'why': ['you', 'we', 'this', 'not']
        }
        
        # Common word completions for fast suggestions
        self.common_completions = {
            't': ['the', 'that', 'this', 'to', 'time', 'take', 'think', 'through'],
            'a': ['and', 'a', 'are', 'all', 'an', 'as', 'at', 'about'],
            'w': ['will', 'with', 'was', 'we', 'would', 'when', 'where', 'what'],
            'h': ['have', 'has', 'he', 'her', 'his', 'how', 'here', 'help'],
            'i': ['is', 'it', 'in', 'i', 'if', 'into', 'its', 'include'],
            's': ['so', 'see', 'some', 'she', 'should', 'start', 'still', 'say'],
            'c': ['can', 'could', 'come', 'call', 'change', 'create', 'complete'],
            'b': ['be', 'but', 'by', 'been', 'back', 'between', 'before', 'both'],
            'o': ['of', 'or', 'on', 'out', 'over', 'only', 'other', 'our'],
            'f': ['for', 'from', 'first', 'find', 'feel', 'following', 'full'],
            'n': ['not', 'now', 'new', 'need', 'no', 'never', 'next', 'number'],
            'm': ['may', 'more', 'make', 'much', 'must', 'my', 'me', 'most'],
            'p': ['people', 'project', 'please', 'put', 'program', 'place', 'point'],
            'y': ['you', 'your', 'yes', 'year', 'yet', 'young', 'yourself']
        }
    
    def train_on_corpus(self, corpus_text: str):
        """Train the model on a text corpus - optimized for speed"""
        print(f"Training optimized model on corpus ({len(corpus_text)} characters)...")
        start_time = time.time()
        
        # Clean and tokenize
        words = self._tokenize(corpus_text)
        self.total_words = len(words)
        
        # Build vocabulary and word frequencies
        word_freq_counter = Counter(words)
        self.word_frequencies = word_freq_counter
        self.vocab = set(words)
        
        print(f"Vocabulary size: {len(self.vocab)}")
        
        # Build n-grams efficiently
        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i + 1]
            
            # Bigrams
            self.bigram_counts[current_word][next_word] += 1
            
            # Trigrams (only if we have enough context)
            if i > 0:
                prev_word = words[i - 1]
                context = f"{prev_word} {current_word}"
                self.trigram_counts[context][next_word] += 1
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Bigram entries: {len(self.bigram_counts)}")
        print(f"Trigram entries: {len(self.trigram_counts)}")
    
    def _tokenize(self, text: str) -> List[str]:
        """Fast tokenization with basic cleaning"""
        # Convert to lowercase and split on whitespace
        text = text.lower()
        # Remove extra whitespace and split
        words = re.findall(r'\b\w+\b', text)
        return words
    
    def predict_next_words(self, context: str = "", partial_word: str = "", top_k: int = 10) -> List[Tuple[str, float]]:
        """Advanced intelligent prediction with multi-layered intelligence"""
        if not context and not partial_word:
            return [("the", 0.3), ("I", 0.25), ("and", 0.2), ("a", 0.15), ("to", 0.1)]
        
        try:
            context = context.lower().strip()
            partial_word = partial_word.lower().strip()
            
            # Get base predictions using optimized n-gram model
            if partial_word:
                base_predictions = self._complete_word(partial_word, top_k)
            elif context:
                base_predictions = self._predict_next_word(context, top_k)
            else:
                base_predictions = self._get_sentence_starters(top_k)
            
            # Learn from the full input text (including unknown words) - with error handling
            full_text = f"{context} {partial_word}".strip()
            if full_text and hasattr(self, 'dynamic_learning'):
                try:
                    learning_stats = self.dynamic_learning.learn_from_text(full_text, self.vocab)
                    if learning_stats and learning_stats.get('new_words_learned', 0) > 0:
                        # Update our vocabulary with newly learned words
                        for word in learning_stats.get('words_promoted', []):
                            self.vocab.add(word)
                            freq = self.dynamic_learning.unknown_word_frequencies.get(word, 5)
                            self.word_frequencies[word] = freq
                            print(f"📚 Added '{word}' to vocabulary with frequency {freq}")
                except Exception as e:
                    print(f"Dynamic learning error: {e}")
            
            # Enhance predictions with learned unknown words - with error handling
            unknown_word_predictions = []
            if hasattr(self, 'dynamic_learning'):
                try:
                    unknown_word_predictions = self.dynamic_learning.get_unknown_word_predictions(context, partial_word)
                    
                    # If we have partial word, also check learned word completions
                    if partial_word:
                        learned_completions = self.dynamic_learning.get_learned_words_for_completion(partial_word)
                        if learned_completions:
                            unknown_word_predictions.extend(learned_completions)
                except Exception as e:
                    print(f"Unknown word prediction error: {e}")
                    unknown_word_predictions = []
            
            # Combine base predictions with unknown word predictions - with error handling
            if unknown_word_predictions and base_predictions:
                try:
                    # Ensure base_predictions is in correct format
                    if not isinstance(base_predictions, list):
                        base_predictions = list(base_predictions)
                    
                    # Boost learned word predictions
                    enhanced_base = [(word, score * 0.8) for word, score in base_predictions if isinstance(word, str) and isinstance(score, (int, float))]  
                    all_predictions = unknown_word_predictions + enhanced_base
                    
                    # Remove duplicates, keeping higher scores
                    seen_words = {}
                    for word, score in all_predictions:
                        if isinstance(word, str) and isinstance(score, (int, float)):
                            if word not in seen_words or score > seen_words[word]:
                                seen_words[word] = score
                    
                    if seen_words:  # Only update if we have valid data
                        combined_predictions = list(seen_words.items())
                        combined_predictions.sort(key=lambda x: x[1], reverse=True)
                        base_predictions = combined_predictions
                except Exception as e:
                    print(f"Prediction combination error: {e}")
                    # Fallback to base predictions if combination fails
            
            # Apply advanced intelligence enhancement
            if hasattr(self, 'smart_context') and base_predictions:
                smart_predictions = self.smart_context.get_smart_predictions(
                    context, partial_word, base_predictions
                )
                
                # Learn from user interaction patterns
                if context:
                    self.smart_context.learn_from_user(context)
                
                return smart_predictions[:top_k]
            
            return base_predictions[:top_k] if base_predictions else self._get_fallback_predictions(context, partial_word, top_k)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._get_fallback_predictions(context, partial_word, top_k)
    
    def _complete_word(self, partial: str, top_k: int) -> List[Tuple[str, float]]:
        """Complete a partial word using frequency-based scoring"""
        candidates = []
        
        # First, try pre-computed common completions
        if partial and partial[0] in self.common_completions:
            common_words = self.common_completions[partial[0]]
            for word in common_words:
                if word.startswith(partial) and len(word) > len(partial):
                    freq = self.word_frequencies.get(word, 1)
                    score = min(freq / self.total_words * 1000, 1.0)  # Normalize
                    candidates.append((word, score))
        
        # Then search vocabulary if needed
        if len(candidates) < top_k:
            for word in self.vocab:
                if word.startswith(partial) and len(word) > len(partial):
                    if not any(word == candidate[0] for candidate in candidates):
                        freq = self.word_frequencies.get(word, 1)
                        score = min(freq / self.total_words * 1000, 0.5)
                        candidates.append((word, score))
                        
                        if len(candidates) >= top_k * 2:  # Get enough candidates
                            break
        
        # Sort by score and return top_k
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]
    
    def _predict_next_word(self, context: str, top_k: int) -> List[Tuple[str, float]]:
        """Predict next word using trigram and bigram models with robust fallbacks"""
        context_words = context.split()
        candidates = []
        
        if len(context_words) >= 2:
            # Try trigram prediction
            trigram_context = f"{context_words[-2]} {context_words[-1]}"
            if trigram_context in self.trigram_counts:
                total = sum(self.trigram_counts[trigram_context].values())
                for word, count in self.trigram_counts[trigram_context].most_common(top_k):
                    score = count / total * 0.8  # High confidence for trigrams
                    candidates.append((word, score))
        
        # Add bigram predictions if we need more candidates
        if len(candidates) < top_k and context_words:
            last_word = context_words[-1]
            if last_word in self.bigram_counts:
                total = sum(self.bigram_counts[last_word].values())
                for word, count in self.bigram_counts[last_word].most_common(top_k):
                    # Avoid duplicates from trigram predictions
                    if not any(word == candidate[0] for candidate in candidates):
                        score = count / total * 0.6  # Lower confidence for bigrams
                        candidates.append((word, score))
        
        # ROBUST FALLBACK: If no candidates found, use fallback suggestions
        if len(candidates) < top_k and context_words:
            last_word = context_words[-1].lower()
            if last_word in self.fallback_suggestions:
                fallback_words = self.fallback_suggestions[last_word]
                for i, word in enumerate(fallback_words[:top_k]):
                    if not any(word == candidate[0] for candidate in candidates):
                        # Decreasing confidence for fallback suggestions
                        score = 0.4 - (i * 0.05)  # 0.4, 0.35, 0.3, etc.
                        candidates.append((word, max(score, 0.1)))
        
        # ULTIMATE FALLBACK: If still no candidates, provide common words
        if not candidates:
            common_fallbacks = ['be', 'have', 'do', 'will', 'can', 'would', 'should']
            for i, word in enumerate(common_fallbacks[:top_k]):
                score = 0.3 - (i * 0.03)  # Decreasing confidence
                candidates.append((word, max(score, 0.1)))
        
        # Sort by score and return top_k
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]
    
    def _get_sentence_starters(self, top_k: int) -> List[Tuple[str, float]]:
        """Get common sentence starters"""
        starters = []
        for word in self.common_starters:
            if word in self.word_frequencies:
                freq = self.word_frequencies[word]
                score = min(freq / self.total_words * 1000, 1.0)
                starters.append((word, score))
        
        starters.sort(key=lambda x: x[1], reverse=True)
        return starters[:top_k]
    
    def _get_fallback_predictions(self, context: str, partial_word: str, top_k: int) -> List[Tuple[str, float]]:
        """Intelligent fallback predictions with typo correction and context awareness"""
        predictions = []
        
        # If we have a partial word, try intelligent completion and typo correction
        if partial_word:
            partial_lower = partial_word.lower()
            
            # 1. First try exact prefix matches
            for word in self.vocab:
                if word.startswith(partial_lower):
                    freq = self.word_frequencies.get(word, 1)
                    score = min(freq / max(self.total_words, 1) * 100, 0.7)
                    predictions.append((word, score))
            
            # 2. If no exact matches, try intelligent typo correction
            if not predictions and len(partial_lower) > 2:
                typo_corrections = self._get_typo_corrections(partial_lower, top_k)
                predictions.extend(typo_corrections)
            
            # 3. Try substring matches (contains the partial word)
            if len(predictions) < top_k // 2:
                for word in self.vocab:
                    if partial_lower in word and not word.startswith(partial_lower):
                        freq = self.word_frequencies.get(word, 1)
                        score = min(freq / max(self.total_words, 1) * 50, 0.4)  # Lower score for substring
                        predictions.append((word, score))
                        if len(predictions) >= top_k:
                            break
            
            # 4. Use smart autocorrect from predictive engine if available
            if hasattr(self, 'predictive_completion') and len(predictions) < top_k:
                autocorrect_suggestions = self.predictive_completion.get_intelligent_autocorrect(partial_word)
                for word, conf in autocorrect_suggestions:
                    if word.lower() in self.vocab:
                        predictions.append((word.lower(), conf * 0.6))  # Slightly lower confidence
        
        # Enhanced context-aware suggestions (no more defaulting to "the"!)
        elif context:
            context_words = context.lower().split()
            if context_words:
                last_word = context_words[-1]
                
                # 1. Use our enhanced fallback suggestions
                if last_word in self.fallback_suggestions:
                    for i, word in enumerate(self.fallback_suggestions[last_word][:top_k]):
                        score = 0.5 - (i * 0.05)
                        predictions.append((word, max(score, 0.2)))
                
                # 2. Use smart context engine for domain-aware suggestions
                if hasattr(self, 'smart_context') and len(predictions) < top_k:
                    smart_suggestions = self.smart_context.get_intelligent_suggestions(context, top_k - len(predictions))
                    for suggestion in smart_suggestions:
                        # Extract words from the suggestion phrase
                        words = suggestion.lower().split()
                        for word in words:
                            if word in self.vocab and not any(word == pred[0] for pred in predictions):
                                predictions.append((word, 0.4))
                                break
                
                # 3. Look for bigram patterns with last word
                if last_word in self.bigram_counts:
                    bigram_total = sum(self.bigram_counts[last_word].values())
                    for word, count in self.bigram_counts[last_word].most_common(3):
                        if not any(word == pred[0] for pred in predictions):
                            score = (count / bigram_total) * 0.3
                            predictions.append((word, score))
        
        # Dynamic learning enhancement - check for learned unknown words
        if hasattr(self, 'dynamic_learning') and len(predictions) < top_k:
            learned_predictions = self.dynamic_learning.get_unknown_word_predictions(context, partial_word)
            predictions.extend(learned_predictions)
        
        # Intelligent ultimate fallback - context-aware, NOT just "the"!
        if not predictions:
            # Analyze context to provide smarter fallbacks
            if context:
                context_lower = context.lower()
                # Question contexts
                if any(q in context_lower for q in ['what', 'how', 'when', 'where', 'why']):
                    smart_fallbacks = [('is', 0.4), ('are', 0.35), ('can', 0.3), ('do', 0.25), ('will', 0.2)]
                # Action contexts
                elif any(a in context_lower for a in ['i', 'we', 'you']):
                    smart_fallbacks = [('can', 0.4), ('should', 0.35), ('will', 0.3), ('need', 0.25), ('want', 0.2)]
                # Object contexts
                elif any(o in context_lower for o in ['the', 'this', 'that']):
                    smart_fallbacks = [('is', 0.4), ('was', 0.35), ('will', 0.3), ('can', 0.25), ('should', 0.2)]
                else:
                    # General smart fallbacks - much better than always "the"!
                    smart_fallbacks = [('is', 0.3), ('and', 0.28), ('to', 0.25), ('can', 0.22), ('will', 0.2)]
            else:
                # Sentence starters - better than "the"
                smart_fallbacks = [('i', 0.3), ('we', 0.25), ('this', 0.22), ('it', 0.2), ('the', 0.18)]
            
            predictions.extend(smart_fallbacks[:top_k])
        
        # Remove duplicates and sort by score
        seen = set()
        unique_predictions = []
        for word, score in predictions:
            if word not in seen:
                seen.add(word)
                unique_predictions.append((word, score))
        
        unique_predictions.sort(key=lambda x: x[1], reverse=True)
        return unique_predictions[:top_k]
    
    def _get_typo_corrections(self, word: str, max_corrections: int = 3) -> List[Tuple[str, float]]:
        """Get intelligent typo corrections using edit distance"""
        corrections = []
        
        # Only try typo correction for words 3+ characters
        if len(word) < 3:
            return corrections
        
        def edit_distance(s1: str, s2: str) -> int:
            """Calculate edit distance between two strings"""
            if len(s1) < len(s2):
                return edit_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        # Find words with small edit distances
        candidates = []
        max_edit_distance = min(2, len(word) // 3)  # Allow 1-2 character differences
        
        for vocab_word in self.vocab:
            if abs(len(vocab_word) - len(word)) <= max_edit_distance:
                distance = edit_distance(word, vocab_word)
                if distance <= max_edit_distance:
                    freq = self.word_frequencies.get(vocab_word, 1)
                    # Score based on edit distance and frequency
                    score = (1.0 / (distance + 1)) * min(freq / max(self.total_words, 1) * 100, 0.5)
                    candidates.append((vocab_word, score))
        
        # Sort by score and return top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:max_corrections]
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model to a file"""
        try:
            model_data = {
                'bigram_counts': {k: dict(v) for k, v in self.bigram_counts.items()},
                'trigram_counts': {k: dict(v) for k, v in self.trigram_counts.items()},
                'word_frequencies': dict(self.word_frequencies),
                'vocab': list(self.vocab),
                'total_words': self.total_words,
                'max_n': self.max_n
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(model_data, f, ensure_ascii=False, separators=(',', ':'))  # Compact format
            
            print(f"Optimized model saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained model from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
            
            self.bigram_counts = defaultdict(Counter)
            for k, v in model_data['bigram_counts'].items():
                self.bigram_counts[k] = Counter(v)
            
            self.trigram_counts = defaultdict(Counter)
            for k, v in model_data['trigram_counts'].items():
                self.trigram_counts[k] = Counter(v)
            
            self.word_frequencies = Counter(model_data['word_frequencies'])
            self.vocab = set(model_data['vocab'])
            self.total_words = model_data['total_words']
            self.max_n = model_data.get('max_n', 3)
            
            print(f"Optimized model loaded from {filepath}")
            print(f"Vocabulary: {len(self.vocab)} words, Total: {self.total_words}")
            return True
        except FileNotFoundError:
            print(f"Model file {filepath} not found")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


def create_optimized_training_corpus() -> str:
    """Create a focused training corpus for the optimized model"""
    
    # Focused, high-quality training data
    training_texts = [
        # Common conversational patterns
        "Hello how are you today? I am doing well thanks for asking. What are you working on?",
        "Can you help me with this project? Sure I would be happy to help you out.",
        "Let me know if you need anything else. Thanks for your help and support.",
        "I think we should start working on this right away. That sounds like a good plan.",
        "Please let me know when you are ready to begin. I will be ready in a few minutes.",
        
        # Business communication
        "I will send you the report by the end of the day. Thank you for the update.",
        "We need to schedule a meeting to discuss the project details and timeline.",
        "Could you please review the document and provide your feedback by tomorrow?",
        "The presentation went very well and the client was satisfied with our work.",
        "I would like to schedule a call to discuss the next steps in the process.",
        
        # Technical writing
        "The system is working properly and all tests are passing successfully.",
        "We need to implement the new feature before the next release cycle.",
        "Please make sure to backup your data before running the update process.",
        "The application is running smoothly and performance has been optimized.",
        "You can find the documentation in the project folder under the docs directory.",
        
        # Daily communication
        "Good morning! I hope you have a great day ahead of you today.",
        "Thank you for your time and consideration. I look forward to hearing from you.",
        "Have a wonderful weekend and I will see you next week at the office.",
        "It was great meeting you today and I enjoyed our conversation about the project.",
        "Please feel free to contact me if you have any questions or concerns.",
    ]
    
    # Multiply the training data for better statistics
    full_corpus = " ".join(training_texts * 10)  # Repeat for better n-gram coverage
    
    return full_corpus


if __name__ == "__main__":
    # Test the optimized model
    model = OptimizedLanguageModel(max_n=3)
    
    # Create and train on corpus
    corpus = create_optimized_training_corpus()
    model.train_on_corpus(corpus)
    
    # Test predictions
    print("\n=== Testing Optimized Model ===")
    
    # Test word completion
    print("\nWord completion test:")
    result = model.predict_next_words(partial_word="th", top_k=3)
    print(f"Completing 'th': {result}")
    
    # Test next word prediction
    print("\nNext word prediction test:")
    result = model.predict_next_words(context="I would like to", top_k=3)
    print(f"After 'I would like to': {result}")
    
    # Test sentence starters
    print("\nSentence starter test:")
    result = model.predict_next_words(top_k=5)
    print(f"Sentence starters: {result}")
    
    # Save model
    model.save_model('optimized_ml_model.json')
