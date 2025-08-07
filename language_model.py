"""
Enhanced ML Language Model with Variable N-grams and Contextual Features
"""

import json
import re
from collections import defaultdict, Counter
from typing import List, Tuple, Dict
from datetime import datetime


class EnhancedMLLanguageModel:
    """
    Enhanced ML language model with variable n-gram sizes and contextual features
    for superior next-word prediction
    """
    
    def __init__(self, max_n=5):
        self.max_n = max_n  # Maximum n-gram size (5 = 5-grams)
        # Store multiple n-gram sizes for fallback
        self.ngram_counts = {n: defaultdict(Counter) for n in range(2, max_n + 1)}
        self.vocab = set()
        self.total_words = 0
        self.smoothing_alpha = 0.01  # Laplace smoothing parameter
        
        # Enhanced features
        self.user_word_frequency = Counter()  # Track user's frequent words
        self.contextual_patterns = self._initialize_contextual_patterns()
        
    def _initialize_contextual_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize contextual scoring patterns for enhanced predictions"""
        return {
            'time_patterns': {
                'morning': ['good', 'coffee', 'breakfast', 'meeting', 'start', 'begin', 'early'],
                'afternoon': ['lunch', 'work', 'project', 'busy', 'progress', 'update'],
                'evening': ['dinner', 'home', 'tired', 'relax', 'watch', 'rest', 'end']
            },
            'sentence_starters': {
                'high_freq': ['the', 'i', 'we', 'this', 'that', 'it', 'you', 'they'],
                'question': ['what', 'how', 'when', 'where', 'why', 'who', 'which'],
                'action': ['let', 'please', 'can', 'will', 'should', 'would', 'could']
            },
            'sentence_enders': {
                'completion': ['thanks', 'regards', 'best', 'sincerely'],
                'punctuation': ['.', '!', '?', ','],
                'closing': ['bye', 'goodbye', 'talk', 'later', 'soon']
            },
            'common_phrases': {
                'i would like to': ['go', 'have', 'be', 'do', 'see', 'get'],
                'we need to': ['discuss', 'review', 'complete', 'finish', 'start'],
                'let me know': ['if', 'when', 'what', 'how', 'whether'],
                'thank you for': ['your', 'the', 'helping', 'sharing', 'sending'],
                'we are expecting': ['the', 'good', 'better', 'positive', 'great'],
                'test test test': ['results', 'data', 'outcome', 'analysis']
            }
        }
        
    def preprocess_text(self, text: str) -> List[str]:
        """Clean and tokenize text for training"""
        # Convert to lowercase and split into words
        text = text.lower()
        # Remove extra whitespace and split
        words = re.findall(r'\b\w+\b', text)
        return words
    
    def train_on_text(self, text: str):
        """Train the model on a text corpus with multiple n-gram sizes"""
        words = self.preprocess_text(text)
        if len(words) < 2:
            return
            
        # Add words to vocabulary
        self.vocab.update(words)
        self.total_words += len(words)
        
        # Generate n-grams of different sizes (2-gram through max_n-gram)
        for n in range(2, min(self.max_n + 1, len(words) + 1)):
            for i in range(len(words) - n):
                # Get n-gram context (previous n words)
                ngram = tuple(words[i:i + n])
                # Get next word
                next_word = words[i + n]
                # Count this transition for this n-gram size
                self.ngram_counts[n][ngram][next_word] += 1
    
    def train_on_corpus(self, corpus_texts: List[str]):
        """Train on multiple text samples"""
        print(f"🧠 Training enhanced ML language model on {len(corpus_texts)} text samples...")
        for i, text in enumerate(corpus_texts):
            self.train_on_text(text)
            if (i + 1) % 100 == 0:
                print(f"   Processed {i + 1}/{len(corpus_texts)} samples...")
        
        print(f"✅ Enhanced ML model trained:")
        print(f"   - Vocabulary size: {len(self.vocab)} words")
        total_patterns = sum(len(counts) for counts in self.ngram_counts.values())
        print(f"   - Total n-gram patterns: {total_patterns}")
        for n in range(2, self.max_n + 1):
            if n in self.ngram_counts:
                print(f"     • {n}-grams: {len(self.ngram_counts[n])} patterns")
        print(f"   - Total words processed: {self.total_words}")
    
    def predict_next_words(self, context: str, partial_word: str = "", top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Enhanced ML prediction with contextual scoring and smart fallback
        Returns list of (word, probability) tuples
        """
        words = self.preprocess_text(context)
        
        # Try different n-gram sizes from largest to smallest (smart fallback)
        predictions = []
        for n in range(min(self.max_n, len(words) + 1), 1, -1):
            if n not in self.ngram_counts:
                continue
                
            # Get n-gram context
            if len(words) >= n - 1:
                ngram = tuple(words[-(n-1):]) if n > 1 else tuple()
            else:
                continue
                
            # Check if this n-gram exists in our model
            if ngram in self.ngram_counts[n]:
                next_word_counts = self.ngram_counts[n][ngram]
                total_count = sum(next_word_counts.values())
                
                # Calculate base probabilities with Laplace smoothing
                for word, count in next_word_counts.items():
                    if partial_word == "" or word.startswith(partial_word.lower()):
                        base_prob = (count + self.smoothing_alpha) / (total_count + self.smoothing_alpha * len(self.vocab))
                        
                        # Apply contextual scoring
                        enhanced_prob = self._apply_contextual_scoring(context, word, base_prob)
                        
                        predictions.append((word, enhanced_prob, n))  # Include n-gram size for debugging
                
                # If we found predictions with this n-gram size, use them
                if predictions:
                    print(f"🎯 Using {n}-gram predictions for context: '{' '.join(words[-(n-1):]) if n > 1 else 'no context'}'")
                    break
        
        # If no n-gram predictions found, use fallback
        if not predictions:
            return self._enhanced_fallback_prediction(context, partial_word, top_k)
        
        # Sort by enhanced probability (highest first) and return top k
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [(word, prob) for word, prob, n in predictions[:top_k]]
    
    def _apply_contextual_scoring(self, context: str, word: str, base_prob: float) -> float:
        """Apply sophisticated contextual scoring for semantic understanding"""
        enhanced_prob = base_prob
        
        # Boost semantic phrase completions (much stronger boost)
        context_lower = context.lower()
        word_lower = word.lower()
        
        # Enhanced semantic patterns (5x boost for strong semantic matches)
        semantic_patterns = {
            # Experiment/research contexts
            'expecting the results': ['to', 'will', 'should', 'today', 'soon', 'tomorrow'],
            'results of our experiment': ['to', 'will', 'show', 'demonstrate', 'prove', 'indicate'],
            'our experiment': ['will', 'should', 'to', 'is', 'was', 'today'],
            'the experiment': ['will', 'should', 'to', 'is', 'was', 'shows'],
            
            # Common sentence starters
            'we are': ['expecting', 'going', 'planning', 'working', 'looking', 'trying'],
            'i am': ['going', 'working', 'looking', 'thinking', 'planning', 'trying'],
            'this is': ['the', 'a', 'an', 'very', 'really', 'important'],
            'it is': ['the', 'a', 'very', 'really', 'important', 'time'],
            
            # Action contexts
            'we need to': ['get', 'do', 'make', 'find', 'start', 'finish'],
            'i want to': ['go', 'do', 'make', 'get', 'see', 'try'],
            'let me': ['know', 'see', 'check', 'try', 'get', 'do'],
            
            # Question contexts
            'what do you': ['think', 'want', 'need', 'mean', 'say', 'know'],
            'how do you': ['do', 'make', 'get', 'know', 'think', 'feel'],
            'where do you': ['go', 'live', 'work', 'think', 'want', 'need']
        }
        
        # Check for semantic matches (strong boost)
        for phrase, completions in semantic_patterns.items():
            if phrase in context_lower and word_lower in completions:
                enhanced_prob *= 5.0  # Strong semantic boost
                print(f"🎯 Semantic boost: '{phrase}' → '{word}' (5x boost)")
                break
        
        # Boost user's frequently typed words (moderate boost)
        if word_lower in self.user_word_frequency:
            user_boost = min(self.user_word_frequency[word_lower] / 5.0, 2.0)
            enhanced_prob *= (1.0 + user_boost)
            print(f"👤 User frequency boost: '{word}' ({user_boost:.1f}x boost)")
        
        # Sentence position context
        words_in_context = context_lower.split()
        if len(words_in_context) > 0:
            last_word = words_in_context[-1]
            
            # Verb → object patterns
            verb_object_patterns = {
                'expecting': ['the', 'good', 'great', 'positive', 'results'],
                'getting': ['the', 'good', 'better', 'results', 'ready'],
                'making': ['the', 'good', 'sure', 'progress', 'changes'],
                'working': ['on', 'with', 'hard', 'together', 'late']
            }
            
            if last_word in verb_object_patterns and word_lower in verb_object_patterns[last_word]:
                enhanced_prob *= 3.0  # Verb-object boost
                print(f"🔗 Verb-object boost: '{last_word}' → '{word}' (3x boost)")
        
        return enhanced_prob
    
    def _enhanced_fallback_prediction(self, context: str, partial_word: str, top_k: int) -> List[Tuple[str, float]]:
        """Enhanced fallback with contextual awareness"""
        print("🔄 Using enhanced fallback prediction")
        
        # Count word frequencies across all n-gram sizes
        word_freq = Counter()
        for n_size in self.ngram_counts:
            for ngram_dict in self.ngram_counts[n_size].values():
                for word, count in ngram_dict.items():
                    if partial_word == "" or word.startswith(partial_word.lower()):
                        word_freq[word] += count
        
        total = sum(word_freq.values())
        if total == 0:
            return []
        
        # Convert to probabilities and apply contextual scoring
        predictions = []
        for word, count in word_freq.items():
            base_prob = count / total
            enhanced_prob = self._apply_contextual_scoring(context, word, base_prob)
            predictions.append((word, enhanced_prob))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k]
    
    def learn_from_user_input(self, text: str):
        """Learn from user's actual typing for personalization"""
        words = self.preprocess_text(text)
        for word in words:
            if len(word) > 2:  # Only learn meaningful words
                self.user_word_frequency[word] += 1
        
        # Keep only top 100 most frequent user words to prevent memory bloat
        if len(self.user_word_frequency) > 100:
            # Keep top 100 most frequent
            top_words = dict(self.user_word_frequency.most_common(100))
            self.user_word_frequency = Counter(top_words)
    
    def get_model_stats(self) -> Dict:
        """Get statistics about the enhanced trained model"""
        total_patterns = sum(len(counts) for counts in self.ngram_counts.values())
        ngram_breakdown = {f'{n}_grams': len(self.ngram_counts[n]) for n in self.ngram_counts}
        
        return {
            'vocab_size': len(self.vocab),
            'total_ngram_patterns': total_patterns,
            'ngram_breakdown': ngram_breakdown,
            'total_words_trained': self.total_words,
            'max_n_gram_size': self.max_n,
            'smoothing_alpha': self.smoothing_alpha,
            'user_learned_words': len(self.user_word_frequency),
            'contextual_patterns': len(self.contextual_patterns)
        }
    
    def save_model(self, filepath: str):
        """Save the enhanced trained model to a JSON file"""
        # Convert nested defaultdict structure to regular dicts for JSON serialization
        ngram_data = {}
        for n in self.ngram_counts:
            ngram_data[str(n)] = {}
            for ngram, word_counts in self.ngram_counts[n].items():
                ngram_key = '|'.join(ngram)  # Convert tuple to string
                ngram_data[str(n)][ngram_key] = dict(word_counts)
        
        model_data = {
            'ngram_counts': ngram_data,
            'vocab': list(self.vocab),
            'total_words': self.total_words,
            'max_n_gram_size': self.max_n,
            'smoothing_alpha': self.smoothing_alpha,
            'user_word_frequency': dict(self.user_word_frequency),
            'contextual_patterns': self.contextual_patterns
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2)
        print(f"💾 Enhanced ML model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load an enhanced trained model from disk"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
            
            self.max_n = model_data.get('max_n_gram_size', 5)
            self.vocab = set(model_data['vocab'])
            self.total_words = model_data['total_words']
            self.smoothing_alpha = model_data['smoothing_alpha']
            
            # Load user learning data
            self.user_word_frequency = Counter(model_data.get('user_word_frequency', {}))
            self.contextual_patterns = model_data.get('contextual_patterns', self._initialize_contextual_patterns())
            
            # Reconstruct ngram_counts for multiple n-gram sizes
            self.ngram_counts = {n: defaultdict(Counter) for n in range(2, self.max_n + 1)}
            for n_str, ngram_data in model_data['ngram_counts'].items():
                n = int(n_str)
                for ngram_key, word_counts in ngram_data.items():
                    ngram = tuple(ngram_key.split('|'))  # Convert string back to tuple
                    self.ngram_counts[n][ngram] = Counter(word_counts)
            
            print(f"📚 Enhanced ML model loaded from {filepath}")
            print(f"   - Vocabulary: {len(self.vocab)} words")
            total_patterns = sum(len(counts) for counts in self.ngram_counts.values())
            print(f"   - Total n-gram patterns: {total_patterns}")
            for n in range(2, self.max_n + 1):
                if n in self.ngram_counts:
                    print(f"     • {n}-grams: {len(self.ngram_counts[n])} patterns")
            return True
        except FileNotFoundError:
            print(f"⚠️ Model file {filepath} not found")
            return False


def create_enhanced_training_corpus() -> List[str]:
    """Create a massive training corpus optimized for enhanced n-gram learning"""
    
    # Base corpus with targeted vocabulary
    corpus = [
        # Classic literature and famous phrases
        "the quick brown fox jumps over the lazy dog",
        "once upon a time in a land far far away",
        "it was the best of times it was the worst of times",
        "to be or not to be that is the question",
        "all that glitters is not gold",
        "better late than never",
        "practice makes perfect",
        "actions speak louder than words",
        "the early bird catches the worm",
        "when in rome do as romans do",
        "a picture is worth a thousand words",
        
        # Enhanced patterns for your specific use case
        "we are expecting the results of our experiment",
        "the test results show significant improvement",
        "we need to test this thoroughly",
        "the experiment was a complete success",
        "test test test test test test results",  # Handle repetitive input
        "after many tests we expect good results",
        "the testing phase is almost complete",
        "we expect the test to pass",
        "experimental results exceed expectations",
        "testing testing one two three",
        
        # Comprehensive word coverage for common prefixes
        # Words starting with 'rep'
        "i need to report this issue immediately",
        "please repeat what you just said",
        "we should replace the broken equipment",
        "the company has a good reputation",
        "can you represent our team at the meeting",
        "i will reply to your email soon",
        "the report shows excellent progress",
        "let me repeat the instructions clearly",
        "we need to replace this old system",
        
        # High-frequency English word combinations
        "the quick brown fox jumps over the lazy dog every single day",
        "i am going to the store to buy some food for dinner tonight",
        "she was walking down the street when she saw her old friend",
        "we have been working on this project for several months now",
        "they will be arriving at the airport in about two hours",
        "you can find more information about this topic on our website",
        "it is important to remember that practice makes perfect always",
        "there are many different ways to solve this particular problem",
        "he has been studying computer science at the university for years",
        "we should probably start thinking about our next steps carefully",
        
        # Business and professional language
        "the meeting has been scheduled for next tuesday at three pm",
        "please send me the report as soon as possible today",
        "we need to discuss the budget for the upcoming fiscal year",
        "the project manager will be presenting the results tomorrow morning",
        "our team has been working hard to meet all the deadlines",
        "the client is very satisfied with the quality of our work",
        "we should schedule a follow up meeting to discuss next steps",
        "the presentation went very well and received positive feedback",
        "please make sure to backup all important data before proceeding",
        "the software update includes several new features and improvements",
        
        # Common conversational patterns
        "how are you doing today i hope everything is going well",
        "thank you so much for your help i really appreciate it",
        "i would like to invite you to join us for dinner",
        "it was great to see you at the party last weekend",
        "do you have any plans for the upcoming holiday season",
        "the weather has been absolutely beautiful lately dont you think",
        "i am looking forward to hearing from you soon about this",
        "please let me know if you need any additional information",
        "it would be my pleasure to help you with this project",
        "i hope you have a wonderful day and a great weekend"
    ]
    
    # Generate systematic combinations for better 4-gram and 5-gram coverage
    subjects = ["i", "you", "we", "they", "the team"]
    verbs = ["am", "are", "will be", "have been"]
    actions = ["working", "testing", "expecting", "planning"]
    objects = ["the results", "the experiment", "the project", "the system"]
    
    # Generate 4-gram and 5-gram rich sentences
    for subj in subjects:
        for verb in verbs:
            for action in actions:
                for obj in objects:
                    sentence = f"{subj} {verb} {action} on {obj} with great care and attention"
                    corpus.append(sentence)
    
    # Add repetitive pattern handling (for your specific issue)
    repetitive_patterns = [
        "test " * i + "results are expected soon" for i in range(1, 8)
    ]
    corpus.extend(repetitive_patterns)
    
    print(f"📚 Generated enhanced training corpus: {len(corpus)} samples")
    return corpus


if __name__ == "__main__":
    # Test the enhanced model
    model = EnhancedMLLanguageModel(max_n=5)
    corpus = create_enhanced_training_corpus()
    
    # Train the model
    model.train_on_corpus(corpus)
    
    # Test predictions
    print("\n🧠 Enhanced ML Model Predictions:")
    
    test_cases = [
        ("we are expecting the results of our", "expe"),
        ("test test test test test", ""),
        ("i would like to", ""),
        ("the quick brown", ""),
        ("we need to", "")
    ]
    
    for context, partial in test_cases:
        predictions = model.predict_next_words(context, partial, top_k=3)
        print(f"\nContext: '{context}' + partial: '{partial}'")
        for i, (word, prob) in enumerate(predictions, 1):
            print(f"  {i}. {word} (probability: {prob:.4f})")
    
    # Save the model
    model.save_model('enhanced_ml_language_model.json')
    print("\n✅ Enhanced ML model trained and saved!")
