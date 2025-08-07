"""Word Dictionary - Comprehensive English dictionary for predictive text"""

from typing import List, Dict, Set, Tuple
import re
import os


class WordDictionary:
    def __init__(self):
        # Load comprehensive dictionary from file
        self.words = set()
        self.word_frequencies = {}
        self.prefix_tree = {}
        
        # Load the comprehensive word list
        self._load_dictionary()
        
        # Build prefix tree for fast lookups
        self.prefix_tree = self._build_prefix_tree()
    
    def _load_dictionary(self):
        """Load comprehensive English dictionary from words.txt file"""
        try:
            with open('words.txt', 'r', encoding='utf-8') as f:
                words = [line.strip().lower() for line in f if line.strip()]
                self.words = set(words)
                print(f"Loaded {len(self.words)} words from dictionary")
                
                # Assign frequency scores based on word length and commonality
                # Use much more aggressive frequency weighting for common words
                common_words = self._get_common_words()
                
                for word in self.words:
                    if word in common_words:
                        # Super high frequency for most common words
                        if word in {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at'}:
                            base_freq = 1000  # Extremely high
                        elif word in {'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their'}:
                            base_freq = 500   # Very high
                        else:
                            base_freq = 200   # High for other common words
                    elif len(word) <= 3:  # Very short words are likely common
                        base_freq = 100
                    elif len(word) <= 5:  # Short words are somewhat common
                        base_freq = 50
                    elif len(word) <= 7:  # Medium words
                        base_freq = 20
                    else:  # Long words are typically rare
                        base_freq = 5
                    
                    self.word_frequencies[word] = base_freq
                    
        except FileNotFoundError:
            print("Warning: words.txt not found, using minimal fallback dictionary")
            # Fallback to minimal dictionary
            self._load_fallback_dictionary()
    
    def _get_common_words(self) -> Set[str]:
        """Return set of most common English words for frequency boosting"""
        return {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on',
            'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we',
            'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
            'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when',
            'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take', 'people', 'into',
            'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now',
            'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two',
            'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any',
            'these', 'give', 'day', 'most', 'us', 'is', 'was', 'are', 'been', 'has', 'had', 'were',
            'said', 'each', 'which', 'did', 'very', 'where', 'much', 'too', 'may', 'such', 'here',
            'every', 'why', 'many', 'write', 'number', 'water', 'call', 'oil', 'sit', 'find', 'long',
            'down', 'made', 'part', 'am', 'as', 'at', 'by', 'he', 'if', 'in', 'is', 'it', 'me', 'my',
            'no', 'of', 'on', 'or', 'so', 'to', 'up', 'we'
        }
    
    def _load_fallback_dictionary(self):
        """Load minimal fallback dictionary if main file not found"""
        fallback_words = [
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on',
            'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we',
            'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
            'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when',
            'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take', 'people', 'into',
            'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now',
            'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two',
            'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any',
            'these', 'give', 'day', 'most', 'us'
        ]
        
        self.words = set(fallback_words)
        for word in fallback_words:
            self.word_frequencies[word] = 50  # High frequency for common words
    
    def _build_prefix_tree(self) -> Dict:
        """Build a prefix tree (trie) for fast word lookups"""
        tree = {}
        for word in self.words:
            frequency = self.word_frequencies.get(word, 1)
            current = tree
            for char in word.lower():
                if char not in current:
                    current[char] = {}
                current = current[char]
            current['_word'] = word
            current['_freq'] = frequency
        return tree
    
    def get_word_completions(self, prefix: str, max_results: int = 5) -> List[tuple]:
        """Get word completions for a given prefix"""
        if not prefix:
            # Return most common starting words with proper frequencies
            common_starters = [
                ('the', 1000), ('i', 1000), ('to', 1000), ('a', 1000), ('and', 1000),
                ('you', 1000), ('it', 1000), ('that', 1000), ('is', 200), ('for', 1000)
            ]
            return common_starters[:max_results]
        
        prefix = prefix.lower().strip()
        current = self.prefix_tree
        
        # Navigate to the prefix
        for char in prefix:
            if char not in current:
                return []
            current = current[char]
        
        # Collect all completions
        completions = []
        self._collect_completions(current, prefix, completions)
        
        # Sort by frequency and return top results
        completions.sort(key=lambda x: x[1], reverse=True)
        return completions[:max_results]
    
    def _collect_completions(self, node: Dict, prefix: str, completions: List):
        """Recursively collect all word completions from a node"""
        if '_word' in node:
            completions.append((node['_word'], node['_freq']))
        
        for char, child_node in node.items():
            if char not in ['_word', '_freq']:
                self._collect_completions(child_node, prefix + char, completions)
    
    def get_next_char_predictions(self, current_word: str) -> Dict[str, float]:
        """Get predictions for the next character that lead to valid words"""
        if not current_word:
            # Get actual first characters of common words
            return self._get_valid_starting_chars()
        
        # Get all valid next characters that lead to real words
        valid_chars = self._get_valid_next_chars(current_word)
        
        if not valid_chars:
            return {}
        
        return valid_chars
    
    def _get_valid_starting_chars(self) -> Dict[str, float]:
        """Get valid starting characters weighted by word frequency"""
        char_scores = {}
        
        # Get most common words and their first characters
        common_words = ['the', 'and', 'you', 'that', 'was', 'for', 'are', 'with', 'his', 'they', 
                       'have', 'this', 'will', 'can', 'had', 'her', 'what', 'said', 'each', 'which',
                       'she', 'how', 'their', 'has', 'two', 'more', 'like', 'very', 'time', 'up',
                       'out', 'many', 'then', 'them', 'these', 'so', 'some', 'would', 'make', 'into']
        
        for word in common_words:
            if word in self.words:
                first_char = word[0].lower()
                freq = self.word_frequencies.get(word, 1)
                char_scores[first_char] = char_scores.get(first_char, 0) + freq
        
        # Normalize to probabilities
        if char_scores:
            total = sum(char_scores.values())
            char_scores = {char: score / total for char, score in char_scores.items()}
        
        return char_scores
    
    def _get_valid_next_chars(self, current_word: str) -> Dict[str, float]:
        """Get next characters that lead to valid words, weighted by word frequency"""
        char_scores = {}
        current_word = current_word.lower()
        
        # Check if current word is already a complete valid word
        is_complete_word = current_word in self.words
        
        # Find all words that start with current_word and are longer
        matching_words = []
        for word in self.words:
            if word.startswith(current_word) and len(word) > len(current_word):
                matching_words.append((word, self.word_frequencies.get(word, 1)))
        
        # If current word is complete and no extensions exist, don't suggest letters
        if is_complete_word and not matching_words:
            return {}  # This will show no letter predictions, allowing space/punctuation
        
        # If no matching words at all, return empty (prevents invalid paths)
        if not matching_words:
            return {}
        
        # Get next characters from matching words
        for word, freq in matching_words:
            next_char = word[len(current_word)].lower()
            # Use logarithmic scaling for frequency weighting
            import math
            score = math.log(freq + 1) * 10
            
            # If current word is already complete, reduce scores for extensions
            # This encourages completing the word rather than extending it
            if is_complete_word:
                score *= 0.3  # Reduce extension probability when word is complete
            
            char_scores[next_char] = char_scores.get(next_char, 0) + score
        
        # Normalize to probabilities
        if char_scores:
            total = sum(char_scores.values())
            char_scores = {char: score / total for char, score in char_scores.items()}
            
            # Only return characters with reasonable probability (>2%)
            char_scores = {k: v for k, v in char_scores.items() if v > 0.02}
        
        return char_scores
    
    def is_valid_word(self, word: str) -> bool:
        """Check if a word exists in the dictionary"""
        return word.lower() in self.words
    
    def get_word_frequency(self, word: str) -> int:
        """Get the frequency score of a word"""
        return self.word_frequencies.get(word.lower(), 0)
