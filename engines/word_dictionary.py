"""Word Dictionary - Comprehensive English dictionary for predictive text"""

from typing import List, Dict, Set, Tuple
import re
import os


class WordDictionary:
    def __init__(self):
        self.words = set()
        self.word_frequencies = {}
        self.prefix_tree = {}
        
        self._load_dictionary()
        
        self.prefix_tree = self._build_prefix_tree()
    
    def _load_dictionary(self):
        try:
            with open('words.txt', 'r', encoding='utf-8') as f:
                words = [line.strip().lower() for line in f if line.strip()]
                self.words = set(words)
                print(f"Loaded {len(self.words)} words from dictionary")
                
                common_words = self._get_common_words()
                
                for word in self.words:
                    if word in common_words:
                        if word in {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at'}:
                            base_freq = 1000  
                        elif word in {'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their'}:
                            base_freq = 500   
                        else:
                            base_freq = 200   
                    elif len(word) <= 3:  
                        base_freq = 100
                    elif len(word) <= 5:  
                        base_freq = 50
                    elif len(word) <= 7:  
                        base_freq = 20
                    else: 
                        base_freq = 5
                    
                    self.word_frequencies[word] = base_freq
                    
        except FileNotFoundError:
            print("Warning: words.txt not found, using minimal fallback dictionary")
            self._load_fallback_dictionary()
    
    def _get_common_words(self) -> Set[str]:
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
            self.word_frequencies[word] = 50 
    
    def _build_prefix_tree(self) -> Dict:
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
        if not prefix:
            common_starters = [
                ('the', 1000), ('i', 1000), ('to', 1000), ('a', 1000), ('and', 1000),
                ('you', 1000), ('it', 1000), ('that', 1000), ('is', 200), ('for', 1000)
            ]
            return common_starters[:max_results]
        
        prefix = prefix.lower().strip()
        current = self.prefix_tree
        
        for char in prefix:
            if char not in current:
                return []
            current = current[char]
        
        completions = []
        self._collect_completions(current, prefix, completions)
        
        completions.sort(key=lambda x: x[1], reverse=True)
        return completions[:max_results]
    
    def _collect_completions(self, node: Dict, prefix: str, completions: List):
        if '_word' in node:
            completions.append((node['_word'], node['_freq']))
        
        for char, child_node in node.items():
            if char not in ['_word', '_freq']:
                self._collect_completions(child_node, prefix + char, completions)
    
    def get_next_char_predictions(self, current_word: str) -> Dict[str, float]:
        if not current_word:
            return self._get_valid_starting_chars()
        
        valid_chars = self._get_valid_next_chars(current_word)
        
        if not valid_chars:
            return {}
        
        return valid_chars
    
    def _get_valid_starting_chars(self) -> Dict[str, float]:
        char_scores = {}
        
        common_words = ['the', 'and', 'you', 'that', 'was', 'for', 'are', 'with', 'his', 'they', 
                       'have', 'this', 'will', 'can', 'had', 'her', 'what', 'said', 'each', 'which',
                       'she', 'how', 'their', 'has', 'two', 'more', 'like', 'very', 'time', 'up',
                       'out', 'many', 'then', 'them', 'these', 'so', 'some', 'would', 'make', 'into']
        
        for word in common_words:
            if word in self.words:
                first_char = word[0].lower()
                freq = self.word_frequencies.get(word, 1)
                char_scores[first_char] = char_scores.get(first_char, 0) + freq
        
        if char_scores:
            total = sum(char_scores.values())
            char_scores = {char: score / total for char, score in char_scores.items()}
        
        return char_scores
    
    def _get_valid_next_chars(self, current_word: str) -> Dict[str, float]:
        char_scores = {}
        current_word = current_word.lower()
        
        is_complete_word = current_word in self.words
        
        matching_words = []
        for word in self.words:
            if word.startswith(current_word) and len(word) > len(current_word):
                matching_words.append((word, self.word_frequencies.get(word, 1)))
        
        if is_complete_word and not matching_words:
            return {} 
        
        if not matching_words:
            return {}
        
        for word, freq in matching_words:
            next_char = word[len(current_word)].lower()
            import math
            score = math.log(freq + 1) * 10
            
            if is_complete_word:
                score *= 0.3
            
            char_scores[next_char] = char_scores.get(next_char, 0) + score
        
        if char_scores:
            total = sum(char_scores.values())
            char_scores = {char: score / total for char, score in char_scores.items()}
            
            char_scores = {k: v for k, v in char_scores.items() if v > 0.02}
        
        return char_scores
    
    def is_valid_word(self, word: str) -> bool:
        return word.lower() in self.words
    
    def get_word_frequency(self, word: str) -> int:
        return self.word_frequencies.get(word.lower(), 0)
