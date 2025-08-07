"""
Predictive Completion Engine - Advanced sentence and phrase completion
"""

import re
from typing import List, Tuple, Dict, Set
from collections import defaultdict, Counter
import difflib


class PredictiveCompletionEngine:
    """Advanced predictive completion with sentence-level intelligence"""
    
    def __init__(self):
        self.sentence_templates = self._build_sentence_templates()
        self.phrase_completions = self._build_phrase_completions()
        self.grammar_patterns = self._build_grammar_patterns()
        self.user_sentence_history = []
        self.completion_success_rate = Counter()
        
    def _build_sentence_templates(self) -> Dict[str, List[str]]:
        """Build intelligent sentence completion templates"""
        return {
            'questions': {
                'what': [
                    'what time is the meeting?',
                    'what do you think about this?',
                    'what are the next steps?',
                    'what should we do next?'
                ],
                'how': [
                    'how can we improve this?',
                    'how does this work?',
                    'how long will this take?',
                    'how should we proceed?'
                ],
                'when': [
                    'when is the deadline?',
                    'when can we schedule this?',
                    'when will this be ready?',
                    'when should we start?'
                ],
                'where': [
                    'where should we meet?',
                    'where is the document?',
                    'where can I find this?',
                    'where do we go from here?'
                ]
            },
            'statements': {
                'i think': [
                    'i think we should proceed with this plan',
                    'i think this is a good approach',
                    'i think we need more time',
                    'i think this will work well'
                ],
                'we need': [
                    'we need to discuss this further',
                    'we need more information',
                    'we need to test this thoroughly',
                    'we need to make a decision'
                ],
                'please': [
                    'please let me know if you have questions',
                    'please review this document',
                    'please confirm your availability',
                    'please send me the details'
                ],
                'thank you': [
                    'thank you for your time and consideration',
                    'thank you for the update',
                    'thank you for your help',
                    'thank you for clarifying this'
                ]
            },
            'business': {
                'i would like to': [
                    'i would like to schedule a meeting to discuss',
                    'i would like to follow up on our conversation',
                    'i would like to propose the following changes',
                    'i would like to get your feedback on'
                ],
                'looking forward': [
                    'looking forward to hearing from you',
                    'looking forward to our meeting',
                    'looking forward to working together',
                    'looking forward to the next steps'
                ]
            }
        }
    
    def _build_phrase_completions(self) -> Dict[str, List[Tuple[str, float]]]:
        """Build smart phrase completion patterns with confidence scores"""
        return {
            'in order': [('to', 0.9), ('for', 0.1)],
            'as soon as': [('possible', 0.8), ('you can', 0.2)],
            'thank you': [('for', 0.6), ('so much', 0.3), ('very much', 0.1)],
            'looking forward': [('to', 0.9), ('for', 0.1)],
            'please let': [('me know', 0.7), ('us know', 0.2), ('me', 0.1)],
            'i would': [('like to', 0.5), ('appreciate', 0.3), ('love to', 0.2)],
            'on the other': [('hand', 0.9), ('side', 0.1)],
            'in addition': [('to', 0.8), ('we should', 0.2)],
            'with regard': [('to', 0.9), ('for', 0.1)],
            'at your': [('earliest convenience', 0.6), ('convenience', 0.4)],
            'best': [('regards', 0.7), ('wishes', 0.2), ('practices', 0.1)]
        }
    
    def _build_grammar_patterns(self) -> Dict[str, List[str]]:
        """Build grammatical completion patterns"""
        return {
            'articles': {
                'a': ['new', 'good', 'great', 'quick', 'simple', 'better'],
                'an': ['important', 'interesting', 'excellent', 'easy', 'urgent'],
                'the': ['best', 'most', 'main', 'next', 'first', 'last', 'current']
            },
            'prepositions': {
                'in': ['the', 'order', 'addition', 'case', 'fact'],
                'on': ['the', 'time', 'schedule', 'track'],
                'at': ['the', 'your', 'our', 'this'],
                'for': ['the', 'your', 'our', 'this', 'example'],
                'with': ['the', 'your', 'our', 'this', 'regard']
            },
            'conjunctions': {
                'and': ['we', 'i', 'the', 'this', 'also'],
                'but': ['i', 'we', 'the', 'this', 'also'],
                'or': ['we', 'you', 'the', 'maybe'],
                'so': ['we', 'i', 'the', 'this', 'that']
            }
        }
    
    def predict_sentence_completion(self, partial_sentence: str) -> List[Tuple[str, float]]:
        """Predict intelligent sentence completions"""
        partial_lower = partial_sentence.lower().strip()
        completions = []
        
        # Find matching sentence templates
        for category, patterns in self.sentence_templates.items():
            for pattern_key, sentences in patterns.items():
                if partial_lower.startswith(pattern_key):
                    for sentence in sentences:
                        if sentence.startswith(partial_lower):
                            remaining = sentence[len(partial_lower):].strip()
                            if remaining:
                                confidence = self._calculate_completion_confidence(
                                    partial_sentence, sentence
                                )
                                completions.append((remaining, confidence))
        
        # Check for phrase completions
        for phrase, phrase_completions in self.phrase_completions.items():
            if partial_lower.endswith(phrase):
                for completion, confidence in phrase_completions:
                    completions.append((completion, confidence * 0.9))
        
        # Grammar-based completions
        words = partial_lower.split()
        if words:
            last_word = words[-1]
            
            # Article + adjective/noun patterns
            if last_word in self.grammar_patterns['articles']:
                for word in self.grammar_patterns['articles'][last_word]:
                    completions.append((word, 0.7))
            
            # Preposition patterns
            elif last_word in self.grammar_patterns['prepositions']:
                for word in self.grammar_patterns['prepositions'][last_word]:
                    completions.append((word, 0.6))
            
            # Conjunction patterns
            elif last_word in self.grammar_patterns['conjunctions']:
                for word in self.grammar_patterns['conjunctions'][last_word]:
                    completions.append((word, 0.5))
        
        # Remove duplicates and sort by confidence
        unique_completions = {}
        for completion, confidence in completions:
            if completion not in unique_completions or confidence > unique_completions[completion]:
                unique_completions[completion] = confidence
        
        sorted_completions = sorted(unique_completions.items(), key=lambda x: x[1], reverse=True)
        return sorted_completions[:5]
    
    def _calculate_completion_confidence(self, partial: str, full_sentence: str) -> float:
        """Calculate confidence score for a completion"""
        partial_len = len(partial)
        full_len = len(full_sentence)
        
        # Base confidence on how much of sentence is already typed
        completion_ratio = partial_len / full_len
        
        # Higher confidence for longer partial matches
        base_confidence = 0.3 + (completion_ratio * 0.7)
        
        # Boost confidence for common patterns
        if any(phrase in partial.lower() for phrase in ['please', 'thank you', 'i would like']):
            base_confidence *= 1.2
        
        # Check against user history for personalization
        similarity_boost = self._get_similarity_boost(full_sentence)
        
        return min(base_confidence + similarity_boost, 1.0)
    
    def _get_similarity_boost(self, sentence: str) -> float:
        """Boost confidence based on similarity to user's previous sentences"""
        if not self.user_sentence_history:
            return 0.0
        
        max_similarity = 0.0
        for historical_sentence in self.user_sentence_history[-10:]:  # Check last 10 sentences
            similarity = difflib.SequenceMatcher(None, sentence.lower(), historical_sentence.lower()).ratio()
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity * 0.3  # Max 30% boost from similarity
    
    def get_next_word_suggestions(self, context: str, num_suggestions: int = 5) -> List[Tuple[str, float]]:
        """Get intelligent next word suggestions based on context"""
        suggestions = []
        context_lower = context.lower()
        
        # Context-aware word suggestions
        if context_lower.endswith('i would'):
            suggestions = [('like', 0.9), ('love', 0.7), ('prefer', 0.6), ('appreciate', 0.5), ('recommend', 0.4)]
        elif context_lower.endswith('thank you'):
            suggestions = [('for', 0.8), ('so', 0.6), ('very', 0.5), (',', 0.3)]
        elif context_lower.endswith('please'):
            suggestions = [('let', 0.7), ('send', 0.6), ('review', 0.5), ('confirm', 0.4), ('consider', 0.3)]
        elif context_lower.endswith('we need'):
            suggestions = [('to', 0.9), ('more', 0.6), ('a', 0.4), ('your', 0.3)]
        elif context_lower.endswith('looking forward'):
            suggestions = [('to', 0.95), ('for', 0.05)]
        else:
            # Grammar-based suggestions
            words = context_lower.split()
            if words:
                last_word = words[-1]
                suggestions = self._get_grammar_suggestions(last_word)
        
        return suggestions[:num_suggestions]
    
    def _get_grammar_suggestions(self, last_word: str) -> List[Tuple[str, float]]:
        """Get grammatically intelligent word suggestions"""
        suggestions = []
        
        # Common word transitions
        transitions = {
            'the': [('best', 0.8), ('most', 0.7), ('next', 0.6), ('first', 0.5), ('main', 0.4)],
            'a': [('good', 0.8), ('new', 0.7), ('great', 0.6), ('quick', 0.5), ('better', 0.4)],
            'and': [('we', 0.7), ('i', 0.6), ('the', 0.5), ('also', 0.4), ('then', 0.3)],
            'with': [('the', 0.6), ('your', 0.5), ('our', 0.4), ('this', 0.3), ('regard', 0.7)],
            'for': [('the', 0.6), ('your', 0.5), ('this', 0.4), ('our', 0.3), ('example', 0.6)]
        }
        
        if last_word in transitions:
            suggestions = transitions[last_word]
        
        return suggestions
    
    def learn_from_completion(self, partial: str, completed: str, was_accepted: bool):
        """Learn from user's completion choices"""
        if was_accepted:
            self.completion_success_rate[completed] += 1
            
            # Add to sentence history for future learning
            if len(completed.split()) > 3:  # Only store meaningful sentences
                self.user_sentence_history.append(completed)
                
                # Keep history manageable
                if len(self.user_sentence_history) > 50:
                    self.user_sentence_history = self.user_sentence_history[-25:]
    
    def get_intelligent_autocorrect(self, word: str) -> List[Tuple[str, float]]:
        """Provide intelligent autocorrect suggestions"""
        corrections = []
        
        # Common typo patterns
        typo_corrections = {
            'teh': ('the', 0.95),
            'adn': ('and', 0.95),
            'taht': ('that', 0.95),
            'thsi': ('this', 0.95),
            'recieve': ('receive', 0.9),
            'seperate': ('separate', 0.9),
            'definitley': ('definitely', 0.9),
            'occured': ('occurred', 0.9)
        }
        
        word_lower = word.lower()
        if word_lower in typo_corrections:
            corrected, confidence = typo_corrections[word_lower]
            corrections.append((corrected, confidence))
        
        # Letter swapping corrections (common typing errors)
        if len(word) > 3:
            for i in range(len(word) - 1):
                # Swap adjacent letters
                swapped = list(word.lower())
                swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
                swapped_word = ''.join(swapped)
                
                if self._is_likely_word(swapped_word):
                    corrections.append((swapped_word, 0.7))
        
        return corrections[:3]
    
    def _is_likely_word(self, word: str) -> bool:
        """Check if a word is likely to be a real word"""
        # Simple heuristic - check if it contains reasonable letter patterns
        if len(word) < 2:
            return False
        
        # Common English letter patterns
        common_patterns = ['th', 'he', 'in', 'er', 'an', 're', 'ed', 'nd', 'ou', 'ti']
        has_common_pattern = any(pattern in word for pattern in common_patterns)
        
        # Avoid too many repeated letters
        has_excessive_repeats = any(word.count(c) > 3 for c in set(word))
        
        return has_common_pattern and not has_excessive_repeats
