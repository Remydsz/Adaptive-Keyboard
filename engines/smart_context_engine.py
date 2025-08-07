"""
Smart Context Engine - Advanced semantic understanding for predictions
"""

import re
import time
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
from datetime import datetime, timedelta


class SmartContextEngine:
    """Advanced context-aware prediction engine with semantic understanding"""
    
    def __init__(self):
        self.semantic_patterns = self._build_semantic_patterns()
        self.domain_contexts = self._build_domain_contexts()
        self.temporal_patterns = defaultdict(list)
        self.user_writing_style = self._initialize_writing_profile()
        self.conversation_memory = []
        self.error_patterns = defaultdict(Counter)
        
    def _build_semantic_patterns(self) -> Dict[str, List[str]]:
        """Build semantic relationship patterns for intelligent completion"""
        return {
            'action_objects': {
                'send': ['email', 'message', 'report', 'file', 'document', 'invitation'],
                'create': ['document', 'file', 'report', 'presentation', 'project'],
                'review': ['document', 'code', 'report', 'proposal', 'changes'],
                'schedule': ['meeting', 'call', 'appointment', 'interview', 'demo'],
                'analyze': ['data', 'results', 'performance', 'metrics', 'trends'],
                'implement': ['feature', 'solution', 'changes', 'improvement', 'fix'],
                'test': ['code', 'feature', 'system', 'functionality', 'performance']
            },
            'question_patterns': {
                'how': ['can', 'do', 'to', 'should', 'would', 'much', 'many'],
                'what': ['is', 'are', 'do', 'time', 'about', 'kind'],
                'when': ['will', 'can', 'should', 'do', 'is', 'are'],
                'where': ['is', 'are', 'can', 'should', 'do'],
                'why': ['is', 'are', 'do', 'should', 'would'],
                'which': ['one', 'way', 'option', 'approach', 'method']
            },
            'completion_phrases': {
                'thank you': ['for', 'so', 'very'],
                'i would': ['like', 'love', 'prefer', 'appreciate'],
                'please let': ['me', 'us', 'them'],
                'looking forward': ['to', 'for'],
                'in order': ['to', 'for'],
                'as soon as': ['possible', 'you'],
                'on the other': ['hand', 'side'],
                'in addition': ['to', 'we']
            }
        }
    
    def _build_domain_contexts(self) -> Dict[str, Dict]:
        """Build domain-specific vocabulary and patterns"""
        return {
            'email': {
                'starters': ['hi', 'hello', 'dear', 'greetings'],
                'closings': ['best', 'regards', 'sincerely', 'thanks'],
                'transitions': ['however', 'furthermore', 'additionally', 'meanwhile'],
                'vocab_boost': ['meeting', 'schedule', 'deadline', 'project', 'update']
            },
            'technical': {
                'starters': ['the', 'we', 'this', 'our'],
                'vocab_boost': ['function', 'method', 'class', 'variable', 'implementation', 
                               'algorithm', 'data', 'system', 'performance', 'error'],
                'patterns': ['implement', 'optimize', 'debug', 'test', 'deploy']
            },
            'casual': {
                'starters': ['hey', 'hi', 'so', 'well'],
                'vocab_boost': ['cool', 'awesome', 'great', 'nice', 'fun', 'good'],
                'patterns': ['gonna', 'wanna', 'kinda', 'sorta']
            },
            'formal': {
                'starters': ['we', 'i', 'the', 'this'],
                'vocab_boost': ['proposal', 'recommendation', 'analysis', 'evaluation',
                               'consideration', 'implementation', 'furthermore', 'however'],
                'patterns': ['would like to', 'please consider', 'we recommend']
            }
        }
    
    def _initialize_writing_profile(self) -> Dict:
        """Initialize user writing style profile"""
        return {
            'avg_sentence_length': 0,
            'complexity_preference': 0.5,  # 0=simple, 1=complex
            'formality_level': 0.5,        # 0=casual, 1=formal
            'domain_preferences': Counter(),
            'common_phrases': Counter(),
            'typing_speed': 0,
            'error_frequency': 0
        }
    
    def detect_domain(self, text: str) -> str:
        """Intelligently detect the writing domain/context"""
        text_lower = text.lower()
        
        # Domain indicators
        email_indicators = ['dear', 'regards', 'sincerely', 'meeting', 'schedule', '@']
        tech_indicators = ['function', 'class', 'method', 'algorithm', 'code', 'system', 'debug']
        formal_indicators = ['furthermore', 'however', 'consequently', 'recommendation']
        casual_indicators = ['hey', 'gonna', 'wanna', 'cool', 'awesome']
        
        scores = {
            'email': sum(1 for indicator in email_indicators if indicator in text_lower),
            'technical': sum(1 for indicator in tech_indicators if indicator in text_lower),
            'formal': sum(1 for indicator in formal_indicators if indicator in text_lower),
            'casual': sum(1 for indicator in casual_indicators if indicator in text_lower)
        }
        
        return max(scores, key=scores.get) if max(scores.values()) > 0 else 'general'
    
    def get_smart_predictions(self, context: str, partial_word: str, base_predictions: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Enhance predictions with semantic intelligence"""
        if not context and not partial_word:
            return base_predictions
        
        # Detect domain and adapt predictions
        domain = self.detect_domain(context)
        enhanced_predictions = []
        
        for word, base_score in base_predictions:
            enhanced_score = base_score
            
            # Domain-specific boosting
            if domain in self.domain_contexts:
                domain_data = self.domain_contexts[domain]
                if word in domain_data.get('vocab_boost', []):
                    enhanced_score *= 1.5
            
            # Semantic pattern boosting
            enhanced_score = self._apply_semantic_boost(context, word, enhanced_score)
            
            # Temporal pattern boosting (time-aware predictions)
            enhanced_score = self._apply_temporal_boost(word, enhanced_score)
            
            # User style adaptation
            enhanced_score = self._apply_style_boost(word, enhanced_score)
            
            enhanced_predictions.append((word, enhanced_score))
        
        # Add intelligent completions based on context
        smart_completions = self._generate_smart_completions(context, partial_word)
        enhanced_predictions.extend(smart_completions)
        
        # Sort and return top predictions
        enhanced_predictions.sort(key=lambda x: x[1], reverse=True)
        return enhanced_predictions[:10]
    
    def _apply_semantic_boost(self, context: str, word: str, score: float) -> float:
        """Apply semantic understanding boost"""
        context_words = context.lower().split()
        
        if len(context_words) > 0:
            last_word = context_words[-1]
            
            # Action-object relationships
            if last_word in self.semantic_patterns['action_objects']:
                if word in self.semantic_patterns['action_objects'][last_word]:
                    score *= 2.0
            
            # Question pattern completion
            if last_word in self.semantic_patterns['question_patterns']:
                if word in self.semantic_patterns['question_patterns'][last_word]:
                    score *= 1.8
        
        # Phrase completion patterns
        for phrase, completions in self.semantic_patterns['completion_phrases'].items():
            if context.lower().endswith(phrase):
                if word in completions:
                    score *= 2.5
        
        return score
    
    def _apply_temporal_boost(self, word: str, score: float) -> float:
        """Apply time-based pattern learning"""
        current_hour = datetime.now().hour
        
        # Time-based vocabulary preferences
        if 6 <= current_hour <= 10:  # Morning
            morning_words = ['good', 'morning', 'coffee', 'start', 'begin', 'early']
            if word in morning_words:
                score *= 1.3
        elif 17 <= current_hour <= 21:  # Evening
            evening_words = ['dinner', 'home', 'tired', 'finished', 'done', 'evening']
            if word in evening_words:
                score *= 1.3
        
        return score
    
    def _apply_style_boost(self, word: str, score: float) -> float:
        """Apply user writing style adaptation"""
        # Complexity preference
        word_complexity = len(word) / 10.0  # Simple metric
        complexity_match = 1 - abs(self.user_writing_style['complexity_preference'] - word_complexity)
        score *= (1 + complexity_match * 0.3)
        
        # Formality level adaptation
        formal_words = ['furthermore', 'however', 'consequently', 'recommendation', 'consideration']
        casual_words = ['cool', 'awesome', 'great', 'gonna', 'wanna']
        
        if word in formal_words and self.user_writing_style['formality_level'] > 0.6:
            score *= 1.4
        elif word in casual_words and self.user_writing_style['formality_level'] < 0.4:
            score *= 1.4
        
        return score
    
    def _generate_smart_completions(self, context: str, partial_word: str) -> List[Tuple[str, float]]:
        """Generate intelligent completions based on deep context understanding"""
        completions = []
        
        # Sentence structure prediction
        if context.count('.') > 0:  # Not first sentence
            sentence_starters = ['the', 'this', 'we', 'i', 'it', 'they']
            for starter in sentence_starters:
                if starter.startswith(partial_word.lower()):
                    completions.append((starter, 0.6))
        
        # Smart phrase completions
        context_lower = context.lower()
        if context_lower.endswith('thank you'):
            completions.extend([('for', 0.8), ('so', 0.7), ('very', 0.6)])
        elif context_lower.endswith('i would'):
            completions.extend([('like', 0.9), ('love', 0.7), ('prefer', 0.6)])
        
        return completions
    
    def learn_from_user(self, text: str, typing_speed: float = 0):
        """Learn and adapt from user's actual typing"""
        # Update writing style profile
        sentences = text.split('.')
        if sentences:
            avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
            self.user_writing_style['avg_sentence_length'] = (
                self.user_writing_style['avg_sentence_length'] * 0.9 + avg_length * 0.1
            )
        
        # Track domain preferences
        domain = self.detect_domain(text)
        self.user_writing_style['domain_preferences'][domain] += 1
        
        # Learn common phrases
        words = text.lower().split()
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            self.user_writing_style['common_phrases'][bigram] += 1
        
        # Update complexity preference based on word choices
        complex_words = [w for w in words if len(w) > 6]
        complexity_ratio = len(complex_words) / max(len(words), 1)
        self.user_writing_style['complexity_preference'] = (
            self.user_writing_style['complexity_preference'] * 0.9 + complexity_ratio * 0.1
        )
        
        # Store temporal patterns
        current_time = datetime.now()
        self.temporal_patterns[current_time.hour].append({
            'text': text,
            'domain': domain,
            'timestamp': current_time
        })
    
    def get_intelligent_suggestions(self, context: str, num_suggestions: int = 3) -> List[str]:
        """Get intelligent multi-word suggestions based on deep context analysis"""
        domain = self.detect_domain(context)
        suggestions = []
        
        # Domain-specific intelligent suggestions
        if domain == 'email':
            if any(greeting in context.lower() for greeting in ['hi', 'hello', 'dear']):
                suggestions.extend(['i hope this email finds you well', 'thank you for your time'])
            elif 'meeting' in context.lower():
                suggestions.extend(['let me know if this works', 'looking forward to hearing from you'])
        
        elif domain == 'technical':
            if any(word in context.lower() for word in ['implement', 'code', 'function']):
                suggestions.extend(['this should improve performance', 'we need to test this thoroughly'])
        
        # Context-aware sentence completions
        context_words = context.lower().split()
        if len(context_words) > 2:
            last_words = ' '.join(context_words[-2:])
            if last_words in ['would like', 'want to', 'need to']:
                suggestions.extend(['schedule a meeting', 'discuss this further', 'get your feedback'])
        
        return suggestions[:num_suggestions]
