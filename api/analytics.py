"""
Typing Analytics Module
"""

import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Any


class TypingAnalytics:
    def __init__(self):
        self.keystrokes = []
        self.session_start = time.time()
        self.total_characters = 0
        self.key_timings = defaultdict(list)
        self.typing_speed_window = deque(maxlen=60)  
        self.key_frequency = defaultdict(int)
        self.error_patterns = defaultdict(int)
        
    def record_keystroke(self, key: str, current_text: str):
        timestamp = time.time()
        
        keystroke_data = {
            'key': key,
            'timestamp': timestamp,
            'text_length': len(current_text),
            'context': current_text[-5:] if len(current_text) >= 5 else current_text
        }
        
        self.keystrokes.append(keystroke_data)
        self.key_frequency[key.lower()] += 1
        
        if len(self.keystrokes) > 1:
            time_diff = timestamp - self.keystrokes[-2]['timestamp']
            self.typing_speed_window.append(time_diff)
        
        if key not in ['BACKSPACE', 'DELETE']:
            self.total_characters += 1
    
    def get_wpm(self) -> float:
        if len(self.typing_speed_window) < 5:
            return 0.0
        
        avg_keystroke_time = sum(self.typing_speed_window) / len(self.typing_speed_window)
        
        if avg_keystroke_time == 0:
            return 0.0
        
        keystrokes_per_minute = 60 / avg_keystroke_time
        words_per_minute = keystrokes_per_minute / 5
        
        return min(words_per_minute, 200)
    
    def get_most_frequent_key(self) -> str:
        if not self.key_frequency:
            return "None"
        
        return max(self.key_frequency.items(), key=lambda x: x[1])[0]
    
    def get_key_frequency_distribution(self) -> Dict[str, float]:
        if not self.key_frequency:
            return {}
        
        total = sum(self.key_frequency.values())
        return {key: count / total for key, count in self.key_frequency.items()}
    
    def get_typing_rhythm(self) -> Dict[str, float]:
        if len(self.keystrokes) < 10:
            return {'consistency': 0.0, 'average_interval': 0.0}
        
        intervals = []
        for i in range(1, len(self.keystrokes)):
            interval = self.keystrokes[i]['timestamp'] - self.keystrokes[i-1]['timestamp']
            if interval < 2.0:
                intervals.append(interval)
        
        if not intervals:
            return {'consistency': 0.0, 'average_interval': 0.0}
        
        avg_interval = sum(intervals) / len(intervals)
        variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
        consistency = max(0, 1 - (variance / avg_interval)) if avg_interval > 0 else 0
        
        return {
            'consistency': consistency,
            'average_interval': avg_interval,
            'variance': variance
        }
    
    def get_session_stats(self) -> Dict[str, Any]:
        session_duration = time.time() - self.session_start
        rhythm = self.get_typing_rhythm()
        
        return {
            'session_duration': session_duration,
            'total_keystrokes': len(self.keystrokes),
            'total_characters': self.total_characters,
            'wpm': self.get_wpm(),
            'most_frequent_key': self.get_most_frequent_key(),
            'typing_consistency': rhythm['consistency'],
            'average_keystroke_interval': rhythm['average_interval'],
            'unique_keys_used': len(self.key_frequency)
        }
    
    def get_current_stats(self) -> Dict[str, Any]:
        return {
            'total_keys': len(self.keystrokes),
            'most_frequent': self.get_most_frequent_key(),
            'wpm': self.get_wpm(),
            'session_time': time.time() - self.session_start
        }
    
    def get_heatmap_data(self) -> Dict[str, float]:
        freq_dist = self.get_key_frequency_distribution()
        
        if not freq_dist:
            return {}
        
        max_freq = max(freq_dist.values())
        return {key: freq / max_freq for key, freq in freq_dist.items()}
    
    def predict_next_key_timing(self, context: str) -> float:
        recent_intervals = list(self.typing_speed_window)[-10:]
        
        if not recent_intervals:
            return 0.1
        
        return sum(recent_intervals) / len(recent_intervals)
    
    def export_data(self) -> Dict[str, Any]:
        return {
            'keystrokes': self.keystrokes[-1000:],  
            'session_stats': self.get_session_stats(),
            'key_frequency': dict(self.key_frequency),
            'export_timestamp': datetime.now().isoformat()
        }
    
    def import_data(self, data: Dict[str, Any]):
        if 'keystrokes' in data:
            self.keystrokes.extend(data['keystrokes'])
        
        if 'key_frequency' in data:
            for key, freq in data['key_frequency'].items():
                self.key_frequency[key] += freq
