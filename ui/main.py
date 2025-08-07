"""
Adaptive Keyboard - Main Application Entry Point
A mobile keyboard that learns and adapts to user typing patterns
"""

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.animation import Animation
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.graphics import Color, Rectangle
from kivy.uix.widget import Widget

import json
import os
from datetime import datetime
from collections import defaultdict, Counter

from keyboard_engine import AdaptiveKeyboardEngine
from analytics import TypingAnalytics
from ui_components import AdaptiveKey, KeyboardLayout


class AdaptiveKeyboardApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.engine = AdaptiveKeyboardEngine()
        self.analytics = TypingAnalytics()
        self.current_text = ""
        
    def build(self):
        """Build the main application interface"""
        self.title = "Adaptive Keyboard"
        
        # Main layout
        main_layout = BoxLayout(orientation='vertical', padding=dp(10), spacing=dp(10))
        
        # Text display area
        self.text_display = TextInput(
            text="Start typing to see the keyboard adapt...",
            multiline=True,
            size_hint=(1, 0.3),
            font_size=dp(16)
        )
        main_layout.add_widget(self.text_display)
        
        # Analytics display
        self.analytics_label = Label(
            text="Analytics: No data yet",
            size_hint=(1, 0.1),
            font_size=dp(12)
        )
        main_layout.add_widget(self.analytics_label)
        
        # Adaptive keyboard
        self.keyboard_widget = KeyboardLayout(
            engine=self.engine,
            analytics=self.analytics,
            on_key_press=self.on_key_press
        )
        main_layout.add_widget(self.keyboard_widget)
        
        # Start analytics update timer
        Clock.schedule_interval(self.update_analytics_display, 1.0)
        
        return main_layout
    
    def on_key_press(self, key_char):
        """Handle key press events"""
        if key_char == 'BACKSPACE':
            if self.current_text:
                self.current_text = self.current_text[:-1]
        elif key_char == 'SPACE':
            self.current_text += ' '
        else:
            self.current_text += key_char
            
        # Update text display
        self.text_display.text = self.current_text
        
        # Record keystroke for analytics
        self.analytics.record_keystroke(key_char, self.current_text)
        
        # Update keyboard predictions
        self.engine.update_predictions(self.current_text)
        
        # Trigger keyboard adaptation
        self.keyboard_widget.adapt_to_predictions(self.engine.get_predictions())
    
    def update_analytics_display(self, dt):
        """Update the analytics display"""
        stats = self.analytics.get_current_stats()
        self.analytics_label.text = f"Keys pressed: {stats['total_keys']} | Most frequent: {stats['most_frequent']} | WPM: {stats['wpm']:.1f}"


if __name__ == '__main__':
    AdaptiveKeyboardApp().run()
