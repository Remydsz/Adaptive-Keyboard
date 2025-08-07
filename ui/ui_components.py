"""
UI Components - Custom Kivy widgets for the adaptive keyboard
"""

from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.widget import Widget
from kivy.animation import Animation
from kivy.graphics import Color, Rectangle, RoundedRectangle
from kivy.metrics import dp
from kivy.clock import Clock
import colorsys


class AdaptiveKey(Button):
    def __init__(self, char, **kwargs):
        super().__init__(**kwargs)
        self.char = char
        self.base_text = char.upper() if char not in [' ', 'SPACE'] else 'SPACE'
        self.text = self.base_text  
    
        if char in ['SPACE']:
            self.base_size = (dp(200), dp(40))
        elif char in ['BACKSPACE']:
            self.base_size = (dp(80), dp(40))
        else:
            self.base_size = (dp(40), dp(40))
            
        self.prediction_score = 0.0
        self.heat_value = 0.0 
        self.font_size = dp(12)  
        
        self.size_hint = (None, None)
        self.size = self.base_size
        self.background_normal = ''
        self.background_down = ''
        
        self.background_color = (0.9, 0.9, 0.9, 1.0)  
        self.color = (0, 0, 0, 1)  
        
        self.bind(on_press=self.on_key_press)
    
    def on_key_press(self, instance):
        original_color = self.background_color if hasattr(self, 'background_color') else [1, 1, 1, 1]
        
        parent = self.parent
        while parent and not hasattr(parent, 'on_key_press_callback'):
            parent = parent.parent
        
        if parent and hasattr(parent, 'on_key_press_callback'):
            parent.on_key_press_callback(self.char)
    
    def update_prediction(self, score: float):
        self.prediction_score = score
        self.update_appearance()
        
        if score > 0.05:  
            probability_percent = int(score * 100)
            self.text = f"{self.base_text}\n{probability_percent}%"
        else:
            self.text = self.base_text  
    
    def update_heat(self, heat: float):
        self.heat_value = heat
        self.update_appearance()
    
    def update_appearance(self):
        hue = 0.6 - (self.heat_value * 0.4)  
        saturation = 0.3 + (self.prediction_score * 0.7)  
        value = 0.7 + (self.prediction_score * 0.3)  
        
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        
        self.background_color = (*rgb, 0.9)
        self.color = (0, 0, 0, 1)  
        
        self.font_size = dp(14) + (self.prediction_score * dp(6))


class KeyboardLayout(GridLayout):
    def __init__(self, engine, analytics, on_key_press, **kwargs):
        super().__init__(**kwargs)
        self.engine = engine
        self.analytics = analytics
        self.on_key_press_callback = on_key_press
        
        self.cols = 1  
        self.spacing = dp(5)
        self.size_hint = (1, 0.6)
        
        self.layout = [
            ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
            ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
            ['z', 'x', 'c', 'v', 'b', 'n', 'm', 'BACKSPACE'],
            ['SPACE']
        ]
        
        self.keys = {}
        self.create_keyboard()
        
        Clock.schedule_interval(self.update_heat_visualization, 2.0)
    
    def create_keyboard(self):
        for row_chars in self.layout:
            row_layout = GridLayout(
                cols=len(row_chars),
                rows=1,
                spacing=dp(3),
                size_hint_y=None,
                height=dp(50)
            )
            
            for char in row_chars:
                key = AdaptiveKey(char)
                if char not in self.keys:
                    self.keys[char] = key
                row_layout.add_widget(key)
            
            self.add_widget(row_layout)
    
    def adapt_to_predictions(self, predictions):
        for key in self.keys.values():
            key.update_prediction(0.0)
        
        for char, score in predictions.items():
            if char in self.keys:
                self.keys[char].update_prediction(score)
    
    def update_heat_visualization(self, dt):
        for char, key in self.keys.items():
            heat = self.engine.get_key_heat(char)
            key.update_heat(heat)


class StatsDisplay(Widget):
    def __init__(self, analytics, **kwargs):
        super().__init__(**kwargs)
        self.analytics = analytics
        
        with self.canvas:
            Color(0.2, 0.2, 0.2, 0.8)
            self.bg_rect = Rectangle(pos=self.pos, size=self.size)
        
        self.bind(pos=self.update_graphics, size=self.update_graphics)
    
    def update_graphics(self, *args):
        self.bg_rect.pos = self.pos
        self.bg_rect.size = self.size
