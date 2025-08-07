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
        self.text = self.base_text  # Start with just the character
        
        # Set different base sizes for different key types
        if char in ['SPACE']:
            self.base_size = (dp(200), dp(40))  # Wide space bar
        elif char in ['BACKSPACE']:
            self.base_size = (dp(80), dp(40))   # Wide backspace
        else:
            self.base_size = (dp(40), dp(40))   # Regular keys
            
        self.prediction_score = 0.0
        self.heat_value = 0.0  # Start completely neutral
        self.font_size = dp(12)  # Smaller font to fit probability
        
        # Set initial appearance - neutral like standard mobile keyboards
        self.size_hint = (None, None)
        self.size = self.base_size
        self.background_normal = ''
        self.background_down = ''
        
        # Set neutral appearance initially
        self.background_color = (0.9, 0.9, 0.9, 1.0)  # Light gray like iOS/Android
        self.color = (0, 0, 0, 1)  # Black text
        
        # Bind events
        self.bind(on_press=self.on_key_press)
    
    def on_key_press(self, instance):
        """Handle key press with visual feedback"""
        # Simple press feedback without size animation that breaks things
        original_color = self.background_color if hasattr(self, 'background_color') else [1, 1, 1, 1]
        
        # Call the callback to handle the key press
        # Walk up the widget tree to find the keyboard layout
        parent = self.parent
        while parent and not hasattr(parent, 'on_key_press_callback'):
            parent = parent.parent
        
        if parent and hasattr(parent, 'on_key_press_callback'):
            parent.on_key_press_callback(self.char)
    
    def update_prediction(self, score: float):
        """Update prediction score and visual appearance"""
        self.prediction_score = score
        self.update_appearance()
        
        # Display probability as percentage under the key label
        if score > 0.05:  # Only show if probability is significant
            probability_percent = int(score * 100)
            self.text = f"{self.base_text}\n{probability_percent}%"
        else:
            self.text = self.base_text  # Just show the character
    
    def update_heat(self, heat: float):
        """Update heat value and visual appearance"""
        self.heat_value = heat
        self.update_appearance()
    
    def update_appearance(self):
        """Update the visual appearance based on prediction and heat"""
        # Calculate color based on heat and prediction
        hue = 0.6 - (self.heat_value * 0.4)  # Blue to red gradient
        saturation = 0.3 + (self.prediction_score * 0.7)  # More saturated for predictions
        value = 0.7 + (self.prediction_score * 0.3)  # Brighter for predictions
        
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # Set button colors directly
        self.background_color = (*rgb, 0.9)
        self.color = (0, 0, 0, 1)  # Black text
        
        # Update font size based on prediction
        self.font_size = dp(14) + (self.prediction_score * dp(6))


class KeyboardLayout(GridLayout):
    def __init__(self, engine, analytics, on_key_press, **kwargs):
        super().__init__(**kwargs)
        self.engine = engine
        self.analytics = analytics
        self.on_key_press_callback = on_key_press
        
        self.cols = 1  # Single column to stack rows
        self.spacing = dp(5)
        self.size_hint = (1, 0.6)
        
        # Proper QWERTY layout with realistic proportions
        self.layout = [
            ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
            ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
            ['z', 'x', 'c', 'v', 'b', 'n', 'm', 'BACKSPACE'],
            ['SPACE']
        ]
        
        self.keys = {}
        self.create_keyboard()
        
        # Schedule regular updates
        Clock.schedule_interval(self.update_heat_visualization, 2.0)
    
    def create_keyboard(self):
        """Create the keyboard layout with adaptive keys"""
        for row_chars in self.layout:
            # Create a horizontal layout for each row
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
        """Adapt keyboard appearance based on predictions"""
        # Reset all keys to base prediction
        for key in self.keys.values():
            key.update_prediction(0.0)
        
        # Update keys with predictions
        for char, score in predictions.items():
            if char in self.keys:
                self.keys[char].update_prediction(score)
    
    def update_heat_visualization(self, dt):
        """Update heat visualization based on usage frequency"""
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
