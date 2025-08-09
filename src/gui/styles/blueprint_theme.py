from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette

class BlueprintTheme:
    def __init__(self):
        self.colors = {
            'bg_primary': '#0B1426',        # Deep blueprint navy
            'bg_secondary': '#1A2332',      # Panel backgrounds  
            'bg_tertiary': '#243447',       # Widget backgrounds
            'bg_quaternary': '#2F3F52',     # Elevated surfaces
            'accent_blue': '#2E86AB',       # Primary accent
            'accent_cyan': '#00D4FF',       # Active elements
            'accent_orange': '#F18F01',     # Warning/record
            'accent_green': '#00FF88',      # Success/active LED
            'accent_yellow': '#FFD700',     # Playing/focus
            'text_primary': '#E8F4FD',      # Main text
            'text_secondary': '#A4B8D1',    # Secondary text
            'text_muted': '#6B7B94',        # Muted text
            'text_dim': '#4A5568',          # Very muted text
            'border_light': '#3D4F66',      # Light borders
            'border_dark': '#1C2631',       # Dark borders
            'border_accent': '#00D4FF',     # Accent borders
            'knob_body': '#34495E',         # Knob base color
            'knob_body_light': '#4A5F7A',   # Knob highlight
            'knob_indicator': '#00D4FF',    # Knob position indicator
            'led_active': '#00FF88',        # Active LED green
            'led_inactive': '#2A4A3A',      # Inactive LED
            'led_playing': '#FFD700',       # Playing indicator
            'fader_track': '#2C3E50',       # Fader background
            'fader_handle': '#3498DB',      # Fader handle
            'meter_green': '#00FF88',       # VU meter green zone
            'meter_yellow': '#FFD700',      # VU meter yellow zone
            'meter_red': '#FF4757',         # VU meter red zone
            'shadow_light': '#1A2332AA',    # Light shadow
            'shadow_dark': '#0B1426DD',     # Dark shadow
            'glass_overlay': '#E8F4FD0A',   # Glass effect overlay
        }

    def get_stylesheet(self):
        return f"""
        /* Main Application Window */
        QMainWindow {{
            background-color: {self.colors['bg_primary']};
            background-image: linear-gradient(135deg, {self.colors['bg_primary']} 0%, {self.colors['bg_secondary']} 100%);
            color: {self.colors['text_primary']};
            font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
            border: 2px solid {self.colors['border_dark']};
        }}

        QWidget {{
            background-color: transparent;
            color: {self.colors['text_primary']};
            selection-background-color: {self.colors['accent_cyan']};
            selection-color: {self.colors['bg_primary']};
        }}

        /* Professional Frames and Panels */
        QFrame {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                      stop:0 {self.colors['bg_secondary']}, 
                                      stop:1 {self.colors['bg_tertiary']});
            border: 2px solid {self.colors['border_light']};
            border-radius: 8px;
            box-shadow: 0 4px 8px {self.colors['shadow_dark']};
        }}

        /* Advanced Knob Styling */
        QDial {{
            background: qradialgradient(cx:0.5, cy:0.5, radius:0.8,
                                      fx:0.3, fy:0.3,
                                      stop:0 {self.colors['knob_body_light']},
                                      stop:0.7 {self.colors['knob_body']},
                                      stop:1 {self.colors['border_dark']});
            border: 3px solid {self.colors['border_light']};
            border-radius: 30px;
            min-width: 60px;
            min-height: 60px;
        }}

        QDial::handle {{
            background-color: {self.colors['knob_indicator']};
            border: 1px solid {self.colors['border_accent']};
            border-radius: 4px;
            width: 8px;
            height: 25px;
        }}

        /* Professional Button System */
        QPushButton {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                       stop:0 {self.colors['bg_quaternary']}, 
                                       stop:1 {self.colors['bg_tertiary']});
            border: 2px solid {self.colors['border_light']};
            border-radius: 6px;
            padding: 10px 18px;
            font-weight: bold;
            font-size: 12px;
            color: {self.colors['text_primary']};
            min-height: 24px;
        }}

        QPushButton:hover {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                       stop:0 {self.colors['accent_blue']}, 
                                       stop:1 {self.colors['accent_cyan']});
            border-color: {self.colors['border_accent']};
            color: {self.colors['text_primary']};
        }}

        QPushButton:pressed {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                       stop:0 {self.colors['accent_cyan']}, 
                                       stop:1 {self.colors['accent_blue']});
            border-color: {self.colors['accent_cyan']};
            color: {self.colors['bg_primary']};
            padding: 11px 17px 9px 19px;
        }}

        QPushButton:disabled {{
            background-color: {self.colors['bg_tertiary']};
            border-color: {self.colors['border_dark']};
            color: {self.colors['text_dim']};
        }}

        /* LED Step Buttons */
        .StepButton {{
            background: qradialgradient(cx:0.5, cy:0.5, radius:0.8,
                                      stop:0 {self.colors['bg_quaternary']},
                                      stop:1 {self.colors['bg_tertiary']});
            border: 2px solid {self.colors['border_dark']};
            border-radius: 10px;
            min-width: 45px;
            min-height: 45px;
            font-size: 11px;
            font-weight: bold;
        }}

        .StepButton:hover {{
            border-color: {self.colors['border_accent']};
            background: qradialgradient(cx:0.5, cy:0.5, radius:0.8,
                                      stop:0 {self.colors['bg_quaternary']},
                                      stop:1 {self.colors['accent_blue']});
        }}

        /* Transport Controls */
        .TransportButton {{
            background: qradialgradient(cx:0.5, cy:0.5, radius:1.0,
                                      stop:0 {self.colors['bg_quaternary']},
                                      stop:1 {self.colors['bg_secondary']});
            border: 3px solid {self.colors['border_light']};
            border-radius: 25px;
            min-width: 70px;
            min-height: 70px;
            font-size: 20px;
            font-weight: bold;
            color: {self.colors['text_primary']};
        }}

        .TransportButton:hover {{
            border-color: {self.colors['border_accent']};
            background: qradialgradient(cx:0.5, cy:0.5, radius:1.0,
                                      stop:0 {self.colors['accent_blue']},
                                      stop:1 {self.colors['bg_secondary']});
        }}

        .RecordButton {{
            background: qradialgradient(cx:0.5, cy:0.5, radius:1.0,
                                      stop:0 {self.colors['accent_orange']},
                                      stop:1 {self.colors['bg_secondary']});
        }}

        .PlayButton {{
            background: qradialgradient(cx:0.5, cy:0.5, radius:1.0,
                                      stop:0 {self.colors['accent_green']},
                                      stop:1 {self.colors['bg_secondary']});
        }}

        /* Professional Slider System */
        QSlider::groove:vertical {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                       stop:0 {self.colors['border_dark']},
                                       stop:0.5 {self.colors['fader_track']},
                                       stop:1 {self.colors['border_dark']});
            width: 24px;
            border-radius: 12px;
            border: 1px solid {self.colors['border_dark']};
        }}

        QSlider::handle:vertical {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                       stop:0 {self.colors['fader_handle']},
                                       stop:1 {self.colors['accent_blue']});
            border: 2px solid {self.colors['border_light']};
            height: 45px;
            border-radius: 12px;
            margin: -4px;
        }}

        QSlider::handle:vertical:hover {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                       stop:0 {self.colors['accent_cyan']},
                                       stop:1 {self.colors['accent_blue']});
            border-color: {self.colors['border_accent']};
        }}

        /* Text Labels */
        QLabel {{
            color: {self.colors['text_primary']};
            background-color: transparent;
        }}

        .TitleLabel {{
            color: {self.colors['accent_cyan']};
            font-size: 18px;
            font-weight: bold;
            text-shadow: 0 0 10px {self.colors['accent_cyan']};
        }}

        .SubtitleLabel {{
            color: {self.colors['text_secondary']};
            font-size: 14px;
            font-weight: bold;
        }}

        .MetricLabel {{
            color: {self.colors['text_muted']};
            font-size: 11px;
            font-family: 'Courier New', monospace;
        }}

        /* Professional Menu Bar */
        QMenuBar {{
            background-color: {self.colors['bg_secondary']};
            border-bottom: 2px solid {self.colors['border_light']};
            color: {self.colors['text_primary']};
            padding: 4px;
        }}

        QMenuBar::item {{
            background-color: transparent;
            padding: 6px 12px;
            border-radius: 4px;
        }}

        QMenuBar::item:selected {{
            background-color: {self.colors['accent_blue']};
        }}

        /* Status Bar */
        QStatusBar {{
            background-color: {self.colors['bg_secondary']};
            border-top: 1px solid {self.colors['border_light']};
            color: {self.colors['text_secondary']};
            font-size: 12px;
        }}

        /* Tool Tips */
        QToolTip {{
            background-color: {self.colors['bg_quaternary']};
            border: 2px solid {self.colors['border_accent']};
            border-radius: 6px;
            padding: 8px;
            color: {self.colors['text_primary']};
            font-size: 12px;
        }}

        /* Scroll Bars */
        QScrollBar:vertical {{
            background: {self.colors['bg_tertiary']};
            width: 16px;
            border-radius: 8px;
        }}

        QScrollBar::handle:vertical {{
            background: {self.colors['accent_blue']};
            border-radius: 8px;
            min-height: 30px;
        }}

        QScrollBar::handle:vertical:hover {{
            background: {self.colors['accent_cyan']};
        }}
        """