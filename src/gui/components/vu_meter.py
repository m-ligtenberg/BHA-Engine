from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtSignal
from PyQt6.QtGui import QPainter, QLinearGradient, QColor, QPen, QBrush, QFont
import math
import random

class VUMeter(QWidget):
    """Professional studio-grade VU meter with peak and RMS display"""

    peak_exceeded = pyqtSignal(float)  # Signal when peak limit is exceeded

    def __init__(self, channel_name="", orientation=Qt.Orientation.Vertical, parent=None):
        super().__init__(parent)
        self.channel_name = channel_name
        self.orientation = orientation
        
        # Audio level parameters
        self.current_level = 0.0    # Current RMS level (0-1)
        self.peak_level = 0.0       # Peak level (0-1) 
        self.peak_hold_level = 0.0  # Peak hold level for visual indication
        self.clip_indicator = False # Red clip indicator
        
        # Visual parameters
        self.segment_count = 20     # Number of LED segments
        self.green_threshold = 0.7  # Green to yellow transition
        self.yellow_threshold = 0.85 # Yellow to red transition
        self.clip_threshold = 0.95  # Clipping threshold
        
        # Animation and timing
        self.peak_hold_timer = QTimer()
        self.peak_hold_timer.setSingleShot(True)
        self.peak_hold_timer.timeout.connect(self.decay_peak_hold)
        
        self.clip_timer = QTimer()
        self.clip_timer.setSingleShot(True)
        self.clip_timer.timeout.connect(self.clear_clip_indicator)
        
        # Smooth level animation
        self.level_animation = QPropertyAnimation(self, b"currentLevel")
        self.level_animation.setDuration(50)  # Fast response for audio
        self.level_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        
        self.setup_ui()

    def setup_ui(self):
        """Setup meter layout and appearance"""
        if self.orientation == Qt.Orientation.Vertical:
            self.setFixedSize(30, 200)
        else:
            self.setFixedSize(200, 30)
            
        # Tool tip with channel info
        self.setToolTip(f"VU Meter - {self.channel_name}\nGreen: -20dB to -6dB\nYellow: -6dB to -3dB\nRed: -3dB to 0dB")

    def set_audio_level(self, rms_level, peak_level=None):
        """Update meter with new audio levels (0.0 to 1.0 range)"""
        # Clamp levels to valid range
        rms_level = max(0.0, min(1.0, rms_level))
        
        if peak_level is not None:
            peak_level = max(0.0, min(1.0, peak_level))
            self.peak_level = peak_level
            
            # Update peak hold if new peak is higher
            if peak_level > self.peak_hold_level:
                self.peak_hold_level = peak_level
                self.peak_hold_timer.start(2000)  # Hold peak for 2 seconds
                
            # Check for clipping
            if peak_level >= self.clip_threshold:
                self.clip_indicator = True
                self.clip_timer.start(3000)  # Show clip for 3 seconds
                self.peak_exceeded.emit(peak_level)
        
        # Animate RMS level change
        self.level_animation.setStartValue(self.current_level)
        self.level_animation.setEndValue(rms_level)
        self.level_animation.start()
        
        self.update()  # Trigger repaint

    def decay_peak_hold(self):
        """Decay peak hold level gradually"""
        self.peak_hold_level = max(0.0, self.peak_hold_level - 0.02)
        self.update()
        
        if self.peak_hold_level > 0.0:
            self.peak_hold_timer.start(100)  # Continue decay

    def clear_clip_indicator(self):
        """Clear the clip indicator"""
        self.clip_indicator = False
        self.update()

    def paintEvent(self, event):
        """Custom paint event for professional VU meter rendering"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = self.rect().adjusted(2, 2, -2, -2)
        
        if self.orientation == Qt.Orientation.Vertical:
            self.paint_vertical_meter(painter, rect)
        else:
            self.paint_horizontal_meter(painter, rect)

    def paint_vertical_meter(self, painter, rect):
        """Paint vertical VU meter"""
        segment_height = rect.height() / self.segment_count
        segment_width = rect.width() - 4
        
        for i in range(self.segment_count):
            # Calculate segment position (bottom to top)
            segment_rect_y = rect.bottom() - (i + 1) * segment_height
            segment_level = (i + 1) / self.segment_count
            
            # Determine segment color based on level
            if segment_level <= self.green_threshold:
                color = QColor("#00FF88")  # Green zone
            elif segment_level <= self.yellow_threshold:
                color = QColor("#FFD700")  # Yellow zone  
            else:
                color = QColor("#FF4757")  # Red zone
                
            # Determine if segment should be lit
            is_lit = segment_level <= self.current_level
            is_peak_hold = abs(segment_level - self.peak_hold_level) < 0.05
            
            # Adjust color brightness based on state
            if is_lit or is_peak_hold:
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(color.lighter(120), 1))
            else:
                # Unlit segment - darker version
                dark_color = QColor(color.red() // 4, color.green() // 4, color.blue() // 4)
                painter.setBrush(QBrush(dark_color))
                painter.setPen(QPen(dark_color, 1))
            
            # Draw segment with small gap
            segment_rect = rect.adjusted(2, 0, -2, 0)
            segment_rect.setTop(int(segment_rect_y))
            segment_rect.setHeight(int(segment_height - 1))
            
            painter.drawRect(segment_rect)
            
        # Draw clip indicator
        if self.clip_indicator:
            clip_rect = rect.adjusted(0, 0, 0, -rect.height() + 15)
            painter.setBrush(QBrush(QColor("#FF0000")))
            painter.setPen(QPen(QColor("#FFFFFF"), 2))
            painter.drawRect(clip_rect)
            
            # "CLIP" text
            painter.setPen(QPen(QColor("#FFFFFF")))
            painter.setFont(QFont("Arial", 8, QFont.Weight.Bold))
            painter.drawText(clip_rect, Qt.AlignmentFlag.AlignCenter, "CLIP")

    def paint_horizontal_meter(self, painter, rect):
        """Paint horizontal VU meter"""
        segment_width = rect.width() / self.segment_count
        segment_height = rect.height() - 4
        
        for i in range(self.segment_count):
            # Calculate segment position (left to right)
            segment_rect_x = rect.left() + i * segment_width
            segment_level = (i + 1) / self.segment_count
            
            # Determine segment color based on level
            if segment_level <= self.green_threshold:
                color = QColor("#00FF88")  # Green zone
            elif segment_level <= self.yellow_threshold:
                color = QColor("#FFD700")  # Yellow zone
            else:
                color = QColor("#FF4757")  # Red zone
                
            # Determine if segment should be lit
            is_lit = segment_level <= self.current_level
            is_peak_hold = abs(segment_level - self.peak_hold_level) < 0.05
            
            # Adjust color brightness based on state
            if is_lit or is_peak_hold:
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(color.lighter(120), 1))
            else:
                # Unlit segment - darker version
                dark_color = QColor(color.red() // 4, color.green() // 4, color.blue() // 4)
                painter.setBrush(QBrush(dark_color))
                painter.setPen(QPen(dark_color, 1))
            
            # Draw segment with small gap
            segment_rect = rect.adjusted(0, 2, 0, -2)
            segment_rect.setLeft(int(segment_rect_x))
            segment_rect.setWidth(int(segment_width - 1))
            
            painter.drawRect(segment_rect)

class StereoVUMeter(QWidget):
    """Stereo VU meter with left and right channels"""

    def __init__(self, channel_name="Stereo", parent=None):
        super().__init__(parent)
        self.channel_name = channel_name
        
        self.setup_ui()

    def setup_ui(self):
        """Setup stereo meter layout"""
        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        
        # Channel label
        label = QLabel(self.channel_name)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: #00D4FF; font-weight: bold; font-size: 12px;")
        layout.addWidget(label)
        
        # Meter container
        meter_container = QHBoxLayout()
        meter_container.setSpacing(4)
        
        # Left channel meter
        self.left_meter = VUMeter("L", Qt.Orientation.Vertical)
        meter_container.addWidget(self.left_meter)
        
        # Right channel meter  
        self.right_meter = VUMeter("R", Qt.Orientation.Vertical)
        meter_container.addWidget(self.right_meter)
        
        layout.addLayout(meter_container)
        
        # Peak value display
        self.peak_label = QLabel("Peak: -∞ dB")
        self.peak_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.peak_label.setStyleSheet("color: #A4B8D1; font-size: 10px; font-family: 'Courier New';")
        layout.addWidget(self.peak_label)

    def set_stereo_levels(self, left_rms, right_rms, left_peak=None, right_peak=None):
        """Update both channel levels"""
        self.left_meter.set_audio_level(left_rms, left_peak)
        self.right_meter.set_audio_level(right_rms, right_peak)
        
        # Update peak display
        if left_peak is not None and right_peak is not None:
            max_peak = max(left_peak, right_peak)
            if max_peak > 0.0:
                # Convert linear to dB
                peak_db = 20 * math.log10(max_peak)
                self.peak_label.setText(f"Peak: {peak_db:+.1f} dB")
            else:
                self.peak_label.setText("Peak: -∞ dB")

class SpectrumAnalyzer(QWidget):
    """Real-time spectrum analyzer for frequency visualization"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frequency_data = [0.0] * 32  # 32 frequency bands
        self.setFixedSize(300, 150)
        
        # Smooth animation for frequency data
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(50)  # 20 FPS

    def set_frequency_data(self, frequency_bands):
        """Update frequency spectrum data (32 bands, 0.0 to 1.0)"""
        self.frequency_data = frequency_bands[:32]
        
    def update_display(self):
        """Smooth decay of frequency bands for natural look"""
        for i in range(len(self.frequency_data)):
            # Smooth decay
            self.frequency_data[i] *= 0.95
        self.update()

    def paintEvent(self, event):
        """Paint spectrum analyzer bars"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = self.rect().adjusted(4, 4, -4, -4)
        bar_width = rect.width() / len(self.frequency_data)
        
        for i, level in enumerate(self.frequency_data):
            # Calculate bar height
            bar_height = level * rect.height()
            bar_rect_x = rect.left() + i * bar_width
            bar_rect_y = rect.bottom() - bar_height
            
            # Color based on frequency (bass=red, mid=green, high=blue)
            if i < 10:  # Bass
                color = QColor("#FF4757")
            elif i < 22:  # Mids
                color = QColor("#00FF88") 
            else:  # Highs
                color = QColor("#00D4FF")
                
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color.lighter(120), 1))
            
            bar_rect = rect.adjusted(0, 0, 0, 0)
            bar_rect.setLeft(int(bar_rect_x))
            bar_rect.setWidth(int(bar_width - 1))
            bar_rect.setTop(int(bar_rect_y))
            
            painter.drawRect(bar_rect)