from PyQt6.QtWidgets import QDial, QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPainter, QRadialGradient, QColor, QPen
import math

class RhythmWolfKnob(QDial):
    """Custom knob widget that replicates Rhythm Wolf hardware knobs"""

    value_changed = pyqtSignal(str, int)  # knob_name, value

    def __init__(self, label="", cc_number=None, parent=None):
        super().__init__(parent)
        self.label_text = label
        self.cc_number = cc_number
        self.knob_name = label

        # Knob appearance settings
        self.setRange(0, 127)
        self.setValue(64)  # Center position
        self.setNotchesVisible(True)
        self.setWrapping(False)

        # Animation for smooth hardware sync
        self.sync_animation = QPropertyAnimation(self, b"value")
        self.sync_animation.setDuration(150)
        self.sync_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        # Connect value changes to signal emission
        self.valueChanged.connect(self.emit_value_changed)

    def emit_value_changed(self, value):
        """Emit custom signal with knob name"""
        self.value_changed.emit(self.knob_name, value)

    def sync_to_hardware_value(self, hw_value):
        """Smoothly animate knob to match hardware position"""
        self.sync_animation.setStartValue(self.value())
        self.sync_animation.setEndValue(hw_value)
        self.sync_animation.start()

    def paintEvent(self, event):
        """Custom paint for realistic knob appearance"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw knob body with gradient
        rect = self.rect().adjusted(5, 5, -5, -5)
        gradient = QRadialGradient(rect.center(), rect.width() / 2)
        gradient.setColorAt(0, QColor("#4A5F7A"))
        gradient.setColorAt(0.7, QColor("#34495E"))
        gradient.setColorAt(1, QColor("#2C3E50"))

        painter.setBrush(gradient)
        painter.setPen(QPen(QColor("#1C2631"), 2))
        painter.drawEllipse(rect)

        # Draw position indicator
        angle = (self.value() - self.minimum()) / (self.maximum() - self.minimum()) * 300 - 150
        indicator_length = rect.width() * 0.35
        center = rect.center()

        end_x = center.x() + indicator_length * math.cos(math.radians(angle))
        end_y = center.y() + indicator_length * math.sin(math.radians(angle))

        painter.setPen(QPen(QColor("#00D4FF"), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(center, int(end_x), int(end_y))

class KnobGroup(QWidget):
    """Group of knobs representing one Rhythm Wolf voice section"""

    def __init__(self, voice_name, knob_configs, parent=None):
        super().__init__(parent)
        self.voice_name = voice_name
        self.knobs = {}

        layout = QVBoxLayout(self)

        # Voice label
        label = QLabel(voice_name.upper())
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-weight: bold; color: #00D4FF; font-size: 14px;")
        layout.addWidget(label)

        # Create knobs based on configuration
        for knob_config in knob_configs:
            knob = RhythmWolfKnob(
                label=knob_config['name'],
                cc_number=knob_config['cc']
            )
            self.knobs[knob_config['name']] = knob
            layout.addWidget(knob)

    def get_knob_values(self):
        """Get current values of all knobs in this group"""
        return {name: knob.value() for name, knob in self.knobs.items()}

    def set_knob_value(self, knob_name, value):
        """Set knob value with animation"""
        if knob_name in self.knobs:
            self.knobs[knob_name].sync_to_hardware_value(value)