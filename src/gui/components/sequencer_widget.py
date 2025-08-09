from PyQt6.QtWidgets import QWidget, QGridLayout, QPushButton, QLabel
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtSignal
from PyQt6.QtGui import QColor

class LEDStepButton(QPushButton):
    """LED-style step button with smooth animations"""

    step_toggled = pyqtSignal(str, int, bool)  # track_name, step_number, is_active

    def __init__(self, step_number, track_name, parent=None):
        super().__init__(parent)
        self.step_number = step_number
        self.track_name = track_name
        self.is_active = False
        self.is_playing = False

        self.setFixedSize(45, 45)
        self.setCheckable(True)
        self.setObjectName("StepButton")

        # Connect click to signal emission
        self.clicked.connect(self.handle_clicked)

        # Pulsing animation for playback position
        self.pulse_animation = QPropertyAnimation(self, b"geometry")
        self.pulse_animation.setDuration(100)
        self.pulse_animation.setEasingCurve(QEasingCurve.Type.OutBounce)

    def handle_clicked(self):
        """Handle button click and emit signal"""
        self.is_active = self.isChecked()
        self.update_style()
        self.step_toggled.emit(self.track_name, self.step_number, self.is_active)

    def set_active(self, active):
        """Set step active state with visual feedback"""
        self.is_active = active
        self.setChecked(active)
        self.update_style()

    def set_playing(self, playing):
        """Highlight step during playback"""
        self.is_playing = playing
        if playing:
            self.start_pulse_animation()
        self.update_style()

    def update_style(self):
        """Update button appearance based on state"""
        if self.is_playing:
            color = "#FFD700"  # Gold for current step
            border_color = "#FFA500"
        elif self.is_active:
            color = "#00FF88"  # Green for active steps
            border_color = "#00D4FF"
        else:
            color = "#2A4A3A"  # Dark green for inactive
            border_color = "#3D5A5A"

        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                border: 2px solid {border_color};
                border-radius: 8px;
                font-weight: bold;
                font-size: 10px;
                color: #000000;
            }}
            QPushButton:hover {{
                border-color: #FFFFFF;
            }}
        """)

    def start_pulse_animation(self):
        """Animate button during playback"""
        current_rect = self.geometry()
        expanded_rect = current_rect.adjusted(-3, -3, 3, 3)

        self.pulse_animation.setStartValue(current_rect)
        self.pulse_animation.setEndValue(expanded_rect)
        self.pulse_animation.finished.connect(lambda: self.pulse_animation.setDirection(self.pulse_animation.Direction.Backward))
        self.pulse_animation.start()

class SequencerWidget(QWidget):
    """16-step sequencer grid with professional appearance"""

    pattern_changed = pyqtSignal(str, list)  # track_name, step_pattern

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tracks = ['KICK', 'SNARE', 'PERC', 'OPEN', 'CLOSED', 'BASS']
        self.steps = {}
        self.current_step = 0
        self.is_playing = False

        self.setup_ui()

        # Playback position timer
        self.position_timer = QTimer()
        self.position_timer.timeout.connect(self.advance_step)

    def setup_ui(self):
        layout = QGridLayout(self)
        layout.setSpacing(4)

        # Add title
        title_label = QLabel("16-STEP SEQUENCER")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #00D4FF; font-weight: bold; font-size: 16px; margin-bottom: 10px;")
        layout.addWidget(title_label, 0, 0, 1, 17)

        # Step numbers header
        for step in range(16):
            step_label = QLabel(str(step + 1))
            step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            step_label.setStyleSheet("color: #A4B8D1; font-size: 10px; font-weight: bold;")
            layout.addWidget(step_label, 1, step + 1)

        # Create step buttons for each track
        for track_idx, track_name in enumerate(self.tracks):
            # Track label
            track_label = QLabel(track_name)
            track_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            track_label.setStyleSheet("color: #00D4FF; font-weight: bold; font-size: 12px;")
            layout.addWidget(track_label, track_idx + 2, 0)

            # Step buttons
            self.steps[track_name] = []
            for step in range(16):
                button = LEDStepButton(step, track_name)
                button.step_toggled.connect(self.handle_step_toggled)
                layout.addWidget(button, track_idx + 2, step + 1)
                self.steps[track_name].append(button)

    def handle_step_toggled(self, track_name, step_number, is_active):
        """Handle step toggle and emit pattern change"""
        pattern = [button.is_active for button in self.steps[track_name]]
        self.pattern_changed.emit(track_name, pattern)

    def toggle_step(self, track, step):
        """Toggle step on/off"""
        button = self.steps[track][step]
        button.set_active(not button.is_active)

    def set_pattern(self, track_name, pattern):
        """Set entire pattern for a track"""
        if track_name in self.steps:
            for step, is_active in enumerate(pattern[:16]):
                self.steps[track_name][step].set_active(is_active)

    def update_playback_position(self, position=None):
        """Update visual playback position"""
        if position is not None:
            self.current_step = position % 16

        # Clear previous position highlighting
        for track_steps in self.steps.values():
            for button in track_steps:
                button.set_playing(False)

        # Highlight current position
        if self.is_playing:
            for track_steps in self.steps.values():
                track_steps[self.current_step].set_playing(True)

    def start_playback(self):
        """Start playback visualization"""
        self.is_playing = True
        self.position_timer.start(125)  # 120 BPM = 125ms per 16th note

    def stop_playback(self):
        """Stop playback visualization"""
        self.is_playing = False
        self.position_timer.stop()
        self.update_playback_position()  # Clear highlighting

    def advance_step(self):
        """Advance to next step"""
        self.current_step = (self.current_step + 1) % 16
        self.update_playback_position()