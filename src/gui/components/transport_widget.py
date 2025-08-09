from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import Qt, pyqtSignal

class TransportWidget(QWidget):
    """Transport controls for playback, record, stop"""

    play_clicked = pyqtSignal()
    record_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_playing = False
        self.is_recording = False
        self.current_bpm = 120.0
        self.setup_ui()

    def setup_ui(self):
        self.setFixedHeight(80)
        layout = QHBoxLayout(self)

        # Transport buttons
        self.play_button = QPushButton("⏵ PLAY")
        self.play_button.setObjectName("TransportButton")
        self.play_button.clicked.connect(self.handle_play)

        self.record_button = QPushButton("⏺ REC")
        self.record_button.setObjectName("RecordButton")
        self.record_button.clicked.connect(self.handle_record)

        self.stop_button = QPushButton("⏹ STOP")
        self.stop_button.setObjectName("TransportButton")
        self.stop_button.clicked.connect(self.handle_stop)

        # BPM display
        self.bpm_label = QLabel(f"BPM: {self.current_bpm}")
        self.bpm_label.setStyleSheet("color: #00D4FF; font-weight: bold; font-size: 14px;")

        # Pattern display
        self.pattern_label = QLabel("Pattern: A1")
        self.pattern_label.setStyleSheet("color: #A4B8D1; font-size: 12px;")

        # Add to layout
        layout.addWidget(self.play_button)
        layout.addWidget(self.record_button)
        layout.addWidget(self.stop_button)
        layout.addStretch()
        layout.addWidget(self.bpm_label)
        layout.addWidget(self.pattern_label)

    def handle_play(self):
        """Handle play button click"""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_button.setText("⏸ PAUSE")
        else:
            self.play_button.setText("⏵ PLAY")
        self.play_clicked.emit()

    def handle_record(self):
        """Handle record button click"""
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.record_button.setStyleSheet("background-color: #FF4444;")
        else:
            self.record_button.setStyleSheet("")
        self.record_clicked.emit()

    def handle_stop(self):
        """Handle stop button click"""
        self.is_playing = False
        self.is_recording = False
        self.play_button.setText("⏵ PLAY")
        self.record_button.setStyleSheet("")
        self.stop_clicked.emit()

    def update_bpm(self, new_bpm):
        """Update BPM display"""
        self.current_bpm = new_bpm
        self.bpm_label.setText(f"BPM: {new_bpm:.1f}")