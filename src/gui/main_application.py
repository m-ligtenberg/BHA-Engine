import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon

from gui.components.rhythm_wolf_widget import RhythmWolfWidget
from gui.components.sequencer_widget import SequencerWidget  
from gui.components.mixer_widget import MixerWidget
from gui.components.transport_widget import TransportWidget
from gui.styles.blueprint_theme import BlueprintTheme
from core.daw_engine import DAWEngine
from midi.auto_record import RhythmWolfAutoRecord
from midi.bpm_sync import RealTimeBPMSync

class RhythmWolfDAW(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎛️ Rhythm Wolf Mini-DAW")
        self.setMinimumSize(1400, 900)

        # Initialize DAW engine
        self.daw_engine = DAWEngine()

        # Initialize MIDI systems
        self.auto_record = RhythmWolfAutoRecord(self.daw_engine)
        self.bpm_sync = RealTimeBPMSync(self.daw_engine)

        # Apply blueprint theme
        self.theme = BlueprintTheme()
        self.setStyleSheet(self.theme.get_stylesheet())

        # Create central widget with layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create main components
        self.rhythm_wolf_panel = RhythmWolfWidget(self)
        self.sequencer_grid = SequencerWidget(self)
        self.mixer_console = MixerWidget(self)
        self.transport_controls = TransportWidget(self)

        # Add components to layout
        layout.addWidget(self.rhythm_wolf_panel, stretch=2)
        layout.addWidget(self.sequencer_grid, stretch=2)
        layout.addWidget(self.mixer_console, stretch=2)
        layout.addWidget(self.transport_controls, stretch=1)

        # Setup real-time updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_interface)
        self.update_timer.start(16)  # 60fps updates

    def update_interface(self):
        """Real-time interface updates for hardware sync"""
        # Update knob positions from hardware
        self.rhythm_wolf_panel.sync_with_hardware()
        # Update step LEDs with beat position
        self.sequencer_grid.update_playback_position()
        # Update VU meters
        self.mixer_console.update_audio_levels()

def main():
    app = QApplication(sys.argv)
    app.setOrganizationName("RhythmWolf Studios")
    app.setApplicationName("Rhythm Wolf Mini-DAW")

    # Set application icon
    app.setWindowIcon(QIcon("assets/gui_resources/app_icon.svg"))

    daw = RhythmWolfDAW()
    daw.show()

    return app.exec()

if __name__ == "__main__":
    main()