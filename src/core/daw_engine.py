"""
DAW Engine - Main orchestrator for the Rhythm Wolf Mini-DAW
"""

from PyQt6.QtCore import QObject, pyqtSignal

class DAWEngine(QObject):
    """Main DAW orchestrator that coordinates all components"""

    # Signals for state changes
    playback_started = pyqtSignal()
    playback_stopped = pyqtSignal()
    recording_started = pyqtSignal()
    recording_stopped = pyqtSignal()
    tempo_changed = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.is_playing_flag = False
        self.is_recording_flag = False
        self.master_tempo = 120.0
        self.recording_mode = 'full'

    def is_playing(self):
        """Check if playback is active"""
        return self.is_playing_flag

    def is_recording(self):
        """Check if recording is active"""
        return self.is_recording_flag

    def set_master_tempo(self, bpm):
        """Set master tempo"""
        self.master_tempo = bpm
        self.tempo_changed.emit(bpm)
        print(f"Master tempo set to {bpm} BPM")

    def get_master_tempo(self):
        """Get current master tempo"""
        return self.master_tempo

    def set_recording_mode(self, mode):
        """Set recording mode"""
        valid_modes = ['full', 'overdub', 'replace', 'automation']
        if mode in valid_modes:
            self.recording_mode = mode

    def start_playback(self):
        """Start playback"""
        if not self.is_playing_flag:
            self.is_playing_flag = True
            self.playback_started.emit()
            print("Playback started")

    def stop_playback(self):
        """Stop playback"""
        if self.is_playing_flag:
            self.is_playing_flag = False
            self.playback_stopped.emit()
            print("Playback stopped")

    def resume_playback(self):
        """Resume playback"""
        self.start_playback()

    def start_recording(self):
        """Start recording"""
        if not self.is_recording_flag:
            self.is_recording_flag = True
            self.recording_started.emit()
            print(f"Recording started in {self.recording_mode} mode")

    def stop_recording(self):
        """Stop recording"""
        if self.is_recording_flag:
            self.is_recording_flag = False
            self.recording_stopped.emit()
            print("Recording stopped")

    def resume_recording(self):
        """Resume recording"""
        self.start_recording()

    def sync_effects_to_tempo(self, bpm):
        """Sync time-based effects to new tempo"""
        # Placeholder for effects synchronization
        print(f"Syncing effects to {bpm} BPM")

    def schedule_tempo_change(self, new_bpm, fade_time=0.1):
        """Schedule smooth tempo change during playback"""
        # Placeholder for smooth tempo transitions
        print(f"Scheduling tempo change to {new_bpm} BPM with {fade_time}s fade")
        self.set_master_tempo(new_bpm)

    def gradual_effects_sync(self, new_bpm):
        """Gradually sync effects to new tempo"""
        # Placeholder for gradual effects sync
        self.sync_effects_to_tempo(new_bpm)

    def resync_pattern_playback(self, new_bpm):
        """Maintain pattern sync with new tempo"""
        # Placeholder for pattern resync
        print(f"Resyncing patterns to {new_bpm} BPM")