import mido
from PyQt6.QtCore import QObject, pyqtSignal, QTimer
from typing import Optional, Dict, List
import time

class RhythmWolfAutoRecord(QObject):
    """Handles automatic recording when hardware play button is pressed"""

    recording_started = pyqtSignal()
    recording_stopped = pyqtSignal()

    def __init__(self, daw_instance):
        super().__init__()
        self.daw = daw_instance
        self.is_hardware_playing = False
        self.auto_record_enabled = True
        self.recording_mode = 'FULL_CAPTURE'  # PATTERN_OVERDUB, PATTERN_REPLACE, AUTOMATION_ONLY, FULL_CAPTURE
        
        # Enhanced state tracking
        self.hardware_connected = False
        self.last_transport_message_time = 0
        self.transport_debounce_ms = 50  # Prevent rapid transport toggles
        self.error_count = 0
        self.max_errors = 5
        
        # Auto-record configuration
        self.pre_record_bars = 0  # Count-in bars before recording
        self.auto_stop_enabled = True  # Auto-stop when hardware stops
        self.punch_recording = False  # Punch in/out recording
        
        # Performance tracking
        self.record_start_latency = 0
        self.total_recordings = 0
        
        # Watchdog timer for transport state
        self.transport_watchdog = QTimer()
        self.transport_watchdog.timeout.connect(self._check_transport_state)
        self.transport_watchdog.start(1000)  # Check every second

    def set_auto_record_enabled(self, enabled):
        """Enable/disable auto-recording feature"""
        self.auto_record_enabled = enabled

    def set_recording_mode(self, mode):
        """Set recording mode for auto-record"""
        valid_modes = ['PATTERN_OVERDUB', 'PATTERN_REPLACE', 'AUTOMATION_ONLY', 'FULL_CAPTURE']
        if mode in valid_modes:
            self.recording_mode = mode

    def handle_hardware_transport(self, message):
        """Handle transport control from Rhythm Wolf hardware with enhanced error handling"""
        
        current_time = time.time() * 1000  # milliseconds
        
        # Debounce transport messages to prevent rapid toggles
        if current_time - self.last_transport_message_time < self.transport_debounce_ms:
            return
            
        self.last_transport_message_time = current_time
        
        try:
            # Rhythm Wolf Play button sends Start (0xFA) message
            if message.type == 'start':
                if self.auto_record_enabled and not self.is_hardware_playing:
                    if self.start_auto_recording():
                        self.is_hardware_playing = True
                        self.error_count = 0  # Reset error count on success
                    else:
                        self._handle_recording_error("Failed to start auto-recording")

            # Stop button sends Stop (0xFC) message  
            elif message.type == 'stop':
                if self.is_hardware_playing and self.auto_stop_enabled:
                    if self.stop_auto_recording():
                        self.is_hardware_playing = False
                        self.error_count = 0
                    else:
                        self._handle_recording_error("Failed to stop auto-recording")

            # Continue button sends Continue (0xFB) message
            elif message.type == 'continue':
                if self.auto_record_enabled and not self.is_hardware_playing:
                    if self.resume_auto_recording():
                        self.is_hardware_playing = True
                        self.error_count = 0
                    else:
                        self._handle_recording_error("Failed to resume auto-recording")
                        
        except Exception as e:
            self._handle_recording_error(f"Transport handling error: {e}")

    def start_auto_recording(self) -> bool:
        """Automatically start recording when hardware play is pressed"""
        record_start_time = time.time()
        
        try:
            print("🔴 Auto-Recording Started (Hardware Play Pressed)")

            # Pre-flight checks
            if not self._validate_recording_state():
                return False

            # Configure recording based on mode
            if self.recording_mode == 'PATTERN_OVERDUB':
                self.daw.set_recording_mode('overdub')
            elif self.recording_mode == 'PATTERN_REPLACE':
                self.daw.set_recording_mode('replace')
            elif self.recording_mode == 'AUTOMATION_ONLY':
                self.daw.set_recording_mode('automation')
            else:  # FULL_CAPTURE
                self.daw.set_recording_mode('full')

            # Handle count-in if enabled
            if self.pre_record_bars > 0:
                self._start_count_in()
                return True

            # Enable recording on all armed tracks
            if not self.daw.start_recording():
                print("❌ Failed to start DAW recording")
                return False

            # Start the sequencer playback synchronized
            if not self.daw.start_playback():
                print("❌ Failed to start DAW playback")
                self.daw.stop_recording()  # Clean up
                return False

            # Track performance
            self.record_start_latency = (time.time() - record_start_time) * 1000
            self.total_recordings += 1
            
            # Emit signal for UI updates
            self.recording_started.emit()
            
            print(f"✅ Auto-recording started successfully ({self.record_start_latency:.1f}ms latency)")
            return True
            
        except Exception as e:
            print(f"❌ Auto-recording start failed: {e}")
            return False

    def stop_auto_recording(self) -> bool:
        """Stop recording when hardware stop is pressed"""
        try:
            print("⏹️ Auto-Recording Stopped (Hardware Stop Pressed)")

            # Stop recording but keep the recorded data
            if not self.daw.stop_recording():
                print("⚠️ Warning: DAW recording stop failed")

            # Stop playback
            if not self.daw.stop_playback():
                print("⚠️ Warning: DAW playback stop failed")

            # Emit signal for UI updates
            self.recording_stopped.emit()
            
            print("✅ Auto-recording stopped successfully")
            return True
            
        except Exception as e:
            print(f"❌ Auto-recording stop failed: {e}")
            return False

    def resume_auto_recording(self) -> bool:
        """Resume recording when hardware continue is pressed"""
        try:
            print("▶️ Auto-Recording Resumed (Hardware Continue Pressed)")

            # Resume recording and playback
            if not self.daw.resume_recording():
                print("❌ Failed to resume DAW recording")
                return False
                
            if not self.daw.resume_playback():
                print("❌ Failed to resume DAW playback")
                self.daw.stop_recording()  # Clean up
                return False

            # Emit signal for UI updates
            self.recording_started.emit()
            
            print("✅ Auto-recording resumed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Auto-recording resume failed: {e}")
            return False

class SmartRecordingModes:
    """Smart recording mode management"""

    def __init__(self):
        self.recording_modes = {
            'PATTERN_OVERDUB': 'Layer new patterns over existing',
            'PATTERN_REPLACE': 'Replace current pattern completely', 
            'AUTOMATION_ONLY': 'Record only knob movements',
            'FULL_CAPTURE': 'Record everything (notes + automation)'
        }
        self.current_mode = 'FULL_CAPTURE'

    def set_recording_mode_from_hardware(self, shift_pressed, pad_number):
        """Change recording mode using Shift + Pad combinations"""
        if shift_pressed:
            if pad_number == 1:    # Shift + Pad 1
                self.current_mode = 'PATTERN_OVERDUB'
            elif pad_number == 2:  # Shift + Pad 2  
                self.current_mode = 'PATTERN_REPLACE'
            elif pad_number == 3:  # Shift + Pad 3
                self.current_mode = 'AUTOMATION_ONLY'
            elif pad_number == 4:  # Shift + Pad 4
                self.current_mode = 'FULL_CAPTURE'

        return self.current_mode
        
    def _validate_recording_state(self) -> bool:
        """Validate that recording can be started"""
        if not self.daw:
            print("❌ No DAW instance available")
            return False
            
        if self.daw.is_recording():
            print("⚠️ DAW is already recording")
            return False
            
        return True
        
    def _handle_recording_error(self, error_message: str):
        """Handle recording errors with retry logic"""
        self.error_count += 1
        print(f"❌ Auto-Record Error ({self.error_count}/{self.max_errors}): {error_message}")
        
        if self.error_count >= self.max_errors:
            print("🚨 Max errors reached - disabling auto-record")
            self.auto_record_enabled = False
            
    def _start_count_in(self):
        """Start count-in before recording"""
        print(f"⏱️ Count-in: {self.pre_record_bars} bars")
        # Implement count-in logic here
        # For now, just delay the actual recording start
        QTimer.singleShot(int(self.pre_record_bars * 2000), self._delayed_record_start)
        
    def _delayed_record_start(self):
        """Start recording after count-in"""
        self.daw.start_recording()
        self.daw.start_playback()
        self.recording_started.emit()
        
    def _check_transport_state(self):
        """Watchdog to check transport state consistency"""
        if not self.daw:
            return
            
        # Check for state inconsistencies
        daw_playing = self.daw.is_playing()
        daw_recording = self.daw.is_recording()
        
        # If hardware says playing but DAW is not, try to sync
        if self.is_hardware_playing and not daw_playing:
            print("⚠️ Transport state mismatch detected - attempting sync")
            self.daw.start_playback()
            
        # Reset error count periodically if things are working
        if daw_playing or daw_recording:
            self.error_count = max(0, self.error_count - 1)
            
    def set_hardware_connected(self, connected: bool):
        """Update hardware connection status"""
        self.hardware_connected = connected
        if not connected:
            # Hardware disconnected - stop any active recording
            if self.is_hardware_playing:
                self.stop_auto_recording()
                self.is_hardware_playing = False
                
    def set_pre_record_bars(self, bars: int):
        """Set number of count-in bars"""
        self.pre_record_bars = max(0, bars)
        
    def set_auto_stop_enabled(self, enabled: bool):
        """Enable/disable auto-stop when hardware stops"""
        self.auto_stop_enabled = enabled
        
    def set_punch_recording(self, enabled: bool):
        """Enable/disable punch recording mode"""
        self.punch_recording = enabled
        
    def get_recording_stats(self) -> Dict[str, float]:
        """Get recording performance statistics"""
        return {
            'total_recordings': self.total_recordings,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.total_recordings, 1),
            'avg_start_latency_ms': self.record_start_latency,
            'auto_record_enabled': self.auto_record_enabled,
            'hardware_connected': self.hardware_connected
        }
        
    def reset_error_count(self):
        """Reset error count and re-enable auto-record if disabled"""
        self.error_count = 0
        if not self.auto_record_enabled:
            self.auto_record_enabled = True
            print("✅ Auto-record re-enabled after error reset")


class AdvancedAutoRecord(RhythmWolfAutoRecord):
    """Enhanced auto-record with advanced features"""
    
    # Additional signals
    count_in_started = pyqtSignal(int)      # bars remaining
    punch_in_activated = pyqtSignal()       # punch recording started
    punch_out_activated = pyqtSignal()      # punch recording stopped
    
    def __init__(self, daw_instance):
        super().__init__(daw_instance)
        
        # Advanced features
        self.loop_recording = False
        self.auto_quantize = True
        self.record_arm_tracks = []  # Specific tracks to record
        self.midi_merge_mode = True  # Merge with existing MIDI
        
        # Timing features
        self.sync_to_pattern = True
        self.pattern_length_bars = 4
        self.auto_loop_length = 4  # bars
        
        # Smart recording features
        self.auto_track_selection = True
        self.velocity_recording = True
        self.aftertouch_recording = False
        
    def set_loop_recording(self, enabled: bool, loop_bars: int = 4):
        """Enable loop recording with specified length"""
        self.loop_recording = enabled
        self.auto_loop_length = loop_bars
        
    def set_record_arm_tracks(self, track_list: List[str]):
        """Set specific tracks to arm for recording"""
        self.record_arm_tracks = track_list.copy()
        
    def start_punch_recording(self, start_bar: int, end_bar: int):
        """Start punch-in recording at specified bars"""
        if not self.daw.is_playing():
            print("⚠️ Punch recording requires playback to be active")
            return False
            
        self.punch_recording = True
        # Schedule punch in/out
        QTimer.singleShot(start_bar * 2000, self._punch_in)
        QTimer.singleShot(end_bar * 2000, self._punch_out)
        return True
        
    def _punch_in(self):
        """Execute punch-in"""
        if self.daw.start_recording():
            self.punch_in_activated.emit()
            print("🎯 Punch-in activated")
            
    def _punch_out(self):
        """Execute punch-out"""
        if self.daw.stop_recording():
            self.punch_out_activated.emit()
            print("🎯 Punch-out activated")
            self.punch_recording = False