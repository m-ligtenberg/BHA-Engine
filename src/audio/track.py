"""
Professional Audio Track Management System
Handles individual track state, routing, and processing for the Rhythm Wolf Mini-DAW
"""

import numpy as np
from typing import Optional, List, Dict, Any, Callable
from threading import RLock, Event
from enum import Enum
from PyQt6.QtCore import QObject, pyqtSignal
import time


class TrackState(Enum):
    """Track playback states"""
    STOPPED = "stopped"
    PLAYING = "playing" 
    RECORDING = "recording"
    MUTED = "muted"
    SOLO = "solo"


class AudioTrack(QObject):
    """
    Professional audio track with routing, effects, and comprehensive state management.
    Optimized for real-time processing with lock-free audio paths.
    """
    
    # Signals for GUI updates
    level_changed = pyqtSignal(float, float)  # peak, rms
    state_changed = pyqtSignal(str)  # state name
    parameter_changed = pyqtSignal(str, float)  # param_name, value
    
    def __init__(self, track_id: int, name: str, track_type: str = "audio", 
                 sample_rate: int = 44100, buffer_size: int = 512):
        super().__init__()
        
        # Basic track properties
        self.track_id = track_id
        self.name = name
        self.track_type = track_type  # 'audio', 'midi', 'aux'
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        
        # Thread safety
        self._lock = RLock()
        self._audio_lock = RLock()  # Separate lock for audio data
        
        # Track state
        self.state = TrackState.STOPPED
        self.is_armed = False
        self.is_muted = False
        self.is_solo = False
        self.is_monitoring = False
        
        # Audio parameters (using atomic operations for thread safety)
        self._volume = 1.0  # Linear gain (0.0 - 2.0)
        self._pan = 0.0     # Pan position (-1.0 = left, 1.0 = right)
        self._gain_db = 0.0 # Pre-amp gain in dB (-60 to +24)
        
        # Audio routing
        self.input_source = None
        self.output_routing = "main"  # main, aux1, aux2, etc.
        self.send_levels = {}  # Send levels to aux buses
        
        # Audio buffers (lock-free circular buffers for real-time use)
        self.audio_buffer_frames = sample_rate * 2  # 2 seconds of audio
        self.input_buffer = np.zeros((self.audio_buffer_frames, 2), dtype=np.float32)
        self.output_buffer = np.zeros((self.audio_buffer_frames, 2), dtype=np.float32)
        self.recorded_buffer = np.zeros((0, 2), dtype=np.float32)
        
        # Buffer pointers for lock-free operation
        self._input_write_pos = 0
        self._input_read_pos = 0
        self._output_write_pos = 0
        self._output_read_pos = 0
        
        # Effects chain (will be populated by effects system)
        self.effects_chain = None
        self.effects_enabled = True
        
        # Metering data
        self.peak_level_l = 0.0
        self.peak_level_r = 0.0
        self.rms_level_l = 0.0
        self.rms_level_r = 0.0
        self.peak_hold_time = 0.0
        self.meter_update_interval = 0.05  # 50ms updates
        self.last_meter_update = 0.0
        
        # Recording state
        self.punch_in_enabled = False
        self.punch_out_enabled = False
        self.punch_in_time = 0.0
        self.punch_out_time = 0.0
        self.auto_punch_enabled = False
        
        # Performance monitoring
        self.processing_time_ms = 0.0
        self.cpu_usage_percent = 0.0
        self.buffer_underruns = 0
        self.max_processing_time = 0.0

    @property
    def volume(self) -> float:
        """Get track volume (linear gain)"""
        return self._volume
    
    @volume.setter
    def volume(self, value: float):
        """Set track volume with bounds checking"""
        self._volume = max(0.0, min(2.0, value))
        self.parameter_changed.emit("volume", self._volume)
    
    @property
    def volume_db(self) -> float:
        """Get volume in dB"""
        if self._volume <= 0.0:
            return -60.0
        return 20.0 * np.log10(self._volume)
    
    @volume_db.setter
    def volume_db(self, db_value: float):
        """Set volume from dB value"""
        if db_value <= -60.0:
            self._volume = 0.0
        else:
            self._volume = 10.0 ** (db_value / 20.0)
        self.parameter_changed.emit("volume", self._volume)
    
    @property
    def pan(self) -> float:
        """Get pan position (-1.0 to 1.0)"""
        return self._pan
    
    @pan.setter  
    def pan(self, value: float):
        """Set pan position with bounds checking"""
        self._pan = max(-1.0, min(1.0, value))
        self.parameter_changed.emit("pan", self._pan)
    
    @property
    def gain_db(self) -> float:
        """Get pre-amp gain in dB"""
        return self._gain_db
    
    @gain_db.setter
    def gain_db(self, db_value: float):
        """Set pre-amp gain with bounds checking"""
        self._gain_db = max(-60.0, min(24.0, db_value))
        self.parameter_changed.emit("gain", self._gain_db)

    def set_state(self, new_state: TrackState):
        """Thread-safe state change"""
        with self._lock:
            old_state = self.state
            self.state = new_state
            if old_state != new_state:
                self.state_changed.emit(new_state.value)

    def arm_track(self, armed: bool = True):
        """Arm/disarm track for recording"""
        with self._lock:
            self.is_armed = armed
            if armed:
                self.clear_recorded_audio()

    def mute(self, muted: bool = True):
        """Mute/unmute track"""
        with self._lock:
            self.is_muted = muted
            if muted:
                self.clear_output_buffer()

    def solo(self, solo: bool = True):
        """Solo/unsolo track"""
        with self._lock:
            self.is_solo = solo

    def set_monitoring(self, enabled: bool):
        """Enable/disable input monitoring"""
        with self._lock:
            self.is_monitoring = enabled

    def process_audio(self, input_data: np.ndarray, output_data: np.ndarray, 
                     frames: int, time_info: Dict) -> np.ndarray:
        """
        Main audio processing function - optimized for real-time performance.
        This is called from the audio thread and must be lock-free.
        """
        start_time = time.perf_counter()
        
        try:
            # Apply pre-amp gain
            if self._gain_db != 0.0:
                gain_linear = 10.0 ** (self._gain_db / 20.0)
                input_data *= gain_linear
            
            # Process through effects chain if present
            processed_audio = input_data
            if self.effects_chain and self.effects_enabled:
                processed_audio = self.effects_chain.process(processed_audio, frames)
            
            # Apply volume and pan
            if self._volume != 1.0 or self._pan != 0.0:
                processed_audio = self._apply_volume_pan(processed_audio)
            
            # Handle mute
            if self.is_muted:
                processed_audio = np.zeros_like(processed_audio)
            
            # Update metering (non-blocking)
            current_time = time.perf_counter()
            if current_time - self.last_meter_update > self.meter_update_interval:
                self._update_metering(processed_audio)
                self.last_meter_update = current_time
            
            # Recording logic
            if self.state == TrackState.RECORDING and self.is_armed:
                self._record_audio(processed_audio)
            
            # Copy to output buffer for monitoring/playback
            if self.is_monitoring or self.state == TrackState.PLAYING:
                output_data[:] = processed_audio
            
            # Performance monitoring
            processing_time = (time.perf_counter() - start_time) * 1000
            self.processing_time_ms = processing_time
            self.max_processing_time = max(self.max_processing_time, processing_time)
            
            return processed_audio
            
        except Exception as e:
            # Fail-safe: return silence on error to prevent audio glitches
            self.buffer_underruns += 1
            return np.zeros_like(input_data)

    def _apply_volume_pan(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply volume and pan with optimized calculations"""
        if audio_data.shape[1] == 2:  # Stereo
            # Calculate pan gains using constant-power panning law
            pan_rad = self._pan * np.pi / 4  # Convert to radians
            left_gain = np.cos(pan_rad) * self._volume
            right_gain = np.sin(pan_rad + np.pi/2) * self._volume
            
            audio_data[:, 0] *= left_gain
            audio_data[:, 1] *= right_gain
        else:  # Mono
            audio_data *= self._volume
            
        return audio_data

    def _update_metering(self, audio_data: np.ndarray):
        """Update peak and RMS metering (non-blocking)"""
        if audio_data.shape[1] == 2:  # Stereo
            # Peak levels
            self.peak_level_l = max(self.peak_level_l * 0.95, np.max(np.abs(audio_data[:, 0])))
            self.peak_level_r = max(self.peak_level_r * 0.95, np.max(np.abs(audio_data[:, 1])))
            
            # RMS levels (smoothed)
            rms_l = np.sqrt(np.mean(audio_data[:, 0] ** 2))
            rms_r = np.sqrt(np.mean(audio_data[:, 1] ** 2))
            self.rms_level_l = self.rms_level_l * 0.9 + rms_l * 0.1
            self.rms_level_r = self.rms_level_r * 0.9 + rms_r * 0.1
        else:  # Mono
            self.peak_level_l = max(self.peak_level_l * 0.95, np.max(np.abs(audio_data)))
            self.peak_level_r = self.peak_level_l
            rms = np.sqrt(np.mean(audio_data ** 2))
            self.rms_level_l = self.rms_level_r = self.rms_level_l * 0.9 + rms * 0.1
        
        # Emit signal for GUI updates (queued to avoid blocking audio thread)
        peak_db = 20 * np.log10(max(self.peak_level_l, self.peak_level_r) + 1e-10)
        rms_db = 20 * np.log10(max(self.rms_level_l, self.rms_level_r) + 1e-10)
        self.level_changed.emit(peak_db, rms_db)

    def _record_audio(self, audio_data: np.ndarray):
        """Record audio data to buffer (optimized for real-time)"""
        # Check punch-in/out times if auto-punch is enabled
        current_time = time.perf_counter()
        if self.auto_punch_enabled:
            if self.punch_in_enabled and current_time < self.punch_in_time:
                return
            if self.punch_out_enabled and current_time > self.punch_out_time:
                self.set_state(TrackState.PLAYING)
                return
        
        # Append to recorded buffer (will be optimized with circular buffer later)
        if self.recorded_buffer.size == 0:
            self.recorded_buffer = audio_data.copy()
        else:
            self.recorded_buffer = np.vstack([self.recorded_buffer, audio_data])

    def clear_recorded_audio(self):
        """Clear recorded audio buffer"""
        with self._audio_lock:
            self.recorded_buffer = np.zeros((0, 2), dtype=np.float32)

    def clear_output_buffer(self):
        """Clear output buffer"""
        with self._audio_lock:
            self.output_buffer.fill(0.0)

    def get_recorded_audio(self) -> np.ndarray:
        """Get recorded audio data"""
        with self._audio_lock:
            return self.recorded_buffer.copy()

    def set_send_level(self, aux_bus: str, level: float):
        """Set send level to auxiliary bus"""
        with self._lock:
            self.send_levels[aux_bus] = max(0.0, min(2.0, level))

    def get_send_level(self, aux_bus: str) -> float:
        """Get send level for auxiliary bus"""
        return self.send_levels.get(aux_bus, 0.0)

    def set_punch_times(self, punch_in: float, punch_out: float):
        """Set automatic punch-in/out times"""
        with self._lock:
            self.punch_in_time = punch_in
            self.punch_out_time = punch_out
            self.punch_in_enabled = punch_in > 0
            self.punch_out_enabled = punch_out > punch_in

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring"""
        return {
            'processing_time_ms': self.processing_time_ms,
            'cpu_usage_percent': self.cpu_usage_percent,
            'max_processing_time_ms': self.max_processing_time,
            'buffer_underruns': self.buffer_underruns,
            'peak_level_db': 20 * np.log10(max(self.peak_level_l, self.peak_level_r) + 1e-10),
            'rms_level_db': 20 * np.log10(max(self.rms_level_l, self.rms_level_r) + 1e-10)
        }

    def reset_performance_stats(self):
        """Reset performance monitoring counters"""
        self.max_processing_time = 0.0
        self.buffer_underruns = 0

    def __str__(self) -> str:
        return f"AudioTrack(id={self.track_id}, name='{self.name}', type='{self.track_type}')"

    def __repr__(self) -> str:
        return self.__str__()


class TrackManager(QObject):
    """
    Manages multiple audio tracks with optimized routing and processing.
    Handles track creation, deletion, and global operations.
    """
    
    track_added = pyqtSignal(int)  # track_id
    track_removed = pyqtSignal(int)  # track_id
    solo_changed = pyqtSignal(list)  # list of soloed track_ids
    
    def __init__(self, max_tracks: int = 16, sample_rate: int = 44100, buffer_size: int = 512):
        super().__init__()
        
        self.max_tracks = max_tracks
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        
        # Track storage
        self.tracks: Dict[int, AudioTrack] = {}
        self.track_order: List[int] = []  # For maintaining track order
        self.next_track_id = 1
        
        # Solo system
        self.solo_active = False
        self.soloed_tracks: List[int] = []
        
        # Thread safety
        self._lock = RLock()

    def create_track(self, name: str, track_type: str = "audio") -> AudioTrack:
        """Create a new audio track"""
        with self._lock:
            if len(self.tracks) >= self.max_tracks:
                raise ValueError(f"Maximum number of tracks ({self.max_tracks}) reached")
            
            track_id = self.next_track_id
            track = AudioTrack(
                track_id=track_id,
                name=name,
                track_type=track_type,
                sample_rate=self.sample_rate,
                buffer_size=self.buffer_size
            )
            
            self.tracks[track_id] = track
            self.track_order.append(track_id)
            self.next_track_id += 1
            
            # Connect signals
            track.state_changed.connect(self._on_track_state_changed)
            
            self.track_added.emit(track_id)
            return track

    def remove_track(self, track_id: int) -> bool:
        """Remove a track"""
        with self._lock:
            if track_id not in self.tracks:
                return False
            
            # Remove from solo list if present
            if track_id in self.soloed_tracks:
                self.soloed_tracks.remove(track_id)
                self._update_solo_state()
            
            # Remove track
            del self.tracks[track_id]
            self.track_order.remove(track_id)
            
            self.track_removed.emit(track_id)
            return True

    def get_track(self, track_id: int) -> Optional[AudioTrack]:
        """Get track by ID"""
        return self.tracks.get(track_id)

    def get_all_tracks(self) -> List[AudioTrack]:
        """Get all tracks in order"""
        return [self.tracks[tid] for tid in self.track_order if tid in self.tracks]

    def set_track_solo(self, track_id: int, solo: bool):
        """Set track solo state"""
        with self._lock:
            if track_id not in self.tracks:
                return
            
            track = self.tracks[track_id]
            track.solo(solo)
            
            if solo and track_id not in self.soloed_tracks:
                self.soloed_tracks.append(track_id)
            elif not solo and track_id in self.soloed_tracks:
                self.soloed_tracks.remove(track_id)
            
            self._update_solo_state()

    def _update_solo_state(self):
        """Update global solo state and track muting"""
        self.solo_active = len(self.soloed_tracks) > 0
        
        # Update track states based on solo
        for track_id, track in self.tracks.items():
            if self.solo_active:
                # If any tracks are soloed, mute non-soloed tracks
                should_be_audible = track_id in self.soloed_tracks
                track.is_muted = not should_be_audible and not track.is_muted
            
        self.solo_changed.emit(self.soloed_tracks)

    def _on_track_state_changed(self, state: str):
        """Handle track state changes"""
        # Placeholder for global state management
        pass

    def get_track_count(self) -> int:
        """Get current number of tracks"""
        return len(self.tracks)

    def clear_all_tracks(self):
        """Remove all tracks"""
        with self._lock:
            track_ids = list(self.tracks.keys())
            for track_id in track_ids:
                self.remove_track(track_id)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all tracks"""
        total_processing_time = sum(t.processing_time_ms for t in self.tracks.values())
        total_underruns = sum(t.buffer_underruns for t in self.tracks.values())
        max_processing_time = max((t.max_processing_time for t in self.tracks.values()), default=0.0)
        
        return {
            'total_tracks': len(self.tracks),
            'total_processing_time_ms': total_processing_time,
            'total_buffer_underruns': total_underruns,
            'max_processing_time_ms': max_processing_time,
            'solo_active': self.solo_active,
            'soloed_track_count': len(self.soloed_tracks)
        }