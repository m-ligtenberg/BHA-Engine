"""
Professional Audio Playback Engine
Sample-accurate playback with transport controls and synchronization
"""

import numpy as np
import sounddevice as sd
from typing import Dict, List, Optional, Tuple, Any, Callable
from threading import RLock, Thread, Event
from enum import Enum
from PyQt6.QtCore import QObject, pyqtSignal, QThread
import time
import queue

from .track import AudioTrack, TrackManager
from .mixer import ProfessionalMixer
from .recorder import MultiTrackRecorder


class PlaybackState(Enum):
    """Playback engine states"""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    SEEKING = "seeking"
    BUFFERING = "buffering"


class TransportMode(Enum):
    """Transport control modes"""
    PLAY = "play"
    PAUSE = "pause"
    STOP = "stop"
    RECORD = "record"
    REWIND = "rewind"
    FAST_FORWARD = "fast_forward"


class AudioPlaybackEngine(QObject):
    """
    Professional audio playback engine with sample-accurate timing,
    transport controls, and comprehensive synchronization capabilities.
    """
    
    # Transport control signals
    playback_started = pyqtSignal()
    playback_stopped = pyqtSignal()
    playback_paused = pyqtSignal()
    position_changed = pyqtSignal(float)  # position in seconds
    tempo_changed = pyqtSignal(float)     # BPM
    
    # Audio system signals
    audio_device_changed = pyqtSignal(str, str)  # device_name, device_type
    buffer_underrun = pyqtSignal(int)     # underrun count
    audio_overload = pyqtSignal(float)    # peak level
    latency_changed = pyqtSignal(float)   # latency in ms
    
    # Performance monitoring signals
    performance_update = pyqtSignal(dict)  # performance statistics
    
    def __init__(self, sample_rate: int = 44100, buffer_size: int = 512,
                 channels: int = 2):
        super().__init__()
        
        # Audio configuration
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.channels = channels
        self.bit_depth = 32  # float32
        
        # Core components
        self.track_manager = None
        self.mixer = None
        self.recorder = None
        
        # Playback state
        self.state = PlaybackState.STOPPED
        self.transport_mode = TransportMode.STOP
        self.position_samples = 0
        self.position_seconds = 0.0
        self.tempo_bpm = 120.0
        
        # Thread safety
        self._lock = RLock()
        self._audio_lock = RLock()
        
        # Audio device management
        self.audio_stream = None
        self.input_device_id = None
        self.output_device_id = None
        self.device_info = {}
        
        # Timing and synchronization
        self.samples_per_beat = 0
        self.samples_per_bar = 0
        self.current_beat = 0
        self.current_bar = 0
        self.metronome_enabled = False
        self.metronome_volume = 0.5
        
        # Playback buffers
        self.output_buffer = np.zeros((buffer_size, channels), dtype=np.float32)
        self.input_buffer = np.zeros((buffer_size, channels), dtype=np.float32)
        
        # Pre-roll and post-roll
        self.preroll_samples = sample_rate * 2  # 2 seconds pre-roll
        self.postroll_samples = sample_rate * 1  # 1 second post-roll
        
        # Performance monitoring
        self.callback_time_ms = 0.0
        self.max_callback_time = 0.0
        self.underrun_count = 0
        self.overrun_count = 0
        self.dropped_frames = 0
        self.cpu_load = 0.0
        
        # Audio callback queue for thread communication
        self.audio_queue = queue.Queue(maxsize=10)
        self.command_queue = queue.Queue(maxsize=100)
        
        # Synchronization events
        self.playback_event = Event()
        self.stop_event = Event()
        
        # Calculate timing constants
        self._update_timing_constants()
        
        # Initialize audio system
        self._initialize_audio_system()
    
    def set_components(self, track_manager: TrackManager, mixer: ProfessionalMixer, 
                      recorder: MultiTrackRecorder):
        """Set the core audio components"""
        self.track_manager = track_manager
        self.mixer = mixer
        self.recorder = recorder
    
    def _initialize_audio_system(self):
        """Initialize the audio system and detect devices"""
        try:
            # Get available audio devices
            self.device_info = {
                'input_devices': [],
                'output_devices': [],
                'sample_rates': [],
                'buffer_sizes': []
            }
            
            # Query available devices
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                device_info = {
                    'id': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'] if device['max_input_channels'] > 0 else device['max_output_channels'],
                    'default_sample_rate': device['default_samplerate'],
                    'hostapi': device['hostapi']
                }
                
                if device['max_input_channels'] > 0:
                    self.device_info['input_devices'].append(device_info)
                if device['max_output_channels'] > 0:
                    self.device_info['output_devices'].append(device_info)
            
            # Set default devices
            try:
                default_device = sd.query_devices(kind='output')
                self.output_device_id = default_device['name']
                
                default_input = sd.query_devices(kind='input')
                self.input_device_id = default_input['name']
                
            except Exception as e:
                print(f"Warning: Could not set default audio devices: {e}")
                
        except Exception as e:
            print(f"Error initializing audio system: {e}")
    
    def start_playback(self, from_position: float = None) -> bool:
        """Start audio playback"""
        with self._lock:
            if self.state == PlaybackState.PLAYING:
                return True
            
            try:
                # Set playback position if specified
                if from_position is not None:
                    self.seek_to_position(from_position)
                
                # Open audio stream
                if not self._open_audio_stream():
                    return False
                
                # Update state
                self.state = PlaybackState.PLAYING
                self.transport_mode = TransportMode.PLAY
                
                # Start the audio stream
                if self.audio_stream:
                    self.audio_stream.start()
                
                # Set events
                self.playback_event.set()
                self.stop_event.clear()
                
                self.playback_started.emit()
                return True
                
            except Exception as e:
                print(f"Error starting playback: {e}")
                return False
    
    def stop_playback(self) -> bool:
        """Stop audio playback"""
        with self._lock:
            if self.state == PlaybackState.STOPPED:
                return True
            
            try:
                # Update state
                self.state = PlaybackState.STOPPED
                self.transport_mode = TransportMode.STOP
                
                # Set events
                self.stop_event.set()
                self.playback_event.clear()
                
                # Stop and close audio stream
                self._close_audio_stream()
                
                # Reset position to beginning
                self.position_samples = 0
                self.position_seconds = 0.0
                self._update_beat_position()
                
                self.playback_stopped.emit()
                self.position_changed.emit(0.0)
                return True
                
            except Exception as e:
                print(f"Error stopping playback: {e}")
                return False
    
    def pause_playback(self) -> bool:
        """Pause audio playback"""
        with self._lock:
            if self.state != PlaybackState.PLAYING:
                return False
            
            try:
                # Update state
                self.state = PlaybackState.PAUSED
                self.transport_mode = TransportMode.PAUSE
                
                # Pause audio stream
                if self.audio_stream:
                    self.audio_stream.stop()
                
                self.playback_event.clear()
                self.playback_paused.emit()
                return True
                
            except Exception as e:
                print(f"Error pausing playback: {e}")
                return False
    
    def resume_playback(self) -> bool:
        """Resume paused playback"""
        with self._lock:
            if self.state != PlaybackState.PAUSED:
                return False
            
            return self.start_playback()
    
    def seek_to_position(self, position_seconds: float) -> bool:
        """Seek to specific position with sample accuracy"""
        with self._lock:
            try:
                # Validate position
                position_seconds = max(0.0, position_seconds)
                
                # Calculate sample position
                new_position_samples = int(position_seconds * self.sample_rate)
                
                # Update position
                self.position_samples = new_position_samples
                self.position_seconds = position_seconds
                
                # Update beat/bar position
                self._update_beat_position()
                
                self.position_changed.emit(position_seconds)
                return True
                
            except Exception as e:
                print(f"Error seeking to position: {e}")
                return False
    
    def set_tempo(self, bpm: float) -> bool:
        """Set playback tempo with smooth transitions"""
        with self._lock:
            try:
                # Validate BPM
                bpm = max(60.0, min(200.0, bpm))
                
                if abs(bpm - self.tempo_bpm) < 0.1:
                    return True
                
                self.tempo_bpm = bpm
                self._update_timing_constants()
                
                # Update beat position
                self._update_beat_position()
                
                self.tempo_changed.emit(bpm)
                return True
                
            except Exception as e:
                print(f"Error setting tempo: {e}")
                return False
    
    def _update_timing_constants(self):
        """Update timing constants based on current tempo"""
        beats_per_second = self.tempo_bpm / 60.0
        self.samples_per_beat = int(self.sample_rate / beats_per_second)
        self.samples_per_bar = self.samples_per_beat * 4  # Assuming 4/4 time
    
    def _update_beat_position(self):
        """Update current beat and bar based on position"""
        if self.samples_per_beat > 0:
            self.current_beat = int(self.position_samples / self.samples_per_beat) % 4
            self.current_bar = int(self.position_samples / self.samples_per_bar)
    
    def _open_audio_stream(self) -> bool:
        """Open the audio stream with optimal settings"""
        try:
            if self.audio_stream is not None:
                self._close_audio_stream()
            
            # Create audio stream
            self.audio_stream = sd.OutputStream(
                device=self.output_device_id,
                samplerate=self.sample_rate,
                blocksize=self.buffer_size,
                channels=self.channels,
                dtype=np.float32,
                callback=self._audio_callback,
                finished_callback=self._stream_finished_callback,
                latency='low'
            )
            
            return True
            
        except Exception as e:
            print(f"Error opening audio stream: {e}")
            return False
    
    def _close_audio_stream(self):
        """Close the audio stream safely"""
        try:
            if self.audio_stream is not None:
                if self.audio_stream.active:
                    self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None
        except Exception as e:
            print(f"Error closing audio stream: {e}")
    
    def _audio_callback(self, outdata: np.ndarray, frames: int, time, status):
        """
        Main audio callback - must be optimized for real-time performance.
        This runs in the audio thread with strict timing requirements.
        """
        callback_start = time.time()
        
        try:
            # Check for underruns or overruns
            if status:
                if status.output_underflow:
                    self.underrun_count += 1
                    self.buffer_underrun.emit(self.underrun_count)
            
            # Clear output buffer
            outdata.fill(0.0)
            
            # Generate audio if playing
            if self.state == PlaybackState.PLAYING and self.mixer:
                # Create time info for components
                time_info = {
                    'position_samples': self.position_samples,
                    'position_seconds': self.position_seconds,
                    'tempo_bpm': self.tempo_bpm,
                    'current_beat': self.current_beat,
                    'current_bar': self.current_bar,
                    'buffer_size': frames
                }
                
                # Process audio through mixer
                main_output, cue_output = self.mixer.process_audio_frame(
                    self.input_buffer[:frames], frames, time_info
                )
                
                # Copy to output
                if main_output.shape[0] >= frames:
                    outdata[:] = main_output[:frames]
                
                # Add metronome if enabled
                if self.metronome_enabled:
                    self._add_metronome(outdata, frames, time_info)
                
                # Update position
                self.position_samples += frames
                self.position_seconds = self.position_samples / self.sample_rate
                self._update_beat_position()
                
                # Emit position update (throttled)
                if self.position_samples % (self.sample_rate // 10) == 0:  # 10Hz updates
                    self.position_changed.emit(self.position_seconds)
            
            # Performance monitoring
            callback_time = (time.time() - callback_start) * 1000
            self.callback_time_ms = callback_time
            self.max_callback_time = max(self.max_callback_time, callback_time)
            
            # Check for performance issues
            max_allowed_time = (frames / self.sample_rate) * 1000 * 0.8  # 80% of buffer time
            if callback_time > max_allowed_time:
                self.cpu_load = (callback_time / max_allowed_time) * 100
                if self.cpu_load > 90:
                    self.audio_overload.emit(self.cpu_load)
            
        except Exception as e:
            # Fail-safe: output silence on error
            outdata.fill(0.0)
            self.underrun_count += 1
            print(f"Audio callback error: {e}")
    
    def _add_metronome(self, output_data: np.ndarray, frames: int, time_info: Dict):
        """Add metronome click to output"""
        # Simple metronome implementation
        beat_sample = self.position_samples % self.samples_per_beat
        
        if beat_sample < frames:
            # Generate click
            click_frequency = 1000.0 if self.current_beat == 0 else 800.0  # Accent on beat 1
            click_duration_samples = int(0.01 * self.sample_rate)  # 10ms click
            
            if beat_sample + click_duration_samples < frames:
                t = np.arange(click_duration_samples) / self.sample_rate
                click = np.sin(2 * np.pi * click_frequency * t) * self.metronome_volume
                
                # Apply envelope
                envelope = np.exp(-t * 50)  # Fast decay
                click *= envelope
                
                # Add to output
                start_idx = int(beat_sample)
                end_idx = start_idx + click_duration_samples
                
                if self.channels == 2:
                    output_data[start_idx:end_idx, :] += click.reshape(-1, 1)
                else:
                    output_data[start_idx:end_idx] += click
    
    def _stream_finished_callback(self):
        """Called when audio stream finishes"""
        print("Audio stream finished")
    
    def set_input_device(self, device_id: str) -> bool:
        """Set input audio device"""
        try:
            # Validate device
            if device_id not in [d['name'] for d in self.device_info['input_devices']]:
                return False
            
            self.input_device_id = device_id
            self.audio_device_changed.emit(device_id, "input")
            return True
            
        except Exception as e:
            print(f"Error setting input device: {e}")
            return False
    
    def set_output_device(self, device_id: str) -> bool:
        """Set output audio device"""
        try:
            # Validate device
            if device_id not in [d['name'] for d in self.device_info['output_devices']]:
                return False
            
            # Close existing stream
            was_playing = self.state == PlaybackState.PLAYING
            if was_playing:
                self.pause_playback()
            
            self.output_device_id = device_id
            
            # Restart if was playing
            if was_playing:
                self.resume_playback()
            
            self.audio_device_changed.emit(device_id, "output")
            return True
            
        except Exception as e:
            print(f"Error setting output device: {e}")
            return False
    
    def set_buffer_size(self, buffer_size: int) -> bool:
        """Set audio buffer size"""
        try:
            # Validate buffer size
            valid_sizes = [64, 128, 256, 512, 1024, 2048]
            if buffer_size not in valid_sizes:
                return False
            
            # Store old state
            was_playing = self.state == PlaybackState.PLAYING
            old_position = self.position_seconds
            
            # Stop playback
            if was_playing:
                self.stop_playback()
            
            # Update buffer size
            self.buffer_size = buffer_size
            self.output_buffer = np.zeros((buffer_size, self.channels), dtype=np.float32)
            self.input_buffer = np.zeros((buffer_size, self.channels), dtype=np.float32)
            
            # Restart if was playing
            if was_playing:
                self.start_playback(old_position)
            
            return True
            
        except Exception as e:
            print(f"Error setting buffer size: {e}")
            return False
    
    def set_sample_rate(self, sample_rate: int) -> bool:
        """Set audio sample rate"""
        try:
            # Validate sample rate
            valid_rates = [44100, 48000, 88200, 96000]
            if sample_rate not in valid_rates:
                return False
            
            # Store old state
            was_playing = self.state == PlaybackState.PLAYING
            old_position_ratio = self.position_seconds if self.position_seconds > 0 else 0
            
            # Stop playback
            if was_playing:
                self.stop_playback()
            
            # Update sample rate
            self.sample_rate = sample_rate
            self._update_timing_constants()
            
            # Update components if present
            if self.mixer:
                self.mixer.sample_rate = sample_rate
            if self.recorder:
                self.recorder.sample_rate = sample_rate
            
            # Restart if was playing
            if was_playing and old_position_ratio > 0:
                self.start_playback(old_position_ratio)
            
            return True
            
        except Exception as e:
            print(f"Error setting sample rate: {e}")
            return False
    
    def enable_metronome(self, enabled: bool = True, volume: float = 0.5):
        """Enable/disable metronome"""
        self.metronome_enabled = enabled
        self.metronome_volume = max(0.0, min(1.0, volume))
    
    def get_playback_info(self) -> Dict[str, Any]:
        """Get current playback information"""
        return {
            'state': self.state.value,
            'transport_mode': self.transport_mode.value,
            'position_seconds': self.position_seconds,
            'position_samples': self.position_samples,
            'tempo_bpm': self.tempo_bpm,
            'current_beat': self.current_beat,
            'current_bar': self.current_bar,
            'metronome_enabled': self.metronome_enabled,
            'sample_rate': self.sample_rate,
            'buffer_size': self.buffer_size,
            'channels': self.channels,
            'input_device': self.input_device_id,
            'output_device': self.output_device_id
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'callback_time_ms': self.callback_time_ms,
            'max_callback_time_ms': self.max_callback_time,
            'cpu_load_percent': self.cpu_load,
            'underrun_count': self.underrun_count,
            'overrun_count': self.overrun_count,
            'dropped_frames': self.dropped_frames,
            'buffer_usage_percent': (self.callback_time_ms / (self.buffer_size / self.sample_rate * 1000)) * 100
        }
    
    def reset_performance_stats(self):
        """Reset performance counters"""
        self.max_callback_time = 0.0
        self.underrun_count = 0
        self.overrun_count = 0
        self.dropped_frames = 0
        self.cpu_load = 0.0
    
    def __del__(self):
        """Cleanup on destruction"""
        self._close_audio_stream()