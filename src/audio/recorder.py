"""
Professional Multi-Track Audio Recorder
Advanced recording capabilities with punch-in/out, overdubbing, and automation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from threading import RLock, Event, Thread
from enum import Enum
from PyQt6.QtCore import QObject, pyqtSignal, QTimer
import time
import uuid
from pathlib import Path
import json

from .track import AudioTrack, TrackManager


class RecordingMode(Enum):
    """Recording modes supported by the system"""
    OVERDUB = "overdub"      # Layer new audio over existing
    REPLACE = "replace"      # Replace existing audio
    PUNCH = "punch"          # Punch-in/out recording
    LOOP = "loop"            # Loop recording with overdubs
    AUTOMATION = "automation" # Record automation only


class RecordingState(Enum):
    """Recording system states"""
    STOPPED = "stopped"
    WAITING = "waiting"      # Armed and waiting for trigger
    RECORDING = "recording"  # Actively recording
    PAUSED = "paused"       # Recording paused
    PUNCH_WAITING = "punch_waiting"  # Waiting for punch-in
    PUNCH_RECORDING = "punch_recording"  # Active punch recording


class AudioRecording:
    """Represents a single audio recording with metadata"""
    
    def __init__(self, track_id: int, recording_id: str = None):
        self.recording_id = recording_id or str(uuid.uuid4())
        self.track_id = track_id
        self.audio_data = np.array([], dtype=np.float32).reshape(0, 2)
        self.sample_rate = 44100
        self.start_time = 0.0
        self.end_time = 0.0
        self.length_samples = 0
        self.length_seconds = 0.0
        self.created_at = time.time()
        self.metadata = {}
        
        # Recording quality metrics
        self.peak_level = 0.0
        self.rms_level = 0.0
        self.clipping_count = 0
        self.dynamic_range = 0.0
    
    def append_audio(self, audio_chunk: np.ndarray):
        """Append audio data to this recording"""
        if self.audio_data.size == 0:
            self.audio_data = audio_chunk.copy()
        else:
            self.audio_data = np.vstack([self.audio_data, audio_chunk])
        
        self.length_samples = len(self.audio_data)
        self.length_seconds = self.length_samples / self.sample_rate
        
        # Update quality metrics
        chunk_peak = np.max(np.abs(audio_chunk))
        chunk_rms = np.sqrt(np.mean(audio_chunk ** 2))
        
        self.peak_level = max(self.peak_level, chunk_peak)
        if chunk_rms > 0:
            self.rms_level = max(self.rms_level, chunk_rms)
        
        # Check for clipping
        if chunk_peak >= 0.99:
            self.clipping_count += np.sum(np.abs(audio_chunk) >= 0.99)
    
    def get_audio_segment(self, start_sample: int = 0, end_sample: int = None) -> np.ndarray:
        """Get a segment of the recorded audio"""
        if end_sample is None:
            end_sample = self.length_samples
        
        start_sample = max(0, min(start_sample, self.length_samples))
        end_sample = max(start_sample, min(end_sample, self.length_samples))
        
        return self.audio_data[start_sample:end_sample]
    
    def calculate_metrics(self):
        """Calculate comprehensive audio quality metrics"""
        if self.length_samples == 0:
            return
        
        # Dynamic range calculation
        if self.rms_level > 0:
            self.dynamic_range = 20 * np.log10(self.peak_level / self.rms_level)
        
        self.metadata.update({
            'peak_level_db': 20 * np.log10(self.peak_level + 1e-10),
            'rms_level_db': 20 * np.log10(self.rms_level + 1e-10),
            'dynamic_range_db': self.dynamic_range,
            'clipping_percentage': (self.clipping_count / self.length_samples) * 100,
            'length_seconds': self.length_seconds,
            'sample_rate': self.sample_rate
        })


class MultiTrackRecorder(QObject):
    """
    Professional multi-track recording system with advanced features:
    - Punch-in/punch-out recording
    - Multiple recording modes (overdub, replace, loop)
    - Automatic gain control and level monitoring
    - Recording session management
    - Export capabilities
    """
    
    # Signals for system communication
    recording_started = pyqtSignal(int, str)  # track_id, recording_id
    recording_stopped = pyqtSignal(int, str)  # track_id, recording_id
    recording_paused = pyqtSignal(int)  # track_id
    punch_in_triggered = pyqtSignal(int, float)  # track_id, time
    punch_out_triggered = pyqtSignal(int, float)  # track_id, time
    recording_levels = pyqtSignal(int, float, float)  # track_id, peak, rms
    recording_error = pyqtSignal(str, str)  # error_type, message
    auto_punch_triggered = pyqtSignal(int, str)  # track_id, action
    
    def __init__(self, track_manager: TrackManager, sample_rate: int = 44100, 
                 buffer_size: int = 512, max_recording_time: int = 3600):
        super().__init__()
        
        self.track_manager = track_manager
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.max_recording_time = max_recording_time  # Maximum recording time in seconds
        
        # Recording state management
        self.state = RecordingState.STOPPED
        self.recording_mode = RecordingMode.OVERDUB
        self.global_recording = False
        self.playback_position = 0.0  # Current playback position in seconds
        
        # Thread safety
        self._lock = RLock()
        self._recording_lock = RLock()
        
        # Active recordings storage
        self.active_recordings: Dict[int, AudioRecording] = {}  # track_id -> recording
        self.recording_history: Dict[int, List[AudioRecording]] = {}  # track_id -> recordings list
        
        # Punch recording settings
        self.punch_in_time = 0.0
        self.punch_out_time = 0.0
        self.auto_punch_enabled = False
        self.punch_tracks: List[int] = []
        
        # Loop recording settings
        self.loop_enabled = False
        self.loop_start_time = 0.0
        self.loop_end_time = 0.0
        self.loop_count = 0
        self.max_loop_layers = 10
        
        # Recording quality settings
        self.auto_gain_enabled = False
        self.noise_gate_enabled = False
        self.noise_gate_threshold = -60.0  # dB
        self.recording_format = 'float32'
        
        # Performance monitoring
        self.dropped_samples = 0
        self.buffer_overruns = 0
        self.recording_latency_ms = 0.0
        
        # File management
        self.recording_directory = Path.cwd() / "recordings"
        self.recording_directory.mkdir(exist_ok=True)
        self.auto_save_enabled = True
        self.auto_save_interval = 30.0  # seconds
        
        # Setup auto-save timer
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self._auto_save_recordings)
        if self.auto_save_enabled:
            self.auto_save_timer.start(int(self.auto_save_interval * 1000))
    
    def set_recording_mode(self, mode: RecordingMode):
        """Set the recording mode"""
        with self._lock:
            self.recording_mode = mode
            
            # Configure tracks based on mode
            if mode == RecordingMode.REPLACE:
                # Clear existing audio on armed tracks
                for track in self.track_manager.get_all_tracks():
                    if track.is_armed:
                        track.clear_recorded_audio()
    
    def arm_track(self, track_id: int, armed: bool = True):
        """Arm/disarm track for recording"""
        track = self.track_manager.get_track(track_id)
        if not track:
            return False
        
        with self._lock:
            track.arm_track(armed)
            
            if armed:
                # Initialize recording history if needed
                if track_id not in self.recording_history:
                    self.recording_history[track_id] = []
                
                # Set up recording parameters
                if self.recording_mode == RecordingMode.PUNCH:
                    if track_id not in self.punch_tracks:
                        self.punch_tracks.append(track_id)
            else:
                # Remove from punch tracks
                if track_id in self.punch_tracks:
                    self.punch_tracks.remove(track_id)
        
        return True
    
    def start_recording(self, track_ids: List[int] = None) -> bool:
        """Start recording on specified tracks (or all armed tracks)"""
        with self._recording_lock:
            if self.state == RecordingState.RECORDING:
                return False
            
            # Determine tracks to record
            if track_ids is None:
                tracks_to_record = [t for t in self.track_manager.get_all_tracks() if t.is_armed]
            else:
                tracks_to_record = [self.track_manager.get_track(tid) for tid in track_ids]
                tracks_to_record = [t for t in tracks_to_record if t and t.is_armed]
            
            if not tracks_to_record:
                self.recording_error.emit("no_armed_tracks", "No armed tracks available for recording")
                return False
            
            # Initialize recordings
            for track in tracks_to_record:
                recording = AudioRecording(track.track_id)
                recording.sample_rate = self.sample_rate
                recording.start_time = self.playback_position
                
                self.active_recordings[track.track_id] = recording
                track.set_state(track.TrackState.RECORDING)
                
                self.recording_started.emit(track.track_id, recording.recording_id)
            
            # Set recording state
            if self.recording_mode == RecordingMode.PUNCH and self.auto_punch_enabled:
                self.state = RecordingState.PUNCH_WAITING
            else:
                self.state = RecordingState.RECORDING
            
            self.global_recording = True
            return True
    
    def stop_recording(self, track_ids: List[int] = None) -> bool:
        """Stop recording on specified tracks (or all active recordings)"""
        with self._recording_lock:
            if not self.global_recording:
                return False
            
            # Determine tracks to stop
            if track_ids is None:
                tracks_to_stop = list(self.active_recordings.keys())
            else:
                tracks_to_stop = [tid for tid in track_ids if tid in self.active_recordings]
            
            # Finalize recordings
            for track_id in tracks_to_stop:
                track = self.track_manager.get_track(track_id)
                recording = self.active_recordings[track_id]
                
                if track and recording:
                    # Finalize recording
                    recording.end_time = self.playback_position
                    recording.calculate_metrics()
                    
                    # Move to history
                    if track_id not in self.recording_history:
                        self.recording_history[track_id] = []
                    self.recording_history[track_id].append(recording)
                    
                    # Update track state
                    track.set_state(track.TrackState.STOPPED)
                    
                    self.recording_stopped.emit(track_id, recording.recording_id)
                
                # Remove from active recordings
                del self.active_recordings[track_id]
            
            # Update global state
            if not self.active_recordings:
                self.state = RecordingState.STOPPED
                self.global_recording = False
            
            return True
    
    def pause_recording(self, track_ids: List[int] = None) -> bool:
        """Pause recording on specified tracks"""
        with self._recording_lock:
            if self.state not in [RecordingState.RECORDING, RecordingState.PUNCH_RECORDING]:
                return False
            
            if track_ids is None:
                track_ids = list(self.active_recordings.keys())
            
            for track_id in track_ids:
                track = self.track_manager.get_track(track_id)
                if track:
                    track.set_state(track.TrackState.STOPPED)
                    self.recording_paused.emit(track_id)
            
            if not track_ids:
                self.state = RecordingState.PAUSED
            
            return True
    
    def resume_recording(self, track_ids: List[int] = None) -> bool:
        """Resume paused recording"""
        with self._recording_lock:
            if self.state != RecordingState.PAUSED:
                return False
            
            if track_ids is None:
                track_ids = list(self.active_recordings.keys())
            
            for track_id in track_ids:
                track = self.track_manager.get_track(track_id)
                if track and track.is_armed:
                    track.set_state(track.TrackState.RECORDING)
            
            self.state = RecordingState.RECORDING
            return True
    
    def process_audio_input(self, track_id: int, audio_data: np.ndarray, 
                           current_time: float) -> bool:
        """
        Process incoming audio for recording.
        Called from audio thread - must be optimized for real-time.
        """
        if not self.global_recording or track_id not in self.active_recordings:
            return False
        
        track = self.track_manager.get_track(track_id)
        recording = self.active_recordings[track_id]
        
        if not track or not recording:
            return False
        
        try:
            # Check punch recording logic
            if self.recording_mode == RecordingMode.PUNCH:
                if not self._check_punch_timing(track_id, current_time):
                    return False
            
            # Apply noise gate if enabled
            if self.noise_gate_enabled:
                audio_data = self._apply_noise_gate(audio_data)
            
            # Apply auto-gain if enabled
            if self.auto_gain_enabled:
                audio_data = self._apply_auto_gain(audio_data, track_id)
            
            # Record the audio
            recording.append_audio(audio_data)
            
            # Update real-time levels
            peak = np.max(np.abs(audio_data))
            rms = np.sqrt(np.mean(audio_data ** 2))
            self.recording_levels.emit(track_id, peak, rms)
            
            # Check for maximum recording time
            if recording.length_seconds > self.max_recording_time:
                self.stop_recording([track_id])
                self.recording_error.emit("max_time_exceeded", 
                                        f"Maximum recording time exceeded on track {track_id}")
                return False
            
            return True
            
        except Exception as e:
            self.buffer_overruns += 1
            self.recording_error.emit("processing_error", str(e))
            return False
    
    def _check_punch_timing(self, track_id: int, current_time: float) -> bool:
        """Check if we should be recording based on punch times"""
        if not self.auto_punch_enabled:
            return True
        
        # Check punch-in
        if self.state == RecordingState.PUNCH_WAITING:
            if current_time >= self.punch_in_time:
                self.state = RecordingState.PUNCH_RECORDING
                self.punch_in_triggered.emit(track_id, current_time)
                self.auto_punch_triggered.emit(track_id, "punch_in")
                return True
            else:
                return False
        
        # Check punch-out
        if self.state == RecordingState.PUNCH_RECORDING:
            if current_time >= self.punch_out_time:
                self.stop_recording([track_id])
                self.punch_out_triggered.emit(track_id, current_time)
                self.auto_punch_triggered.emit(track_id, "punch_out")
                return False
            else:
                return True
        
        return True
    
    def _apply_noise_gate(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply noise gate to reduce background noise"""
        # Simple noise gate implementation
        threshold_linear = 10 ** (self.noise_gate_threshold / 20.0)
        gate_mask = np.abs(audio_data) > threshold_linear
        
        # Apply gate with smooth transitions
        gated_audio = audio_data.copy()
        gated_audio[~gate_mask] *= 0.1  # Reduce by 20dB instead of complete silence
        
        return gated_audio
    
    def _apply_auto_gain(self, audio_data: np.ndarray, track_id: int) -> np.ndarray:
        """Apply automatic gain control"""
        # Simple AGC - target around -12dB peak
        current_peak = np.max(np.abs(audio_data))
        target_peak = 10 ** (-12.0 / 20.0)  # -12dB in linear
        
        if current_peak > 0:
            gain_adjustment = target_peak / current_peak
            # Limit gain adjustment to prevent artifacts
            gain_adjustment = np.clip(gain_adjustment, 0.1, 3.0)
            return audio_data * gain_adjustment
        
        return audio_data
    
    def set_punch_times(self, punch_in: float, punch_out: float):
        """Set automatic punch-in/out times"""
        with self._lock:
            self.punch_in_time = punch_in
            self.punch_out_time = punch_out
            self.auto_punch_enabled = punch_out > punch_in > 0
            
            # Update all punch tracks
            for track_id in self.punch_tracks:
                track = self.track_manager.get_track(track_id)
                if track:
                    track.set_punch_times(punch_in, punch_out)
    
    def enable_loop_recording(self, start_time: float, end_time: float, 
                             max_layers: int = 10):
        """Enable loop recording mode"""
        with self._lock:
            self.loop_enabled = True
            self.loop_start_time = start_time
            self.loop_end_time = end_time
            self.max_loop_layers = max_layers
            self.loop_count = 0
    
    def disable_loop_recording(self):
        """Disable loop recording mode"""
        with self._lock:
            self.loop_enabled = False
            self.loop_count = 0
    
    def get_recording(self, track_id: int, recording_id: str = None) -> Optional[AudioRecording]:
        """Get a specific recording"""
        if recording_id is None:
            # Return most recent recording
            if track_id in self.active_recordings:
                return self.active_recordings[track_id]
            elif track_id in self.recording_history and self.recording_history[track_id]:
                return self.recording_history[track_id][-1]
        else:
            # Search for specific recording ID
            if track_id in self.recording_history:
                for recording in self.recording_history[track_id]:
                    if recording.recording_id == recording_id:
                        return recording
        
        return None
    
    def get_track_recordings(self, track_id: int) -> List[AudioRecording]:
        """Get all recordings for a track"""
        recordings = []
        
        # Add active recording if present
        if track_id in self.active_recordings:
            recordings.append(self.active_recordings[track_id])
        
        # Add historical recordings
        if track_id in self.recording_history:
            recordings.extend(self.recording_history[track_id])
        
        return recordings
    
    def delete_recording(self, track_id: int, recording_id: str) -> bool:
        """Delete a specific recording"""
        with self._recording_lock:
            if track_id in self.recording_history:
                for i, recording in enumerate(self.recording_history[track_id]):
                    if recording.recording_id == recording_id:
                        del self.recording_history[track_id][i]
                        return True
        return False
    
    def clear_track_recordings(self, track_id: int):
        """Clear all recordings for a track"""
        with self._recording_lock:
            if track_id in self.active_recordings:
                del self.active_recordings[track_id]
            if track_id in self.recording_history:
                self.recording_history[track_id].clear()
    
    def export_recording(self, track_id: int, recording_id: str, 
                        file_path: Path, format: str = 'wav') -> bool:
        """Export a recording to file"""
        recording = self.get_recording(track_id, recording_id)
        if not recording:
            return False
        
        try:
            # This would integrate with audio file export library
            # For now, save as numpy array
            np.save(file_path.with_suffix('.npy'), recording.audio_data)
            
            # Save metadata
            metadata_path = file_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(recording.metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            self.recording_error.emit("export_error", str(e))
            return False
    
    def _auto_save_recordings(self):
        """Auto-save active recordings"""
        if not self.auto_save_enabled:
            return
        
        try:
            for track_id, recording in self.active_recordings.items():
                if recording.length_samples > 0:
                    filename = f"autosave_track_{track_id}_{int(time.time())}"
                    file_path = self.recording_directory / filename
                    self.export_recording(track_id, recording.recording_id, file_path)
                    
        except Exception as e:
            self.recording_error.emit("autosave_error", str(e))
    
    def get_recording_stats(self) -> Dict[str, Any]:
        """Get comprehensive recording statistics"""
        active_count = len(self.active_recordings)
        total_recordings = sum(len(recordings) for recordings in self.recording_history.values())
        
        total_duration = 0.0
        total_size_mb = 0.0
        
        for recordings in self.recording_history.values():
            for recording in recordings:
                total_duration += recording.length_seconds
                # Estimate size (float32 * 2 channels * samples)
                size_bytes = recording.length_samples * 4 * 2
                total_size_mb += size_bytes / (1024 * 1024)
        
        return {
            'active_recordings': active_count,
            'total_recordings': total_recordings,
            'total_duration_seconds': total_duration,
            'total_size_mb': total_size_mb,
            'recording_state': self.state.value,
            'recording_mode': self.recording_mode.value,
            'dropped_samples': self.dropped_samples,
            'buffer_overruns': self.buffer_overruns,
            'punch_enabled': self.auto_punch_enabled,
            'loop_enabled': self.loop_enabled
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.dropped_samples = 0
        self.buffer_overruns = 0
        self.recording_latency_ms = 0.0