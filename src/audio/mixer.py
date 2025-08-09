"""
Professional Multi-Track Mixer Console
Advanced routing, sends, returns, and professional mixing capabilities
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from threading import RLock
from enum import Enum
from PyQt6.QtCore import QObject, pyqtSignal
import time

from .track import AudioTrack, TrackManager


class BusType(Enum):
    """Audio bus types"""
    MAIN = "main"
    AUX = "aux"
    RETURN = "return"
    GROUP = "group"
    CUE = "cue"


class AudioBus:
    """Professional audio bus with routing and processing"""
    
    def __init__(self, bus_id: str, bus_type: BusType, channels: int = 2, 
                 sample_rate: int = 44100, buffer_size: int = 512):
        self.bus_id = bus_id
        self.bus_type = bus_type
        self.channels = channels
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        
        # Audio parameters
        self._volume = 1.0
        self._pan = 0.0
        self.is_muted = False
        self.is_solo = False
        
        # Audio buffers
        self.input_buffer = np.zeros((buffer_size, channels), dtype=np.float32)
        self.output_buffer = np.zeros((buffer_size, channels), dtype=np.float32)
        self.send_buffer = np.zeros((buffer_size, channels), dtype=np.float32)
        
        # Effects chain
        self.effects_chain = None
        self.effects_enabled = True
        
        # Metering
        self.peak_level_l = 0.0
        self.peak_level_r = 0.0
        self.rms_level_l = 0.0
        self.rms_level_r = 0.0
        
        # Routing
        self.input_sources = []  # List of tracks/buses feeding this bus
        self.output_destinations = []  # List of buses this feeds
        
        # Performance tracking
        self.processing_time_ms = 0.0
    
    @property
    def volume(self) -> float:
        return self._volume
    
    @volume.setter
    def volume(self, value: float):
        self._volume = max(0.0, min(2.0, value))
    
    @property
    def volume_db(self) -> float:
        if self._volume <= 0.0:
            return -60.0
        return 20.0 * np.log10(self._volume)
    
    @volume_db.setter
    def volume_db(self, db_value: float):
        if db_value <= -60.0:
            self._volume = 0.0
        else:
            self._volume = 10.0 ** (db_value / 20.0)
    
    @property
    def pan(self) -> float:
        return self._pan
    
    @pan.setter
    def pan(self, value: float):
        self._pan = max(-1.0, min(1.0, value))
    
    def process_audio(self, frames: int) -> np.ndarray:
        """Process audio through this bus"""
        start_time = time.perf_counter()
        
        # Clear input buffer
        self.input_buffer.fill(0.0)
        
        # Sum all input sources
        for source in self.input_sources:
            if hasattr(source, 'output_buffer'):
                self.input_buffer += source.output_buffer[:frames]
        
        # Apply effects if present
        processed_audio = self.input_buffer
        if self.effects_chain and self.effects_enabled:
            processed_audio = self.effects_chain.process(processed_audio, frames)
        
        # Apply volume and pan
        if self._volume != 1.0 or self._pan != 0.0:
            processed_audio = self._apply_volume_pan(processed_audio)
        
        # Apply mute
        if self.is_muted:
            processed_audio = np.zeros_like(processed_audio)
        
        # Copy to output buffer
        self.output_buffer[:frames] = processed_audio
        
        # Update metering
        self._update_metering(processed_audio)
        
        # Track performance
        self.processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        return self.output_buffer[:frames]
    
    def _apply_volume_pan(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply volume and pan with constant-power law"""
        if audio_data.shape[1] == 2:  # Stereo
            pan_rad = self._pan * np.pi / 4
            left_gain = np.cos(pan_rad) * self._volume
            right_gain = np.sin(pan_rad + np.pi/2) * self._volume
            
            audio_data[:, 0] *= left_gain
            audio_data[:, 1] *= right_gain
        else:  # Mono
            audio_data *= self._volume
            
        return audio_data
    
    def _update_metering(self, audio_data: np.ndarray):
        """Update peak and RMS levels"""
        if audio_data.shape[1] == 2:
            self.peak_level_l = max(self.peak_level_l * 0.95, np.max(np.abs(audio_data[:, 0])))
            self.peak_level_r = max(self.peak_level_r * 0.95, np.max(np.abs(audio_data[:, 1])))
            
            rms_l = np.sqrt(np.mean(audio_data[:, 0] ** 2))
            rms_r = np.sqrt(np.mean(audio_data[:, 1] ** 2))
            self.rms_level_l = self.rms_level_l * 0.9 + rms_l * 0.1
            self.rms_level_r = self.rms_level_r * 0.9 + rms_r * 0.1
        else:
            peak = np.max(np.abs(audio_data))
            self.peak_level_l = self.peak_level_r = max(self.peak_level_l * 0.95, peak)
            rms = np.sqrt(np.mean(audio_data ** 2))
            self.rms_level_l = self.rms_level_r = self.rms_level_l * 0.9 + rms * 0.1


class ProfessionalMixer(QObject):
    """
    Professional multi-track mixer with advanced routing, sends/returns,
    and comprehensive audio management for the Rhythm Wolf Mini-DAW.
    """
    
    # Signals for GUI updates
    master_level_changed = pyqtSignal(float, float)  # peak, rms
    bus_level_changed = pyqtSignal(str, float, float)  # bus_id, peak, rms
    mixer_overload = pyqtSignal(str)  # warning message
    routing_changed = pyqtSignal()
    
    def __init__(self, sample_rate: int = 44100, buffer_size: int = 512):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        
        # Core components
        self.track_manager = TrackManager(sample_rate=sample_rate, buffer_size=buffer_size)
        
        # Audio buses
        self.buses: Dict[str, AudioBus] = {}
        self.bus_order: List[str] = []
        
        # Thread safety
        self._lock = RLock()
        
        # Master section
        self.master_volume = 1.0
        self.master_mute = False
        self.master_peak_l = 0.0
        self.master_peak_r = 0.0
        self.master_rms_l = 0.0
        self.master_rms_r = 0.0
        
        # Master output buffers
        self.master_output = np.zeros((buffer_size, 2), dtype=np.float32)
        self.master_cue_output = np.zeros((buffer_size, 2), dtype=np.float32)
        
        # Auxiliary sends/returns
        self.aux_send_count = 4
        self.aux_return_count = 4
        
        # Performance monitoring
        self.total_processing_time = 0.0
        self.cpu_usage_percent = 0.0
        self.buffer_underruns = 0
        self.overload_count = 0
        
        # Initialize default buses
        self._initialize_default_buses()
        
        # Connect track manager signals
        self.track_manager.track_added.connect(self._on_track_added)
        self.track_manager.track_removed.connect(self._on_track_removed)
    
    def _initialize_default_buses(self):
        """Initialize default mixer buses"""
        # Main stereo bus
        self.create_bus("main", BusType.MAIN, channels=2)
        
        # Auxiliary sends
        for i in range(1, self.aux_send_count + 1):
            self.create_bus(f"aux{i}", BusType.AUX, channels=2)
        
        # Auxiliary returns  
        for i in range(1, self.aux_return_count + 1):
            self.create_bus(f"return{i}", BusType.RETURN, channels=2)
        
        # Cue/headphone bus
        self.create_bus("cue", BusType.CUE, channels=2)
        
        # Group buses (for submixing)
        for i in range(1, 5):
            self.create_bus(f"group{i}", BusType.GROUP, channels=2)
    
    def create_bus(self, bus_id: str, bus_type: BusType, channels: int = 2) -> AudioBus:
        """Create a new audio bus"""
        with self._lock:
            if bus_id in self.buses:
                raise ValueError(f"Bus '{bus_id}' already exists")
            
            bus = AudioBus(
                bus_id=bus_id,
                bus_type=bus_type,
                channels=channels,
                sample_rate=self.sample_rate,
                buffer_size=self.buffer_size
            )
            
            self.buses[bus_id] = bus
            self.bus_order.append(bus_id)
            
            self.routing_changed.emit()
            return bus
    
    def remove_bus(self, bus_id: str) -> bool:
        """Remove an audio bus"""
        with self._lock:
            if bus_id not in self.buses or bus_id == "main":
                return False  # Cannot remove main bus
            
            # Remove routing connections
            bus = self.buses[bus_id]
            for source in bus.input_sources:
                if hasattr(source, 'output_routing'):
                    source.output_routing = "main"
            
            del self.buses[bus_id]
            self.bus_order.remove(bus_id)
            
            self.routing_changed.emit()
            return True
    
    def get_bus(self, bus_id: str) -> Optional[AudioBus]:
        """Get bus by ID"""
        return self.buses.get(bus_id)
    
    def create_track(self, name: str, track_type: str = "audio") -> AudioTrack:
        """Create a new track and set up routing"""
        track = self.track_manager.create_track(name, track_type)
        
        # Default routing to main bus
        self.route_track_to_bus(track.track_id, "main")
        
        return track
    
    def route_track_to_bus(self, track_id: int, bus_id: str):
        """Route track output to specified bus"""
        with self._lock:
            track = self.track_manager.get_track(track_id)
            bus = self.get_bus(bus_id)
            
            if not track or not bus:
                return False
            
            # Remove track from previous bus
            old_bus_id = getattr(track, 'output_routing', None)
            if old_bus_id and old_bus_id in self.buses:
                old_bus = self.buses[old_bus_id]
                if track in old_bus.input_sources:
                    old_bus.input_sources.remove(track)
            
            # Add track to new bus
            track.output_routing = bus_id
            if track not in bus.input_sources:
                bus.input_sources.append(track)
            
            self.routing_changed.emit()
            return True
    
    def set_send_level(self, track_id: int, aux_bus: str, level: float):
        """Set auxiliary send level for track"""
        track = self.track_manager.get_track(track_id)
        if track:
            track.set_send_level(aux_bus, level)
    
    def process_audio_frame(self, input_data: np.ndarray, frames: int, 
                           time_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main mixer processing function - processes all tracks and buses.
        Returns (main_output, cue_output)
        """
        start_time = time.perf_counter()
        
        try:
            # Clear master outputs
            self.master_output.fill(0.0)
            self.master_cue_output.fill(0.0)
            
            # Process all tracks first
            for track in self.track_manager.get_all_tracks():
                track_output = np.zeros((frames, 2), dtype=np.float32)
                track.process_audio(input_data, track_output, frames, time_info)
            
            # Process auxiliary sends
            self._process_auxiliary_sends(frames)
            
            # Process all buses in dependency order
            self._process_buses(frames)
            
            # Generate master output
            main_bus = self.get_bus("main")
            if main_bus:
                self.master_output[:frames] = main_bus.output_buffer[:frames]
                
                # Apply master volume and mute
                if not self.master_mute:
                    self.master_output *= self.master_volume
                else:
                    self.master_output.fill(0.0)
            
            # Generate cue output
            cue_bus = self.get_bus("cue")
            if cue_bus:
                self.master_cue_output[:frames] = cue_bus.output_buffer[:frames]
            
            # Update master metering
            self._update_master_metering()
            
            # Check for overloads
            self._check_overloads()
            
            # Performance tracking
            self.total_processing_time = (time.perf_counter() - start_time) * 1000
            self.cpu_usage_percent = (self.total_processing_time / (frames / self.sample_rate * 1000)) * 100
            
            return self.master_output[:frames], self.master_cue_output[:frames]
            
        except Exception as e:
            self.buffer_underruns += 1
            # Return silence on error
            return np.zeros((frames, 2)), np.zeros((frames, 2))
    
    def _process_auxiliary_sends(self, frames: int):
        """Process auxiliary send routing"""
        # Clear aux buses
        for i in range(1, self.aux_send_count + 1):
            aux_bus = self.get_bus(f"aux{i}")
            if aux_bus:
                aux_bus.input_buffer.fill(0.0)
        
        # Sum track sends to aux buses
        for track in self.track_manager.get_all_tracks():
            for i in range(1, self.aux_send_count + 1):
                aux_name = f"aux{i}"
                send_level = track.get_send_level(aux_name)
                
                if send_level > 0.0:
                    aux_bus = self.get_bus(aux_name)
                    if aux_bus and hasattr(track, 'output_buffer'):
                        # Add track output to aux bus with send level
                        aux_bus.input_buffer[:frames] += track.output_buffer[:frames] * send_level
    
    def _process_buses(self, frames: int):
        """Process all buses in correct order"""
        # Process in order: groups, aux, returns, main
        bus_processing_order = [
            [bus_id for bus_id, bus in self.buses.items() if bus.bus_type == BusType.GROUP],
            [bus_id for bus_id, bus in self.buses.items() if bus.bus_type == BusType.AUX],
            [bus_id for bus_id, bus in self.buses.items() if bus.bus_type == BusType.RETURN],
            [bus_id for bus_id, bus in self.buses.items() if bus.bus_type == BusType.MAIN],
            [bus_id for bus_id, bus in self.buses.items() if bus.bus_type == BusType.CUE]
        ]
        
        for bus_group in bus_processing_order:
            for bus_id in bus_group:
                bus = self.get_bus(bus_id)
                if bus:
                    bus.process_audio(frames)
                    
                    # Emit level changes for GUI
                    peak_db = 20 * np.log10(max(bus.peak_level_l, bus.peak_level_r) + 1e-10)
                    rms_db = 20 * np.log10(max(bus.rms_level_l, bus.rms_level_r) + 1e-10)
                    self.bus_level_changed.emit(bus_id, peak_db, rms_db)
    
    def _update_master_metering(self):
        """Update master bus metering"""
        if self.master_output.shape[1] == 2:
            self.master_peak_l = max(self.master_peak_l * 0.95, np.max(np.abs(self.master_output[:, 0])))
            self.master_peak_r = max(self.master_peak_r * 0.95, np.max(np.abs(self.master_output[:, 1])))
            
            rms_l = np.sqrt(np.mean(self.master_output[:, 0] ** 2))
            rms_r = np.sqrt(np.mean(self.master_output[:, 1] ** 2))
            self.master_rms_l = self.master_rms_l * 0.9 + rms_l * 0.1
            self.master_rms_r = self.master_rms_r * 0.9 + rms_r * 0.1
            
            # Emit master levels for GUI
            peak_db = 20 * np.log10(max(self.master_peak_l, self.master_peak_r) + 1e-10)
            rms_db = 20 * np.log10(max(self.master_rms_l, self.master_rms_r) + 1e-10)
            self.master_level_changed.emit(peak_db, rms_db)
    
    def _check_overloads(self):
        """Check for audio overloads and clipping"""
        # Check master output for clipping
        if np.max(np.abs(self.master_output)) > 0.99:
            self.overload_count += 1
            self.mixer_overload.emit(f"Master output clipping detected (count: {self.overload_count})")
        
        # Check individual buses
        for bus_id, bus in self.buses.items():
            if np.max(np.abs(bus.output_buffer)) > 0.99:
                self.mixer_overload.emit(f"Bus '{bus_id}' clipping detected")
    
    def _on_track_added(self, track_id: int):
        """Handle track addition"""
        # Default routing is already set up in create_track()
        pass
    
    def _on_track_removed(self, track_id: int):
        """Handle track removal"""
        # Remove track from all bus routing
        for bus in self.buses.values():
            bus.input_sources = [src for src in bus.input_sources 
                               if not (hasattr(src, 'track_id') and src.track_id == track_id)]
    
    def set_master_volume(self, volume: float):
        """Set master output volume"""
        self.master_volume = max(0.0, min(2.0, volume))
    
    def set_master_volume_db(self, db_value: float):
        """Set master volume in dB"""
        if db_value <= -60.0:
            self.master_volume = 0.0
        else:
            self.master_volume = 10.0 ** (db_value / 20.0)
    
    def mute_master(self, muted: bool = True):
        """Mute/unmute master output"""
        self.master_mute = muted
    
    def reset_all_levels(self):
        """Reset all levels to unity gain"""
        with self._lock:
            # Reset master
            self.master_volume = 1.0
            self.master_mute = False
            
            # Reset all tracks
            for track in self.track_manager.get_all_tracks():
                track.volume = 1.0
                track.pan = 0.0
                track.is_muted = False
                track.is_solo = False
            
            # Reset all buses
            for bus in self.buses.values():
                bus.volume = 1.0
                bus.pan = 0.0
                bus.is_muted = False
                bus.is_solo = False
    
    def get_mixer_state(self) -> Dict[str, Any]:
        """Get complete mixer state for saving/loading"""
        state = {
            'master_volume': self.master_volume,
            'master_mute': self.master_mute,
            'tracks': {},
            'buses': {},
            'routing': {}
        }
        
        # Save track states
        for track in self.track_manager.get_all_tracks():
            state['tracks'][track.track_id] = {
                'name': track.name,
                'volume': track.volume,
                'pan': track.pan,
                'gain_db': track.gain_db,
                'is_muted': track.is_muted,
                'is_solo': track.is_solo,
                'is_armed': track.is_armed,
                'output_routing': track.output_routing,
                'send_levels': track.send_levels.copy()
            }
        
        # Save bus states
        for bus_id, bus in self.buses.items():
            state['buses'][bus_id] = {
                'volume': bus.volume,
                'pan': bus.pan,
                'is_muted': bus.is_muted,
                'is_solo': bus.is_solo,
                'effects_enabled': bus.effects_enabled
            }
        
        return state
    
    def load_mixer_state(self, state: Dict[str, Any]):
        """Load complete mixer state"""
        with self._lock:
            # Load master settings
            self.master_volume = state.get('master_volume', 1.0)
            self.master_mute = state.get('master_mute', False)
            
            # Load track states
            for track_id, track_state in state.get('tracks', {}).items():
                track = self.track_manager.get_track(int(track_id))
                if track:
                    track.volume = track_state.get('volume', 1.0)
                    track.pan = track_state.get('pan', 0.0)
                    track.gain_db = track_state.get('gain_db', 0.0)
                    track.is_muted = track_state.get('is_muted', False)
                    track.is_solo = track_state.get('is_solo', False)
                    track.is_armed = track_state.get('is_armed', False)
                    track.output_routing = track_state.get('output_routing', 'main')
                    track.send_levels = track_state.get('send_levels', {})
            
            # Load bus states
            for bus_id, bus_state in state.get('buses', {}).items():
                bus = self.get_bus(bus_id)
                if bus:
                    bus.volume = bus_state.get('volume', 1.0)
                    bus.pan = bus_state.get('pan', 0.0)
                    bus.is_muted = bus_state.get('is_muted', False)
                    bus.is_solo = bus_state.get('is_solo', False)
                    bus.effects_enabled = bus_state.get('effects_enabled', True)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        track_stats = self.track_manager.get_performance_summary()
        
        return {
            'mixer_processing_time_ms': self.total_processing_time,
            'mixer_cpu_usage_percent': self.cpu_usage_percent,
            'mixer_buffer_underruns': self.buffer_underruns,
            'mixer_overload_count': self.overload_count,
            'master_peak_db': 20 * np.log10(max(self.master_peak_l, self.master_peak_r) + 1e-10),
            'master_rms_db': 20 * np.log10(max(self.master_rms_l, self.master_rms_r) + 1e-10),
            'total_buses': len(self.buses),
            **track_stats
        }
    
    def reset_performance_stats(self):
        """Reset all performance counters"""
        self.buffer_underruns = 0
        self.overload_count = 0
        self.track_manager.reset_performance_stats()
        
        for bus in self.buses.values():
            bus.processing_time_ms = 0.0