"""
Integrated Professional Audio Engine
Complete audio system integration with optimized performance
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from threading import RLock, Thread
from PyQt6.QtCore import QObject, pyqtSignal
import time

from .track import AudioTrack, TrackManager
from .mixer import ProfessionalMixer
from .recorder import MultiTrackRecorder
from .playback import AudioPlaybackEngine
from .monitoring import AudioMonitoringSystem
from ..effects.effects_chain import EffectsChain, effect_factory
from ..effects.processors.factory_integration import register_all_effects


class IntegratedAudioEngine(QObject):
    """
    Complete integrated audio engine that coordinates all components:
    - Multi-track audio processing
    - Professional mixing console
    - Recording capabilities
    - Effects processing
    - Real-time monitoring
    - Sample-accurate playback
    """
    
    # System-wide signals
    engine_started = pyqtSignal()
    engine_stopped = pyqtSignal()
    performance_warning = pyqtSignal(str, float)  # component, cpu_usage
    system_overload = pyqtSignal(float)  # total_cpu_usage
    track_created = pyqtSignal(int, str)  # track_id, name
    track_deleted = pyqtSignal(int)  # track_id
    
    def __init__(self, sample_rate: int = 44100, buffer_size: int = 512, max_tracks: int = 16):
        super().__init__()
        
        # Core configuration
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.max_tracks = max_tracks
        
        # Thread safety
        self._lock = RLock()
        self._audio_lock = RLock()
        
        # Core components
        self.track_manager = TrackManager(max_tracks, sample_rate, buffer_size)
        self.mixer = ProfessionalMixer(sample_rate, buffer_size)
        self.recorder = MultiTrackRecorder(self.track_manager, sample_rate, buffer_size)
        self.playback_engine = AudioPlaybackEngine(sample_rate, buffer_size)
        self.monitoring_system = AudioMonitoringSystem(sample_rate)
        
        # Initialize effects system
        register_all_effects()
        
        # Track effects chains
        self.track_effects: Dict[int, EffectsChain] = {}
        
        # Engine state
        self.engine_running = False
        self.performance_mode = 'balanced'  # 'balanced', 'low_latency', 'high_quality'
        
        # Performance monitoring
        self.total_cpu_usage = 0.0
        self.component_cpu_usage = {}
        self.performance_history = []
        self.max_history_size = 100
        self.performance_warning_threshold = 80.0  # CPU percentage
        self.performance_critical_threshold = 95.0
        
        # Connect core components
        self._connect_components()
        
        # Initialize rhythm wolf specific tracks
        self._initialize_rhythm_wolf_tracks()
    
    def _connect_components(self):
        """Connect all core components together"""
        # Connect playback engine to mixer and recorder
        self.playback_engine.set_components(self.track_manager, self.mixer, self.recorder)
        
        # Connect track manager signals
        self.track_manager.track_added.connect(self._on_track_added)
        self.track_manager.track_removed.connect(self._on_track_removed)
        
        # Connect monitoring signals
        self.monitoring_system.overload_detected.connect(self._on_overload_detected)
        self.monitoring_system.performance_update.connect(self._on_performance_update)
        
        # Connect mixer signals
        self.mixer.mixer_overload.connect(self._on_mixer_overload)
        
        # Connect recorder signals
        self.recorder.recording_error.connect(self._on_recording_error)
    
    def _initialize_rhythm_wolf_tracks(self):
        """Initialize tracks optimized for Akai Rhythm Wolf"""
        rhythm_wolf_tracks = [
            ('Kick', 'audio'),
            ('Snare', 'audio'), 
            ('Percussion', 'audio'),
            ('Open Hi-Hat', 'audio'),
            ('Closed Hi-Hat', 'audio'),
            ('Bass Synth', 'audio')
        ]
        
        for name, track_type in rhythm_wolf_tracks:
            track = self.create_track(name, track_type)
            self._setup_rhythm_wolf_effects(track, name)
    
    def _setup_rhythm_wolf_effects(self, track: AudioTrack, track_name: str):
        """Setup drum-optimized effects for Rhythm Wolf tracks"""
        effects_chain = self.track_effects[track.track_id]
        
        if 'Kick' in track_name:
            # Kick drum effects: Compressor + EQ + Saturation
            compressor = effect_factory.create_effect('compressor', f'{track_name}_comp', self.sample_rate, self.buffer_size)
            eq = effect_factory.create_effect('eq', f'{track_name}_eq', self.sample_rate, self.buffer_size)
            
            if compressor and eq:
                compressor.load_preset('kick_punch')
                eq.load_preset('kick_punch')
                
                effects_chain.add_effect(0, compressor)
                effects_chain.add_effect(1, eq)
        
        elif 'Snare' in track_name:
            # Snare drum effects: EQ + Compressor + Reverb
            eq = effect_factory.create_effect('eq', f'{track_name}_eq', self.sample_rate, self.buffer_size)
            compressor = effect_factory.create_effect('compressor', f'{track_name}_comp', self.sample_rate, self.buffer_size)
            reverb = effect_factory.create_effect('reverb', f'{track_name}_reverb', self.sample_rate, self.buffer_size)
            
            if eq and compressor and reverb:
                eq.load_preset('snare_crack')
                reverb.load_preset('snare_plate')
                
                effects_chain.add_effect(0, eq)
                effects_chain.add_effect(1, compressor)
                effects_chain.add_effect(2, reverb)
        
        elif 'Hi-Hat' in track_name:
            # Hi-hat effects: EQ + subtle compression
            eq = effect_factory.create_effect('eq', f'{track_name}_eq', self.sample_rate, self.buffer_size)
            
            if eq:
                eq.load_preset('hihat_sparkle')
                effects_chain.add_effect(0, eq)
        
        elif 'Bass' in track_name:
            # Bass synth effects: Saturation + Compressor
            saturation = effect_factory.create_effect('saturation', f'{track_name}_sat', self.sample_rate, self.buffer_size)
            compressor = effect_factory.create_effect('compressor', f'{track_name}_comp', self.sample_rate, self.buffer_size)
            
            if saturation and compressor:
                saturation.load_preset('subtle_warmth')
                
                effects_chain.add_effect(0, saturation)
                effects_chain.add_effect(1, compressor)
    
    def start_engine(self) -> bool:
        """Start the integrated audio engine"""
        with self._lock:
            if self.engine_running:
                return True
            
            try:
                # Start playback engine
                success = self.playback_engine.start_playback(0.0)
                if not success:
                    return False
                
                # Initialize monitoring for all tracks
                for track in self.track_manager.get_all_tracks():
                    self.monitoring_system.add_track_monitoring(str(track.track_id))
                
                self.engine_running = True
                self.engine_started.emit()
                
                print(f"Audio Engine Started:")
                print(f"  Sample Rate: {self.sample_rate} Hz")
                print(f"  Buffer Size: {self.buffer_size} samples")
                print(f"  Tracks: {len(self.track_manager.get_all_tracks())}")
                print(f"  Available Effects: {len(effect_factory.get_available_effects())}")
                
                return True
                
            except Exception as e:
                print(f"Error starting audio engine: {e}")
                return False
    
    def stop_engine(self) -> bool:
        """Stop the integrated audio engine"""
        with self._lock:
            if not self.engine_running:
                return True
            
            try:
                # Stop playback
                self.playback_engine.stop_playback()
                
                # Stop any active recordings
                self.recorder.stop_recording()
                
                self.engine_running = False
                self.engine_stopped.emit()
                
                return True
                
            except Exception as e:
                print(f"Error stopping audio engine: {e}")
                return False
    
    def create_track(self, name: str, track_type: str = 'audio') -> AudioTrack:
        """Create a new track with integrated effects and monitoring"""
        with self._lock:
            # Create track through mixer (handles routing)
            track = self.mixer.create_track(name, track_type)
            
            # Create effects chain for track
            effects_chain = EffectsChain(
                chain_id=f'track_{track.track_id}_fx',
                max_slots=6,
                sample_rate=self.sample_rate,
                buffer_size=self.buffer_size
            )
            
            # Integrate effects chain with track
            track.effects_chain = effects_chain
            self.track_effects[track.track_id] = effects_chain
            
            # Add monitoring
            if self.engine_running:
                self.monitoring_system.add_track_monitoring(str(track.track_id))
            
            self.track_created.emit(track.track_id, name)
            
            return track
    
    def delete_track(self, track_id: int) -> bool:
        """Delete a track and cleanup all associated resources"""
        with self._lock:
            # Remove from monitoring
            self.monitoring_system.remove_track_monitoring(str(track_id))
            
            # Remove effects chain
            if track_id in self.track_effects:
                del self.track_effects[track_id]
            
            # Remove from mixer
            success = self.mixer.track_manager.remove_track(track_id)
            
            if success:
                self.track_deleted.emit(track_id)
            
            return success
    
    def get_track(self, track_id: int) -> Optional[AudioTrack]:
        """Get track by ID"""
        return self.mixer.track_manager.get_track(track_id)
    
    def get_all_tracks(self) -> List[AudioTrack]:
        """Get all tracks"""
        return self.mixer.track_manager.get_all_tracks()
    
    def get_track_effects_chain(self, track_id: int) -> Optional[EffectsChain]:
        """Get effects chain for track"""
        return self.track_effects.get(track_id)
    
    def add_track_effect(self, track_id: int, slot: int, effect_type: str, effect_id: str = None) -> bool:
        """Add effect to track"""
        if track_id not in self.track_effects:
            return False
        
        if effect_id is None:
            effect_id = f'{track_id}_{effect_type}_{slot}'
        
        # Create effect
        effect = effect_factory.create_effect(effect_type, effect_id, self.sample_rate, self.buffer_size)
        if not effect:
            return False
        
        # Add to chain
        effects_chain = self.track_effects[track_id]
        return effects_chain.add_effect(slot, effect)
    
    def remove_track_effect(self, track_id: int, slot: int) -> bool:
        """Remove effect from track"""
        if track_id not in self.track_effects:
            return False
        
        effects_chain = self.track_effects[track_id]
        return effects_chain.remove_effect(slot)
    
    def set_effect_parameter(self, track_id: int, effect_id: str, param_name: str, value: float) -> bool:
        """Set effect parameter"""
        if track_id not in self.track_effects:
            return False
        
        effects_chain = self.track_effects[track_id]
        return effects_chain.set_effect_parameter(effect_id, param_name, value)
    
    def process_audio_frame(self, frames: int) -> Tuple[np.ndarray, np.ndarray]:
        """Process a complete audio frame through the engine"""
        start_time = time.perf_counter()
        
        try:
            with self._audio_lock:
                # Create time info
                time_info = {
                    'position_seconds': self.playback_engine.position_seconds,
                    'tempo_bpm': self.playback_engine.tempo_bpm,
                    'buffer_size': frames,
                    'sample_rate': self.sample_rate
                }
                
                # Process through all tracks (handled by mixer)
                input_data = np.zeros((frames, 2), dtype=np.float32)  # Input from audio interface
                main_output, cue_output = self.mixer.process_audio_frame(input_data, frames, time_info)
                
                # Update monitoring for all tracks
                for track in self.mixer.track_manager.get_all_tracks():
                    if hasattr(track, 'output_buffer'):
                        self.monitoring_system.process_track_audio(str(track.track_id), track.output_buffer[:frames])
                
                # Performance monitoring
                processing_time = (time.perf_counter() - start_time) * 1000
                self._update_performance_stats(processing_time)
                
                return main_output, cue_output
                
        except Exception as e:
            print(f"Audio frame processing error: {e}")
            # Return silence on error
            return np.zeros((frames, 2)), np.zeros((frames, 2))
    
    def _update_performance_stats(self, processing_time: float):
        """Update performance statistics and warnings"""
        # Calculate CPU usage
        max_processing_time = (self.buffer_size / self.sample_rate) * 1000  # Max time available
        cpu_usage = (processing_time / max_processing_time) * 100
        
        self.total_cpu_usage = cpu_usage
        
        # Update history
        self.performance_history.append(cpu_usage)
        if len(self.performance_history) > self.max_history_size:
            self.performance_history.pop(0)
        
        # Check for performance warnings
        if cpu_usage > self.performance_warning_threshold:
            self.performance_warning.emit("audio_engine", cpu_usage)
        
        if cpu_usage > self.performance_critical_threshold:
            self.system_overload.emit(cpu_usage)
    
    def _on_track_added(self, track_id: int):
        """Handle track addition"""
        if self.engine_running:
            self.monitoring_system.add_track_monitoring(str(track_id))
    
    def _on_track_removed(self, track_id: int):
        """Handle track removal"""
        self.monitoring_system.remove_track_monitoring(str(track_id))
        if track_id in self.track_effects:
            del self.track_effects[track_id]
    
    def _on_overload_detected(self, track_id: str, level_db: float):
        """Handle audio overload detection"""
        print(f"Audio overload detected on track {track_id}: {level_db:.1f} dB")
    
    def _on_performance_update(self, performance_data: Dict[str, Any]):
        """Handle performance updates"""
        # Store component performance data
        self.component_cpu_usage['monitoring'] = performance_data.get('processing_load_ms', 0.0)
    
    def _on_mixer_overload(self, message: str):
        """Handle mixer overload warnings"""
        print(f"Mixer warning: {message}")
    
    def _on_recording_error(self, error_type: str, message: str):
        """Handle recording errors"""
        print(f"Recording error ({error_type}): {message}")
    
    def set_performance_mode(self, mode: str):
        """Set performance optimization mode"""
        if mode not in ['balanced', 'low_latency', 'high_quality']:
            return False
        
        self.performance_mode = mode
        
        if mode == 'low_latency':
            # Optimize for lowest latency
            self.playback_engine.set_buffer_size(64)  # Smaller buffer
            self.monitoring_system.enable_monitoring(False)  # Disable monitoring
            
        elif mode == 'high_quality':
            # Optimize for highest quality
            self.playback_engine.set_buffer_size(1024)  # Larger buffer
            self.monitoring_system.enable_monitoring(True)
            
        else:  # balanced
            # Balanced performance
            self.playbook_engine.set_buffer_size(512)
            self.monitoring_system.enable_monitoring(True)
        
        return True
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status"""
        return {
            'engine_running': self.engine_running,
            'sample_rate': self.sample_rate,
            'buffer_size': self.buffer_size,
            'performance_mode': self.performance_mode,
            'total_tracks': len(self.mixer.track_manager.get_all_tracks()),
            'active_effects': sum(len(chain.effect_slots) for chain in self.track_effects.values()),
            'total_cpu_usage': self.total_cpu_usage,
            'average_cpu_usage': np.mean(self.performance_history) if self.performance_history else 0.0,
            'playback_info': self.playback_engine.get_playback_info(),
            'mixer_stats': self.mixer.get_performance_stats(),
            'recorder_stats': self.recorder.get_recording_stats(),
            'monitoring_stats': self.monitoring_system.get_performance_stats()
        }
    
    def reset_performance_stats(self):
        """Reset all performance statistics"""
        self.performance_history.clear()
        self.mixer.reset_performance_stats()
        self.recorder.reset_stats()
        self.monitoring_system.reset_performance_stats()
        self.playback_engine.reset_performance_stats()
    
    def __del__(self):
        """Cleanup on destruction"""
        if self.engine_running:
            self.stop_engine()