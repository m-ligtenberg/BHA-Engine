from PyQt6.QtCore import QObject, pyqtSignal, QTimer, QEasingCurve, QPropertyAnimation
from typing import Optional, List, Tuple
import time
import math

class RealTimeBPMSync(QObject):
    """Handles real-time BPM synchronization from Rhythm Wolf hardware"""

    tempo_changed = pyqtSignal(float)  # New BPM value

    def __init__(self, daw_instance):
        super().__init__()
        self.daw = daw_instance
        self.last_bpm_value = 120.0
        self.bpm_smoothing = True
        self.bpm_range = (60, 200)  # Min/Max BPM range
        self.bpm_enabled = True
        
        # Enhanced smoothing parameters
        self.smoothing_factor = 0.15  # Base smoothing amount
        self.adaptive_smoothing = True  # Adapt smoothing to rate of change
        self.bpm_history = []  # Track recent BPM values
        self.history_size = 10
        
        # Performance optimization
        self.last_update_time = 0
        self.min_update_interval_ms = 25  # Maximum 40Hz updates
        self.bpm_change_threshold = 0.3  # Minimum change to trigger update
        
        # Live transition features
        self.transition_enabled = True
        self.transition_duration_ms = 150  # Smooth transition time
        self.sync_mode = 'immediate'  # 'immediate', 'beat_sync', 'bar_sync'
        
        # Tempo automation and curves
        self.bpm_automation = None
        self.tempo_curve_active = False
        
        # Performance monitoring
        self.update_count = 0
        self.average_latency_ms = 0
        
        # Create smooth transition animation
        self.bpm_animation = QPropertyAnimation(self, b"currentBPM")
        self.bpm_animation.setDuration(self.transition_duration_ms)
        self.bpm_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        
        # Tempo change detection timer
        self.tempo_stability_timer = QTimer()
        self.tempo_stability_timer.setSingleShot(True)
        self.tempo_stability_timer.timeout.connect(self._on_tempo_stable)

    def set_bpm_sync_enabled(self, enabled):
        """Enable/disable BPM sync from hardware"""
        self.bpm_enabled = enabled

    def set_bpm_range(self, min_bpm, max_bpm):
        """Set BPM range for CC mapping"""
        self.bpm_range = (min_bpm, max_bpm)

    def handle_bpm_change(self, cc_value):
        """Convert CC value to BPM and update DAW tempo"""
        if not self.bpm_enabled:
            return

        # Convert MIDI CC (0-127) to BPM range
        bpm_min, bpm_max = self.bpm_range
        bpm_range_size = bpm_max - bpm_min

        # Calculate new BPM value
        new_bpm = bpm_min + (cc_value / 127.0) * bpm_range_size

        # Optional smoothing to prevent jitter
        if self.bpm_smoothing:
            new_bpm = self.smooth_bpm_change(new_bpm)

        # Update DAW tempo if change is significant
        if abs(new_bpm - self.last_bpm_value) > 0.5:
            self.update_daw_tempo(new_bpm)
            self.last_bpm_value = new_bpm

    def smooth_bpm_change(self, target_bpm):
        """Smooth BPM changes to prevent rapid fluctuations"""
        smoothing_factor = 0.15
        return (self.last_bpm_value * (1 - smoothing_factor) + 
                target_bpm * smoothing_factor)

    def update_daw_tempo(self, new_bpm):
        """Update the DAW's master tempo"""
        rounded_bpm = round(new_bpm, 1)

        # Update sequencer tempo
        self.daw.set_master_tempo(rounded_bpm)

        # Update all time-based effects (delays, reverbs)
        self.daw.sync_effects_to_tempo(rounded_bpm)

        # Emit signal for GUI updates
        self.tempo_changed.emit(rounded_bpm)

        print(f"🎵 BPM Updated: {rounded_bpm}")

    def handle_live_bpm_changes(self, new_bpm):
        """Handle BPM changes during active playback/recording"""

        if self.daw.is_playing():
            # Smooth tempo transition to avoid audio glitches
            self.daw.schedule_tempo_change(new_bpm, fade_time=0.1)

            # Update all time-synced effects gradually
            self.daw.gradual_effects_sync(new_bpm)

            # Maintain pattern sync with new tempo
            self.daw.resync_pattern_playback(new_bpm)

        else:
            # Immediate tempo change when stopped
            self.daw.set_master_tempo(new_bpm)

    def get_current_bpm(self):
        """Get current BPM value"""
        return self.last_bpm_value
        
    def _enhanced_smoothing(self, target_bpm) -> float:
        """Enhanced BPM smoothing with adaptive behavior"""
        if not self.bpm_history:
            return target_bpm
            
        # Calculate rate of change
        if len(self.bpm_history) >= 2:
            recent_change = abs(self.bpm_history[-1] - self.bpm_history[-2])
            
            # Adapt smoothing factor based on rate of change
            if self.adaptive_smoothing:
                if recent_change > 2.0:  # Rapid change
                    factor = self.smoothing_factor * 0.5  # More smoothing
                elif recent_change < 0.5:  # Slow change
                    factor = self.smoothing_factor * 2.0  # Less smoothing
                else:
                    factor = self.smoothing_factor
            else:
                factor = self.smoothing_factor
                
            # Limit factor to reasonable range
            factor = max(0.05, min(0.5, factor))
        else:
            factor = self.smoothing_factor
            
        return (self.last_bpm_value * (1 - factor) + target_bpm * factor)
        
    def _execute_tempo_change(self, new_bpm: float):
        """Execute tempo change with enhanced features"""
        rounded_bpm = round(new_bpm, 1)
        
        # Determine sync timing based on mode
        if self.sync_mode == 'immediate' or not self.daw.is_playing():
            self._immediate_tempo_change(rounded_bpm)
        elif self.sync_mode == 'beat_sync':
            self._beat_synced_tempo_change(rounded_bpm)
        elif self.sync_mode == 'bar_sync':
            self._bar_synced_tempo_change(rounded_bpm)
            
        self.last_bpm_value = rounded_bpm
        
    def _immediate_tempo_change(self, bpm: float):
        """Apply tempo change immediately"""
        if self.transition_enabled and self.daw.is_playing():
            # Smooth transition during playback
            self.daw.schedule_tempo_change(bpm, fade_time=self.transition_duration_ms/1000)
        else:
            # Immediate change when stopped
            self.daw.set_master_tempo(bpm)

        # Update all time-based effects
        self.daw.sync_effects_to_tempo(bpm)

        # Emit signal for GUI updates
        self.tempo_changed.emit(bpm)

        print(f"🎵 BPM Updated: {bpm} (immediate)")
        
    def _beat_synced_tempo_change(self, bpm: float):
        """Schedule tempo change on next beat"""
        print(f"🎵 BPM Change Scheduled: {bpm} (next beat)")
        # Implementation would depend on DAW's beat tracking
        # For now, fall back to immediate
        self._immediate_tempo_change(bpm)
        
    def _bar_synced_tempo_change(self, bpm: float):
        """Schedule tempo change on next bar"""
        print(f"🎵 BPM Change Scheduled: {bpm} (next bar)")
        # Implementation would depend on DAW's bar tracking
        # For now, fall back to immediate
        self._immediate_tempo_change(bpm)
        
    def _on_tempo_stable(self):
        """Called when tempo has been stable for a period"""
        print(f"🎵 Tempo stabilized at {self.last_bpm_value} BPM")
        
    def set_sync_mode(self, mode: str):
        """Set tempo sync mode: 'immediate', 'beat_sync', 'bar_sync'"""
        valid_modes = ['immediate', 'beat_sync', 'bar_sync']
        if mode in valid_modes:
            self.sync_mode = mode
            print(f"🎵 Sync mode set to: {mode}")
            
    def set_smoothing_parameters(self, factor: float, adaptive: bool = True):
        """Configure smoothing behavior"""
        self.smoothing_factor = max(0.01, min(0.9, factor))
        self.adaptive_smoothing = adaptive
        
    def set_transition_duration(self, duration_ms: int):
        """Set smooth transition duration"""
        self.transition_duration_ms = max(50, min(1000, duration_ms))
        self.bpm_animation.setDuration(duration_ms)
        
    def enable_tempo_automation(self, enabled: bool):
        """Enable/disable tempo automation recording"""
        if enabled and not self.bpm_automation:
            self.bpm_automation = []
        elif not enabled:
            self.bpm_automation = None
            
    def get_performance_stats(self) -> dict:
        """Get BPM sync performance statistics"""
        return {
            'update_count': self.update_count,
            'average_latency_ms': round(self.average_latency_ms, 2),
            'current_bpm': self.last_bpm_value,
            'sync_mode': self.sync_mode,
            'smoothing_enabled': self.bpm_smoothing,
            'history_length': len(self.bpm_history),
            'enabled': self.bpm_enabled
        }
        
    def reset_bpm_history(self):
        """Clear BPM history and reset smoothing"""
        self.bpm_history.clear()
        self.update_count = 0
        self.average_latency_ms = 0
        print("🎵 BPM history reset")