"""
Professional BPM-Synced Delay Processor
Advanced delay with tempo sync, multiple taps, and modulation
"""

import numpy as np
from typing import Dict, Any
import time

from ..effects_chain import BaseEffect, EffectType


class BPMSyncedDelay(BaseEffect):
    """
    Professional BPM-synchronized delay with multiple taps and modulation.
    Perfect for rhythmic drum processing and creative effects.
    """
    
    def __init__(self, effect_id: str, effect_type: EffectType = EffectType.DELAY,
                 sample_rate: int = 44100, buffer_size: int = 512):
        super().__init__(effect_id, effect_type, sample_rate, buffer_size)
        
        # Maximum delay time (4 bars at 60 BPM = 16 seconds)
        self.max_delay_samples = int(16.0 * sample_rate)
        
        # Delay buffers
        self.delay_buffer_l = np.zeros(self.max_delay_samples, dtype=np.float32)
        self.delay_buffer_r = np.zeros(self.max_delay_samples, dtype=np.float32)
        self.write_index = 0
        
        # Current BPM (will be updated from DAW)
        self.current_bpm = 120.0
        
        # Filter states for feedback filtering
        self.filter_state_l = 0.0
        self.filter_state_r = 0.0
        
        # Modulation LFO
        self.lfo_phase = 0.0
        self.lfo_increment = 0.0
        
        # Performance meters
        self.delay_level_db = -60.0
    
    def _initialize_parameters(self):
        """Initialize delay parameters"""
        self.parameters = {
            'delay_sync': 1.0,          # 0=time, 1=sync
            'delay_time_ms': 250.0,     # Manual delay time
            'delay_note_value': 4.0,    # Note value for sync (1=whole, 2=half, 4=quarter, etc.)
            'delay_dotted': 0.0,        # Dotted note (0=off, 1=on)
            'delay_triplet': 0.0,       # Triplet (0=off, 1=on)
            'feedback': 0.3,            # Delay feedback (0-0.95)
            'feedback_filter': 0.5,     # Feedback filtering (0=dark, 1=bright)
            'stereo_spread': 0.5,       # Stereo delay spread
            'ping_pong': 0.0,           # Ping-pong mode (0=off, 1=on)
            'modulation_depth': 0.02,   # Modulation depth for chorus effect
            'modulation_rate': 0.5,     # Modulation rate in Hz
            'high_cut_hz': 8000.0,      # High cut filter
            'low_cut_hz': 100.0,        # Low cut filter
        }
        
        self.parameter_ranges = {
            'delay_sync': (0.0, 1.0),
            'delay_time_ms': (1.0, 2000.0),
            'delay_note_value': (1.0, 32.0),
            'delay_dotted': (0.0, 1.0),
            'delay_triplet': (0.0, 1.0),
            'feedback': (0.0, 0.95),
            'feedback_filter': (0.0, 1.0),
            'stereo_spread': (0.0, 1.0),
            'ping_pong': (0.0, 1.0),
            'modulation_depth': (0.0, 0.1),
            'modulation_rate': (0.1, 10.0),
            'high_cut_hz': (1000.0, 20000.0),
            'low_cut_hz': (20.0, 1000.0)
        }
        
        self.parameter_defaults = self.parameters.copy()
    
    def set_bpm(self, bpm: float):
        """Update current BPM from DAW"""
        self.current_bpm = max(60.0, min(200.0, bpm))
    
    def process(self, input_data: np.ndarray, frames: int) -> np.ndarray:
        """Process audio through BPM-synced delay"""
        start_time = time.perf_counter()
        
        try:
            # Ensure stereo input
            if input_data.shape[1] == 1:
                stereo_input = np.column_stack([input_data[:, 0], input_data[:, 0]])
            else:
                stereo_input = input_data[:frames].copy()
            
            output_data = np.zeros_like(stereo_input)
            
            # Calculate delay time
            if self.parameters['delay_sync'] > 0.5:
                delay_samples = self._calculate_sync_delay()
            else:
                delay_samples = int(self.parameters['delay_time_ms'] * self.sample_rate / 1000.0)
            
            delay_samples = max(1, min(delay_samples, self.max_delay_samples - 1))
            
            # Update modulation LFO
            self.lfo_increment = 2.0 * np.pi * self.parameters['modulation_rate'] / self.sample_rate
            
            # Process sample by sample
            for i in range(frames):
                left_in = stereo_input[i, 0]
                right_in = stereo_input[i, 1]
                
                # Apply modulation to delay time
                mod_depth = self.parameters['modulation_depth']
                lfo_value = np.sin(self.lfo_phase)
                modulated_delay_l = delay_samples + int(mod_depth * delay_samples * lfo_value)
                modulated_delay_r = delay_samples + int(mod_depth * delay_samples * np.sin(self.lfo_phase + np.pi/4))
                
                # Clamp delay values
                modulated_delay_l = max(1, min(modulated_delay_l, self.max_delay_samples - 1))
                modulated_delay_r = max(1, min(modulated_delay_r, self.max_delay_samples - 1))
                
                # Read from delay buffer with interpolation
                delayed_left = self._read_delay_interpolated(self.delay_buffer_l, modulated_delay_l)
                delayed_right = self._read_delay_interpolated(self.delay_buffer_r, modulated_delay_r)
                
                # Apply stereo spread
                if self.parameters['stereo_spread'] > 0:
                    spread = self.parameters['stereo_spread']
                    # Cross-mix delayed signals for stereo width
                    spread_left = delayed_left * (1 - spread) + delayed_right * spread
                    spread_right = delayed_right * (1 - spread) + delayed_left * spread
                    delayed_left = spread_left
                    delayed_right = spread_right
                
                # Apply ping-pong if enabled
                if self.parameters['ping_pong'] > 0.5:
                    # Swap channels on feedback
                    temp = delayed_left
                    delayed_left = delayed_right
                    delayed_right = temp
                
                # Apply feedback filtering
                delayed_left = self._apply_feedback_filter(delayed_left, 0)
                delayed_right = self._apply_feedback_filter(delayed_right, 1)
                
                # Calculate feedback signals
                feedback_amount = self.parameters['feedback']
                feedback_left = delayed_left * feedback_amount
                feedback_right = delayed_right * feedback_amount
                
                # Write to delay buffer (input + feedback)
                self.delay_buffer_l[self.write_index] = left_in + feedback_left
                self.delay_buffer_r[self.write_index] = right_in + feedback_right
                
                # Generate output
                output_data[i, 0] = delayed_left
                output_data[i, 1] = delayed_right
                
                # Update indices and LFO
                self.write_index = (self.write_index + 1) % self.max_delay_samples
                self.lfo_phase += self.lfo_increment
                if self.lfo_phase >= 2 * np.pi:
                    self.lfo_phase -= 2 * np.pi
                
                # Update meters (every 32 samples)
                if i % 32 == 0:
                    delay_level = max(abs(delayed_left), abs(delayed_right))
                    self.delay_level_db = 20 * np.log10(max(delay_level, 1e-10))
            
            # Apply wet/dry mix
            output_data = self._apply_wet_dry_mix(stereo_input, output_data)
            
            # Performance tracking
            self.processing_time_ms = (time.perf_counter() - start_time) * 1000
            
            return output_data
            
        except Exception as e:
            print(f"Delay error: {e}")
            return input_data[:frames]
    
    def _calculate_sync_delay(self) -> int:
        """Calculate delay time based on BPM synchronization"""
        # Base delay for quarter note
        quarter_note_ms = 60000.0 / self.current_bpm
        
        # Apply note value
        note_value = self.parameters['delay_note_value']
        delay_ms = quarter_note_ms * (4.0 / note_value)
        
        # Apply dotted or triplet timing
        if self.parameters['delay_dotted'] > 0.5:
            delay_ms *= 1.5  # Dotted note = 1.5x
        elif self.parameters['delay_triplet'] > 0.5:
            delay_ms *= 2.0 / 3.0  # Triplet = 2/3x
        
        return int(delay_ms * self.sample_rate / 1000.0)
    
    def _read_delay_interpolated(self, buffer: np.ndarray, delay_samples: float) -> float:
        """Read from delay buffer with linear interpolation"""
        # Calculate read position
        read_pos = (self.write_index - delay_samples) % len(buffer)
        
        # Get integer and fractional parts
        int_pos = int(read_pos)
        frac_pos = read_pos - int_pos
        
        # Get samples for interpolation
        sample1 = buffer[int_pos]
        sample2 = buffer[(int_pos + 1) % len(buffer)]
        
        # Linear interpolation
        return sample1 * (1.0 - frac_pos) + sample2 * frac_pos
    
    def _apply_feedback_filter(self, sample: float, channel: int) -> float:
        """Apply feedback filtering (simple low-pass/high-pass)"""
        filter_amount = self.parameters['feedback_filter']
        
        if channel == 0:  # Left channel
            # Simple one-pole filter
            self.filter_state_l = self.filter_state_l * (1 - filter_amount) + sample * filter_amount
            return self.filter_state_l
        else:  # Right channel
            self.filter_state_r = self.filter_state_r * (1 - filter_amount) + sample * filter_amount
            return self.filter_state_r
    
    def get_drum_presets(self) -> Dict[str, Dict[str, float]]:
        """Get drum-optimized delay presets"""
        return {
            'eighth_note_slap': {
                'delay_sync': 1.0,
                'delay_note_value': 8.0,    # Eighth note
                'feedback': 0.2,
                'wet_dry_mix': 0.3,
                'stereo_spread': 0.3
            },
            'quarter_note_echo': {
                'delay_sync': 1.0,
                'delay_note_value': 4.0,    # Quarter note
                'feedback': 0.4,
                'wet_dry_mix': 0.4,
                'ping_pong': 1.0
            },
            'dotted_eighth': {
                'delay_sync': 1.0,
                'delay_note_value': 8.0,
                'delay_dotted': 1.0,        # Dotted eighth
                'feedback': 0.35,
                'wet_dry_mix': 0.25
            },
            'snare_doubler': {
                'delay_sync': 0.0,          # Manual time
                'delay_time_ms': 30.0,      # 30ms doubling
                'feedback': 0.0,
                'wet_dry_mix': 0.4
            },
            'ambient_wash': {
                'delay_sync': 1.0,
                'delay_note_value': 2.0,    # Half note
                'feedback': 0.6,
                'modulation_depth': 0.05,
                'wet_dry_mix': 0.5
            }
        }
    
    def load_preset(self, preset_name: str) -> bool:
        """Load a drum-optimized preset"""
        presets = self.get_drum_presets()
        if preset_name not in presets:
            return False
        
        preset = presets[preset_name]
        for param, value in preset.items():
            if param in self.parameters:
                self.set_parameter(param, value)
        
        return True
    
    def get_meter_data(self) -> Dict[str, float]:
        """Get delay meter data"""
        return {
            'delay_level_db': self.delay_level_db,
            'current_bpm': self.current_bpm,
            'calculated_delay_ms': self._calculate_sync_delay() * 1000.0 / self.sample_rate
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get delay information"""
        info = super().get_info()
        info['meters'] = self.get_meter_data()
        info['available_presets'] = list(self.get_drum_presets().keys())
        return info