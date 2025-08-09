"""
Professional 3-Band Equalizer
Optimized for drum processing with musical curves and surgical precision
"""

import numpy as np
from typing import Dict, Any, Tuple
import time

from ..effects_chain import BaseEffect, EffectType


class BiquadFilter:
    """
    High-quality biquad filter implementation for EQ bands.
    Supports multiple filter types with smooth parameter changes.
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        
        # Filter coefficients
        self.b0 = 1.0
        self.b1 = 0.0
        self.b2 = 0.0
        self.a1 = 0.0
        self.a2 = 0.0
        
        # Filter state (per channel)
        self.x1_l = 0.0
        self.x2_l = 0.0
        self.y1_l = 0.0
        self.y2_l = 0.0
        
        self.x1_r = 0.0
        self.x2_r = 0.0
        self.y1_r = 0.0
        self.y2_r = 0.0
        
        # Current parameters for smooth changes
        self.frequency = 1000.0
        self.gain_db = 0.0
        self.q_factor = 0.707
        self.filter_type = 'peaking'
    
    def set_peaking(self, frequency: float, gain_db: float, q: float):
        """Set peaking filter parameters"""
        self.frequency = frequency
        self.gain_db = gain_db
        self.q_factor = q
        self.filter_type = 'peaking'
        self._calculate_peaking_coefficients()
    
    def set_highpass(self, frequency: float, q: float = 0.707):
        """Set high-pass filter parameters"""
        self.frequency = frequency
        self.gain_db = 0.0
        self.q_factor = q
        self.filter_type = 'highpass'
        self._calculate_highpass_coefficients()
    
    def set_lowpass(self, frequency: float, q: float = 0.707):
        """Set low-pass filter parameters"""
        self.frequency = frequency
        self.gain_db = 0.0
        self.q_factor = q
        self.filter_type = 'lowpass'
        self._calculate_lowpass_coefficients()
    
    def set_highshelf(self, frequency: float, gain_db: float, q: float = 0.707):
        """Set high-shelf filter parameters"""
        self.frequency = frequency
        self.gain_db = gain_db
        self.q_factor = q
        self.filter_type = 'highshelf'
        self._calculate_highshelf_coefficients()
    
    def set_lowshelf(self, frequency: float, gain_db: float, q: float = 0.707):
        """Set low-shelf filter parameters"""
        self.frequency = frequency
        self.gain_db = gain_db
        self.q_factor = q
        self.filter_type = 'lowshelf'
        self._calculate_lowshelf_coefficients()
    
    def _calculate_peaking_coefficients(self):
        """Calculate peaking filter coefficients"""
        w = 2.0 * np.pi * self.frequency / self.sample_rate
        cos_w = np.cos(w)
        sin_w = np.sin(w)
        A = 10 ** (self.gain_db / 40.0)
        alpha = sin_w / (2.0 * self.q_factor)
        
        # Peaking filter coefficients
        self.b0 = 1.0 + alpha * A
        self.b1 = -2.0 * cos_w
        self.b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        self.a1 = -2.0 * cos_w
        self.a2 = 1.0 - alpha / A
        
        # Normalize by a0
        self.b0 /= a0
        self.b1 /= a0
        self.b2 /= a0
        self.a1 /= a0
        self.a2 /= a0
    
    def _calculate_highpass_coefficients(self):
        """Calculate high-pass filter coefficients"""
        w = 2.0 * np.pi * self.frequency / self.sample_rate
        cos_w = np.cos(w)
        sin_w = np.sin(w)
        alpha = sin_w / (2.0 * self.q_factor)
        
        self.b0 = (1.0 + cos_w) / 2.0
        self.b1 = -(1.0 + cos_w)
        self.b2 = (1.0 + cos_w) / 2.0
        a0 = 1.0 + alpha
        self.a1 = -2.0 * cos_w
        self.a2 = 1.0 - alpha
        
        # Normalize by a0
        self.b0 /= a0
        self.b1 /= a0
        self.b2 /= a0
        self.a1 /= a0
        self.a2 /= a0
    
    def _calculate_lowpass_coefficients(self):
        """Calculate low-pass filter coefficients"""
        w = 2.0 * np.pi * self.frequency / self.sample_rate
        cos_w = np.cos(w)
        sin_w = np.sin(w)
        alpha = sin_w / (2.0 * self.q_factor)
        
        self.b0 = (1.0 - cos_w) / 2.0
        self.b1 = 1.0 - cos_w
        self.b2 = (1.0 - cos_w) / 2.0
        a0 = 1.0 + alpha
        self.a1 = -2.0 * cos_w
        self.a2 = 1.0 - alpha
        
        # Normalize by a0
        self.b0 /= a0
        self.b1 /= a0
        self.b2 /= a0
        self.a1 /= a0
        self.a2 /= a0
    
    def _calculate_highshelf_coefficients(self):
        """Calculate high-shelf filter coefficients"""
        w = 2.0 * np.pi * self.frequency / self.sample_rate
        cos_w = np.cos(w)
        sin_w = np.sin(w)
        A = 10 ** (self.gain_db / 40.0)
        beta = np.sqrt(A) / self.q_factor
        
        self.b0 = A * ((A + 1) + (A - 1) * cos_w + beta * sin_w)
        self.b1 = -2 * A * ((A - 1) + (A + 1) * cos_w)
        self.b2 = A * ((A + 1) + (A - 1) * cos_w - beta * sin_w)
        a0 = (A + 1) - (A - 1) * cos_w + beta * sin_w
        self.a1 = 2 * ((A - 1) - (A + 1) * cos_w)
        self.a2 = (A + 1) - (A - 1) * cos_w - beta * sin_w
        
        # Normalize by a0
        self.b0 /= a0
        self.b1 /= a0
        self.b2 /= a0
        self.a1 /= a0
        self.a2 /= a0
    
    def _calculate_lowshelf_coefficients(self):
        """Calculate low-shelf filter coefficients"""
        w = 2.0 * np.pi * self.frequency / self.sample_rate
        cos_w = np.cos(w)
        sin_w = np.sin(w)
        A = 10 ** (self.gain_db / 40.0)
        beta = np.sqrt(A) / self.q_factor
        
        self.b0 = A * ((A + 1) - (A - 1) * cos_w + beta * sin_w)
        self.b1 = 2 * A * ((A - 1) - (A + 1) * cos_w)
        self.b2 = A * ((A + 1) - (A - 1) * cos_w - beta * sin_w)
        a0 = (A + 1) + (A - 1) * cos_w + beta * sin_w
        self.a1 = -2 * ((A - 1) + (A + 1) * cos_w)
        self.a2 = (A + 1) + (A - 1) * cos_w - beta * sin_w
        
        # Normalize by a0
        self.b0 /= a0
        self.b1 /= a0
        self.b2 /= a0
        self.a1 /= a0
        self.a2 /= a0
    
    def process_sample(self, input_l: float, input_r: float) -> Tuple[float, float]:
        """Process a single stereo sample"""
        # Left channel
        output_l = (self.b0 * input_l + 
                   self.b1 * self.x1_l + 
                   self.b2 * self.x2_l - 
                   self.a1 * self.y1_l - 
                   self.a2 * self.y2_l)
        
        # Update left channel state
        self.x2_l = self.x1_l
        self.x1_l = input_l
        self.y2_l = self.y1_l
        self.y1_l = output_l
        
        # Right channel
        output_r = (self.b0 * input_r + 
                   self.b1 * self.x1_r + 
                   self.b2 * self.x2_r - 
                   self.a1 * self.y1_r - 
                   self.a2 * self.y2_r)
        
        # Update right channel state
        self.x2_r = self.x1_r
        self.x1_r = input_r
        self.y2_r = self.y1_r
        self.y1_r = output_r
        
        return output_l, output_r
    
    def reset(self):
        """Reset filter state"""
        self.x1_l = self.x2_l = self.y1_l = self.y2_l = 0.0
        self.x1_r = self.x2_r = self.y1_r = self.y2_r = 0.0


class ThreeBandEQ(BaseEffect):
    """
    Professional 3-band equalizer optimized for drum processing.
    Features high-quality biquad filters with musical curves.
    """
    
    def __init__(self, effect_id: str, effect_type: EffectType = EffectType.EQ,
                 sample_rate: int = 44100, buffer_size: int = 512):
        super().__init__(effect_id, effect_type, sample_rate, buffer_size)
        
        # Create filter instances
        self.low_filter = BiquadFilter(sample_rate)
        self.mid_filter = BiquadFilter(sample_rate)
        self.high_filter = BiquadFilter(sample_rate)
        
        # High-pass and low-pass filters for band separation
        self.hp_filter = BiquadFilter(sample_rate)
        self.lp_filter = BiquadFilter(sample_rate)
        
        # Frequency response analysis
        self.frequency_response_cache = {}
        self.response_cache_valid = False
        
        # Meters for each band
        self.low_level_db = -60.0
        self.mid_level_db = -60.0
        self.high_level_db = -60.0
        
        # Initialize filters
        self._update_filter_parameters()
    
    def _initialize_parameters(self):
        """Initialize EQ parameters optimized for drums"""
        self.parameters = {
            # Low band (shelf filter) - perfect for kick drum
            'low_gain_db': 0.0,         # Low frequency gain
            'low_freq_hz': 80.0,        # Low shelf frequency
            'low_q': 0.707,             # Low band Q factor
            
            # Mid band (peaking filter) - great for snare body
            'mid_gain_db': 0.0,         # Mid frequency gain
            'mid_freq_hz': 1000.0,      # Mid peak frequency
            'mid_q': 1.0,               # Mid band Q factor
            
            # High band (shelf filter) - perfect for hi-hat and cymbals
            'high_gain_db': 0.0,        # High frequency gain
            'high_freq_hz': 8000.0,     # High shelf frequency
            'high_q': 0.707,            # High band Q factor
            
            # Global settings
            'output_gain_db': 0.0,      # Output gain compensation
            'band_separation': 1.0,     # Band separation (0=overlap, 1=clean separation)
            'analog_modeling': 0.5,     # Analog modeling amount
            'phase_linear': 0.0,        # Linear phase mode (0=minimum phase, 1=linear phase)
        }
        
        self.parameter_ranges = {
            'low_gain_db': (-20.0, 20.0),
            'low_freq_hz': (20.0, 200.0),
            'low_q': (0.1, 10.0),
            'mid_gain_db': (-20.0, 20.0),
            'mid_freq_hz': (200.0, 8000.0),
            'mid_q': (0.1, 10.0),
            'high_gain_db': (-20.0, 20.0),
            'high_freq_hz': (2000.0, 20000.0),
            'high_q': (0.1, 10.0),
            'output_gain_db': (-20.0, 20.0),
            'band_separation': (0.0, 1.0),
            'analog_modeling': (0.0, 1.0),
            'phase_linear': (0.0, 1.0)
        }
        
        self.parameter_defaults = self.parameters.copy()
    
    def process(self, input_data: np.ndarray, frames: int) -> np.ndarray:
        """Process audio through the 3-band EQ"""
        start_time = time.perf_counter()
        
        try:
            # Ensure stereo input
            if input_data.shape[1] == 1:
                stereo_input = np.column_stack([input_data[:, 0], input_data[:, 0]])
            else:
                stereo_input = input_data[:frames].copy()
            
            output_data = np.zeros_like(stereo_input)
            
            # Update filter parameters if needed
            self._update_filter_parameters()
            
            # Process sample by sample for maximum quality
            for i in range(frames):
                left_sample = stereo_input[i, 0]
                right_sample = stereo_input[i, 1]
                
                # Apply each band filter
                left_low, right_low = self.low_filter.process_sample(left_sample, right_sample)
                left_mid, right_mid = self.mid_filter.process_sample(left_sample, right_sample)
                left_high, right_high = self.high_filter.process_sample(left_sample, right_sample)
                
                # Combine processed bands
                if self.parameters['band_separation'] < 1.0:
                    # Overlapping bands (more musical)
                    output_left = left_low + left_mid + left_high - 2 * left_sample
                    output_right = right_low + right_mid + right_high - 2 * right_sample
                else:
                    # Clean band separation
                    output_left = left_low + left_mid + left_high
                    output_right = right_low + right_mid + right_high
                
                # Apply output gain
                output_gain = 10 ** (self.parameters['output_gain_db'] / 20.0)
                output_left *= output_gain
                output_right *= output_gain
                
                # Apply analog modeling if enabled
                if self.parameters['analog_modeling'] > 0:
                    output_left, output_right = self._apply_analog_modeling(
                        output_left, output_right, self.parameters['analog_modeling'])
                
                output_data[i, 0] = output_left
                output_data[i, 1] = output_right
                
                # Update band level meters (every 32 samples)
                if i % 32 == 0:
                    self._update_band_meters(left_low, left_mid, left_high)
            
            # Apply wet/dry mix
            if self.wet_dry_mix < 1.0:
                output_data = self._apply_wet_dry_mix(stereo_input, output_data)
            
            # Performance tracking
            self.processing_time_ms = (time.perf_counter() - start_time) * 1000
            
            return output_data
            
        except Exception as e:
            print(f"EQ error: {e}")
            return input_data[:frames]
    
    def _update_filter_parameters(self):
        """Update all filter parameters"""
        # Low band (shelf filter)
        self.low_filter.set_lowshelf(
            self.parameters['low_freq_hz'],
            self.parameters['low_gain_db'],
            self.parameters['low_q']
        )
        
        # Mid band (peaking filter)
        self.mid_filter.set_peaking(
            self.parameters['mid_freq_hz'],
            self.parameters['mid_gain_db'],
            self.parameters['mid_q']
        )
        
        # High band (shelf filter)
        self.high_filter.set_highshelf(
            self.parameters['high_freq_hz'],
            self.parameters['high_gain_db'],
            self.parameters['high_q']
        )
        
        # Invalidate frequency response cache
        self.response_cache_valid = False
    
    def _apply_analog_modeling(self, left: float, right: float, amount: float) -> Tuple[float, float]:
        """Apply subtle analog modeling for warmth"""
        # Simple saturation curve
        drive = 1.0 + amount * 0.1
        
        # Soft clipping with even harmonics
        left_sat = np.tanh(left * drive) / drive
        right_sat = np.tanh(right * drive) / drive
        
        # Blend with dry signal
        left_out = left * (1 - amount) + left_sat * amount
        right_out = right * (1 - amount) + right_sat * amount
        
        return left_out, right_out
    
    def _update_band_meters(self, low_sample: float, mid_sample: float, high_sample: float):
        """Update frequency band level meters"""
        # Convert to dB with smoothing
        low_db = 20 * np.log10(abs(low_sample) + 1e-10)
        mid_db = 20 * np.log10(abs(mid_sample) + 1e-10)
        high_db = 20 * np.log10(abs(high_sample) + 1e-10)
        
        # Smooth meter updates
        smooth_factor = 0.9
        self.low_level_db = self.low_level_db * smooth_factor + low_db * (1 - smooth_factor)
        self.mid_level_db = self.mid_level_db * smooth_factor + mid_db * (1 - smooth_factor)
        self.high_level_db = self.high_level_db * smooth_factor + high_db * (1 - smooth_factor)
    
    def calculate_frequency_response(self, frequencies: np.ndarray) -> np.ndarray:
        """Calculate frequency response for visualization"""
        if self.response_cache_valid:
            return self.frequency_response_cache.get('response', np.ones_like(frequencies))
        
        # This is a simplified calculation - full implementation would
        # calculate the actual filter response
        response = np.ones_like(frequencies, dtype=complex)
        
        for freq in frequencies:
            # Calculate response for each band
            low_response = self._calculate_shelf_response(
                freq, self.parameters['low_freq_hz'], 
                self.parameters['low_gain_db'], 'low')
            
            mid_response = self._calculate_peaking_response(
                freq, self.parameters['mid_freq_hz'],
                self.parameters['mid_gain_db'], self.parameters['mid_q'])
            
            high_response = self._calculate_shelf_response(
                freq, self.parameters['high_freq_hz'],
                self.parameters['high_gain_db'], 'high')
            
            # Combine responses
            total_response = low_response * mid_response * high_response
            idx = np.where(frequencies == freq)[0]
            if len(idx) > 0:
                response[idx[0]] = total_response
        
        self.frequency_response_cache['response'] = response
        self.response_cache_valid = True
        
        return response
    
    def _calculate_shelf_response(self, freq: float, shelf_freq: float, 
                                gain_db: float, shelf_type: str) -> complex:
        """Calculate shelf filter response at given frequency"""
        # Simplified shelf response calculation
        if shelf_type == 'low':
            if freq <= shelf_freq:
                return 10 ** (gain_db / 20.0)
            else:
                # Smooth transition
                ratio = freq / shelf_freq
                transition = 1.0 / (1.0 + ratio)
                return 1.0 + (10 ** (gain_db / 20.0) - 1.0) * transition
        else:  # high shelf
            if freq >= shelf_freq:
                return 10 ** (gain_db / 20.0)
            else:
                # Smooth transition
                ratio = shelf_freq / freq
                transition = 1.0 / (1.0 + ratio)
                return 1.0 + (10 ** (gain_db / 20.0) - 1.0) * transition
    
    def _calculate_peaking_response(self, freq: float, peak_freq: float, 
                                  gain_db: float, q: float) -> complex:
        """Calculate peaking filter response at given frequency"""
        # Simplified peaking response
        freq_ratio = freq / peak_freq
        bandwidth = 1.0 / q
        
        if abs(np.log2(freq_ratio)) <= bandwidth / 2:
            # Within the peak bandwidth
            distance = abs(np.log2(freq_ratio)) / (bandwidth / 2)
            gain_factor = 10 ** (gain_db / 20.0)
            return 1.0 + (gain_factor - 1.0) * (1.0 - distance)
        else:
            return 1.0
    
    def get_drum_presets(self) -> Dict[str, Dict[str, float]]:
        """Get drum-optimized EQ presets"""
        return {
            'kick_punch': {
                'low_gain_db': 3.0,
                'low_freq_hz': 60.0,
                'mid_gain_db': -2.0,
                'mid_freq_hz': 400.0,
                'high_gain_db': 1.0,
                'high_freq_hz': 8000.0
            },
            'snare_crack': {
                'low_gain_db': -1.0,
                'low_freq_hz': 100.0,
                'mid_gain_db': 4.0,
                'mid_freq_hz': 2500.0,
                'high_gain_db': 2.0,
                'high_freq_hz': 10000.0
            },
            'hihat_sparkle': {
                'low_gain_db': -6.0,
                'low_freq_hz': 200.0,
                'mid_gain_db': -1.0,
                'mid_freq_hz': 1000.0,
                'high_gain_db': 3.0,
                'high_freq_hz': 12000.0
            },
            'tom_body': {
                'low_gain_db': 2.0,
                'low_freq_hz': 80.0,
                'mid_gain_db': 3.0,
                'mid_freq_hz': 200.0,
                'high_gain_db': 1.0,
                'high_freq_hz': 6000.0
            },
            'overhead_air': {
                'low_gain_db': -3.0,
                'low_freq_hz': 100.0,
                'mid_gain_db': 0.0,
                'mid_freq_hz': 1000.0,
                'high_gain_db': 4.0,
                'high_freq_hz': 15000.0
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
        """Get EQ band meter data"""
        return {
            'low_level_db': self.low_level_db,
            'mid_level_db': self.mid_level_db,
            'high_level_db': self.high_level_db
        }
    
    def reset_meters(self):
        """Reset meter values"""
        self.low_level_db = -60.0
        self.mid_level_db = -60.0
        self.high_level_db = -60.0
    
    def _on_parameter_changed(self, param_name: str, value: float):
        """Handle parameter changes"""
        self.response_cache_valid = False
        self._update_filter_parameters()
    
    def get_info(self) -> Dict[str, Any]:
        """Get EQ information including meters and presets"""
        info = super().get_info()
        info['meters'] = self.get_meter_data()
        info['available_presets'] = list(self.get_drum_presets().keys())
        return info