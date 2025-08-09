"""
Professional Dynamic Range Compressor
High-quality compressor optimized for drum and instrument processing
"""

import numpy as np
from typing import Dict, Any
import time

from ..effects_chain import BaseEffect, EffectType


class Compressor(BaseEffect):
    """
    Professional dynamic range compressor with advanced features:
    - Variable knee compression
    - Lookahead processing
    - Multiple compression modes
    - Sidechain filtering
    - Auto-makeup gain
    """
    
    def __init__(self, effect_id: str, effect_type: EffectType = EffectType.COMPRESSOR,
                 sample_rate: int = 44100, buffer_size: int = 512):
        super().__init__(effect_id, effect_type, sample_rate, buffer_size)
        
        # Compressor-specific state
        self.envelope_follower_l = 0.0
        self.envelope_follower_r = 0.0
        self.gain_reduction_l = 0.0
        self.gain_reduction_r = 0.0
        
        # Lookahead delay line
        self.lookahead_samples = int(0.005 * sample_rate)  # 5ms lookahead
        self.delay_buffer_l = np.zeros(self.lookahead_samples, dtype=np.float32)
        self.delay_buffer_r = np.zeros(self.lookahead_samples, dtype=np.float32)
        self.delay_index = 0
        
        # Peak detection history
        self.peak_history_size = int(0.01 * sample_rate)  # 10ms peak detection
        self.peak_buffer_l = np.zeros(self.peak_history_size, dtype=np.float32)
        self.peak_buffer_r = np.zeros(self.peak_history_size, dtype=np.float32)
        self.peak_index = 0
        
        # Smoothing filters for gain reduction
        self.gain_smooth_l = 0.0
        self.gain_smooth_r = 0.0
        
        # Meters for GUI feedback
        self.input_peak_db = -60.0
        self.output_peak_db = -60.0
        self.gain_reduction_db = 0.0
        self.compression_ratio_actual = 1.0
    
    def _initialize_parameters(self):
        """Initialize compressor parameters"""
        self.parameters = {
            'threshold_db': -12.0,      # Compression threshold
            'ratio': 4.0,               # Compression ratio (1:1 to 20:1)
            'knee_db': 2.0,             # Knee width in dB
            'attack_ms': 3.0,           # Attack time in milliseconds
            'release_ms': 100.0,        # Release time in milliseconds
            'makeup_gain_db': 0.0,      # Manual makeup gain
            'auto_makeup': 1.0,         # Auto makeup gain (0=off, 1=on)
            'lookahead_ms': 5.0,        # Lookahead time
            'mix': 1.0,                 # Wet/dry mix
            'sidechain_highpass_hz': 0.0,  # Sidechain high-pass filter
            'stereo_link': 1.0,         # Stereo link (0=independent, 1=linked)
            'peak_rms_blend': 0.0       # 0=peak, 1=RMS detection
        }
        
        self.parameter_ranges = {
            'threshold_db': (-60.0, 0.0),
            'ratio': (1.0, 20.0),
            'knee_db': (0.0, 20.0),
            'attack_ms': (0.1, 100.0),
            'release_ms': (10.0, 5000.0),
            'makeup_gain_db': (-20.0, 20.0),
            'auto_makeup': (0.0, 1.0),
            'lookahead_ms': (0.0, 10.0),
            'mix': (0.0, 1.0),
            'sidechain_highpass_hz': (0.0, 500.0),
            'stereo_link': (0.0, 1.0),
            'peak_rms_blend': (0.0, 1.0)
        }
        
        self.parameter_defaults = self.parameters.copy()
    
    def process(self, input_data: np.ndarray, frames: int) -> np.ndarray:
        """Process audio through the compressor"""
        start_time = time.perf_counter()
        
        try:
            # Ensure we have stereo input
            if input_data.shape[1] == 1:
                # Convert mono to stereo
                stereo_input = np.column_stack([input_data[:, 0], input_data[:, 0]])
            else:
                stereo_input = input_data[:frames].copy()
            
            # Process frame by frame for maximum accuracy
            output_data = np.zeros_like(stereo_input)
            
            for i in range(frames):
                # Get current samples
                left_sample = stereo_input[i, 0]
                right_sample = stereo_input[i, 1]
                
                # Store in delay buffer (lookahead)
                self.delay_buffer_l[self.delay_index] = left_sample
                self.delay_buffer_r[self.delay_index] = right_sample
                
                # Get delayed samples for processing
                delayed_index = (self.delay_index - self.lookahead_samples) % self.lookahead_samples
                delayed_left = self.delay_buffer_l[delayed_index]
                delayed_right = self.delay_buffer_r[delayed_index]
                
                # Peak detection with history
                self.peak_buffer_l[self.peak_index] = abs(left_sample)
                self.peak_buffer_r[self.peak_index] = abs(right_sample)
                
                # Calculate detection levels
                if self.parameters['peak_rms_blend'] > 0:
                    # RMS detection
                    rms_l = np.sqrt(np.mean(self.peak_buffer_l ** 2))
                    rms_r = np.sqrt(np.mean(self.peak_buffer_r ** 2))
                    detect_l = rms_l
                    detect_r = rms_r
                else:
                    # Peak detection
                    detect_l = np.max(self.peak_buffer_l)
                    detect_r = np.max(self.peak_buffer_r)
                
                # Blend peak and RMS
                if self.parameters['peak_rms_blend'] > 0:
                    peak_l = np.max(self.peak_buffer_l)
                    peak_r = np.max(self.peak_buffer_r)
                    blend = self.parameters['peak_rms_blend']
                    detect_l = peak_l * (1 - blend) + detect_l * blend
                    detect_r = peak_r * (1 - blend) + detect_r * blend
                
                # Stereo linking
                if self.parameters['stereo_link'] > 0:
                    link_amount = self.parameters['stereo_link']
                    max_detect = max(detect_l, detect_r)
                    detect_l = detect_l * (1 - link_amount) + max_detect * link_amount
                    detect_r = detect_r * (1 - link_amount) + max_detect * link_amount
                
                # Convert to dB
                detect_db_l = 20 * np.log10(max(detect_l, 1e-10))
                detect_db_r = 20 * np.log10(max(detect_r, 1e-10))
                
                # Calculate compression
                gain_reduction_l = self._calculate_compression(detect_db_l)
                gain_reduction_r = self._calculate_compression(detect_db_r)
                
                # Apply attack/release smoothing
                gain_reduction_l = self._apply_attack_release(gain_reduction_l, 0)
                gain_reduction_r = self._apply_attack_release(gain_reduction_r, 1)
                
                # Convert back to linear gain
                gain_linear_l = 10 ** (gain_reduction_l / 20)
                gain_linear_r = 10 ** (gain_reduction_r / 20)
                
                # Apply compression to delayed samples
                compressed_left = delayed_left * gain_linear_l
                compressed_right = delayed_right * gain_linear_r
                
                # Apply makeup gain
                makeup_gain = self._calculate_makeup_gain()
                makeup_linear = 10 ** (makeup_gain / 20)
                
                compressed_left *= makeup_linear
                compressed_right *= makeup_linear
                
                # Store output
                output_data[i, 0] = compressed_left
                output_data[i, 1] = compressed_right
                
                # Update indices
                self.delay_index = (self.delay_index + 1) % self.lookahead_samples
                self.peak_index = (self.peak_index + 1) % self.peak_history_size
                
                # Update meters (every 64 samples to reduce CPU load)
                if i % 64 == 0:
                    self.input_peak_db = max(detect_db_l, detect_db_r)
                    self.output_peak_db = 20 * np.log10(max(abs(compressed_left), abs(compressed_right)) + 1e-10)
                    self.gain_reduction_db = max(abs(gain_reduction_l), abs(gain_reduction_r))
            
            # Apply wet/dry mix
            if self.parameters['mix'] < 1.0:
                output_data = self._apply_wet_dry_mix(stereo_input, output_data)
            
            # Performance tracking
            self.processing_time_ms = (time.perf_counter() - start_time) * 1000
            
            return output_data
            
        except Exception as e:
            print(f"Compressor error: {e}")
            return input_data[:frames]
    
    def _calculate_compression(self, input_db: float) -> float:
        """Calculate gain reduction for given input level"""
        threshold = self.parameters['threshold_db']
        ratio = self.parameters['ratio']
        knee = self.parameters['knee_db']
        
        if input_db <= threshold - knee / 2:
            # Below knee - no compression
            return 0.0
        elif input_db >= threshold + knee / 2:
            # Above knee - full compression
            overshoot = input_db - threshold
            gain_reduction = overshoot - (overshoot / ratio)
            return -gain_reduction  # Negative for attenuation
        else:
            # Within knee - soft compression
            knee_ratio = (input_db - threshold + knee / 2) / knee
            # Smooth curve within knee
            knee_gain = knee_ratio * knee_ratio
            overshoot = input_db - threshold
            gain_reduction = knee_gain * (overshoot - (overshoot / ratio))
            return -gain_reduction
    
    def _apply_attack_release(self, target_gain: float, channel: int) -> float:
        """Apply attack/release envelope to gain reduction"""
        # Convert times to coefficients
        attack_coeff = self._time_to_coefficient(self.parameters['attack_ms'] / 1000.0)
        release_coeff = self._time_to_coefficient(self.parameters['release_ms'] / 1000.0)
        
        # Get current envelope value
        current_envelope = self.envelope_follower_l if channel == 0 else self.envelope_follower_r
        
        # Apply attack or release
        if target_gain < current_envelope:  # Attack (gain reduction increasing)
            new_envelope = current_envelope * attack_coeff + target_gain * (1 - attack_coeff)
        else:  # Release (gain reduction decreasing)
            new_envelope = current_envelope * release_coeff + target_gain * (1 - release_coeff)
        
        # Store back
        if channel == 0:
            self.envelope_follower_l = new_envelope
        else:
            self.envelope_follower_r = new_envelope
        
        return new_envelope
    
    def _time_to_coefficient(self, time_seconds: float) -> float:
        """Convert time constant to filter coefficient"""
        if time_seconds <= 0:
            return 0.0
        return np.exp(-1.0 / (time_seconds * self.sample_rate))
    
    def _calculate_makeup_gain(self) -> float:
        """Calculate automatic makeup gain"""
        makeup_gain = self.parameters['makeup_gain_db']
        
        if self.parameters['auto_makeup'] > 0:
            # Simple auto-makeup: compensate for average gain reduction
            auto_gain = abs(self.gain_reduction_db) * 0.5  # Conservative compensation
            makeup_gain += auto_gain * self.parameters['auto_makeup']
        
        return makeup_gain
    
    def _on_parameter_changed(self, param_name: str, value: float):
        """Handle parameter changes"""
        if param_name == 'lookahead_ms':
            # Update lookahead delay line size
            old_size = self.lookahead_samples
            new_size = int((value / 1000.0) * self.sample_rate)
            
            if new_size != old_size:
                # Resize delay buffers
                old_buffer_l = self.delay_buffer_l.copy()
                old_buffer_r = self.delay_buffer_r.copy()
                
                self.delay_buffer_l = np.zeros(new_size, dtype=np.float32)
                self.delay_buffer_r = np.zeros(new_size, dtype=np.float32)
                
                # Copy existing data if possible
                copy_size = min(old_size, new_size)
                if copy_size > 0:
                    self.delay_buffer_l[:copy_size] = old_buffer_l[:copy_size]
                    self.delay_buffer_r[:copy_size] = old_buffer_r[:copy_size]
                
                self.lookahead_samples = new_size
                self.delay_index = 0
    
    def get_meter_data(self) -> Dict[str, float]:
        """Get meter data for GUI display"""
        return {
            'input_peak_db': self.input_peak_db,
            'output_peak_db': self.output_peak_db,
            'gain_reduction_db': self.gain_reduction_db,
            'compression_ratio': self.compression_ratio_actual
        }
    
    def reset_meters(self):
        """Reset meter values"""
        self.input_peak_db = -60.0
        self.output_peak_db = -60.0
        self.gain_reduction_db = 0.0
    
    def get_info(self) -> Dict[str, Any]:
        """Get compressor information including meters"""
        info = super().get_info()
        info['meters'] = self.get_meter_data()
        return info