"""
Professional Analog-Style Saturation Processor
Tube, tape, and transistor saturation models for musical warmth
"""

import numpy as np
from typing import Dict, Any
import time

from ..effects_chain import BaseEffect, EffectType


class AnalogSaturation(BaseEffect):
    """
    Professional analog-style saturation with multiple models:
    - Tube saturation (warm, even harmonics)
    - Tape saturation (smooth compression and harmonics)  
    - Transistor saturation (aggressive, odd harmonics)
    """
    
    def __init__(self, effect_id: str, effect_type: EffectType = EffectType.SATURATION,
                 sample_rate: int = 44100, buffer_size: int = 512):
        super().__init__(effect_id, effect_type, sample_rate, buffer_size)
        
        # Saturation models
        self.current_model = 'tube'
        
        # High-frequency emphasis/de-emphasis for tape modeling
        self.emphasis_state_l = 0.0
        self.emphasis_state_r = 0.0
        self.deemphasis_state_l = 0.0
        self.deemphasis_state_r = 0.0
        
        # Bias for asymmetric saturation
        self.bias_amount = 0.0
        
        # Output meters
        self.saturation_amount_db = 0.0
        self.harmonic_content = 0.0
    
    def _initialize_parameters(self):
        """Initialize saturation parameters"""
        self.parameters = {
            'model': 0.0,           # 0=tube, 1=tape, 2=transistor
            'drive': 1.0,           # Input drive/gain
            'saturation': 0.3,      # Saturation amount
            'bias': 0.0,            # Asymmetric bias (-1 to 1)
            'warmth': 0.5,          # Low-frequency emphasis
            'presence': 0.5,        # High-frequency emphasis  
            'output_gain': 0.0,     # Output gain compensation
            'color': 0.5,           # Harmonic coloration
            'dynamics': 0.3,        # Dynamic response
        }
        
        self.parameter_ranges = {
            'model': (0.0, 2.0),
            'drive': (0.1, 10.0),
            'saturation': (0.0, 1.0),
            'bias': (-1.0, 1.0),
            'warmth': (0.0, 1.0),
            'presence': (0.0, 1.0),
            'output_gain': (-20.0, 20.0),
            'color': (0.0, 1.0),
            'dynamics': (0.0, 1.0)
        }
        
        self.parameter_defaults = self.parameters.copy()
    
    def process(self, input_data: np.ndarray, frames: int) -> np.ndarray:
        """Process audio through analog saturation"""
        start_time = time.perf_counter()
        
        try:
            # Ensure stereo input
            if input_data.shape[1] == 1:
                stereo_input = np.column_stack([input_data[:, 0], input_data[:, 0]])
            else:
                stereo_input = input_data[:frames].copy()
            
            output_data = np.zeros_like(stereo_input)
            
            # Select saturation model
            model_index = int(self.parameters['model'])
            if model_index == 0:
                self.current_model = 'tube'
            elif model_index == 1:
                self.current_model = 'tape'
            else:
                self.current_model = 'transistor'
            
            # Process sample by sample for maximum quality
            for i in range(frames):
                left_in = stereo_input[i, 0]
                right_in = stereo_input[i, 1]
                
                # Apply input drive
                drive = self.parameters['drive']
                left_driven = left_in * drive
                right_driven = right_in * drive
                
                # Apply bias for asymmetric saturation
                bias = self.parameters['bias']
                left_driven += bias * 0.1
                right_driven += bias * 0.1
                
                # Apply warmth (low-frequency emphasis)
                if self.parameters['warmth'] > 0:
                    left_driven, right_driven = self._apply_warmth(left_driven, right_driven)
                
                # Apply saturation based on model
                if self.current_model == 'tube':
                    left_sat, right_sat = self._tube_saturation(left_driven, right_driven)
                elif self.current_model == 'tape':
                    left_sat, right_sat = self._tape_saturation(left_driven, right_driven)
                else:  # transistor
                    left_sat, right_sat = self._transistor_saturation(left_driven, right_driven)
                
                # Apply presence (high-frequency emphasis)
                if self.parameters['presence'] > 0:
                    left_sat, right_sat = self._apply_presence(left_sat, right_sat)
                
                # Apply output gain
                output_gain = 10 ** (self.parameters['output_gain'] / 20.0)
                left_sat *= output_gain
                right_sat *= output_gain
                
                output_data[i, 0] = left_sat
                output_data[i, 1] = right_sat
                
                # Update meters (every 32 samples)
                if i % 32 == 0:
                    input_level = max(abs(left_in), abs(right_in))
                    output_level = max(abs(left_sat), abs(right_sat))
                    if input_level > 0:
                        self.saturation_amount_db = 20 * np.log10(output_level / input_level)
            
            # Apply wet/dry mix
            output_data = self._apply_wet_dry_mix(stereo_input, output_data)
            
            # Performance tracking
            self.processing_time_ms = (time.perf_counter() - start_time) * 1000
            
            return output_data
            
        except Exception as e:
            print(f"Saturation error: {e}")
            return input_data[:frames]
    
    def _tube_saturation(self, left: float, right: float) -> tuple:
        """Tube saturation model - warm, even harmonics"""
        saturation_amount = self.parameters['saturation']
        
        # Tube-like saturation curve (smooth, even harmonics)
        def tube_curve(x):
            # Gentle saturation with even harmonics
            if abs(x) < 1.0:
                # Smooth polynomial saturation
                return x * (1.0 - abs(x) * saturation_amount * 0.3)
            else:
                # Soft limiting
                return np.sign(x) * (1.0 - np.exp(-abs(x)))
        
        # Apply tube saturation
        left_sat = tube_curve(left)
        right_sat = tube_curve(right)
        
        # Add subtle even harmonics
        color = self.parameters['color']
        if color > 0:
            left_sat += np.sin(left * 2 * np.pi) * color * 0.05
            right_sat += np.sin(right * 2 * np.pi) * color * 0.05
        
        return left_sat, right_sat
    
    def _tape_saturation(self, left: float, right: float) -> tuple:
        """Tape saturation model - smooth compression and saturation"""
        saturation_amount = self.parameters['saturation']
        
        # Tape pre-emphasis (high-frequency boost before saturation)
        left_emphasized = self._apply_emphasis(left, 0)
        right_emphasized = self._apply_emphasis(right, 1)
        
        # Tape saturation curve (smooth compression)
        def tape_curve(x):
            # Tape-like soft compression and saturation
            compressed = np.tanh(x * (1 + saturation_amount * 2))
            return compressed * 0.8  # Slight level reduction
        
        left_sat = tape_curve(left_emphasized)
        right_sat = tape_curve(right_emphasized)
        
        # Tape de-emphasis (high-frequency cut after saturation)
        left_sat = self._apply_deemphasis(left_sat, 0)
        right_sat = self._apply_deemphasis(right_sat, 1)
        
        # Add tape flutter/wow (very subtle modulation)
        dynamics = self.parameters['dynamics']
        if dynamics > 0:
            flutter_amount = dynamics * 0.002
            left_sat *= (1.0 + np.sin(left * 50) * flutter_amount)
            right_sat *= (1.0 + np.sin(right * 50) * flutter_amount)
        
        return left_sat, right_sat
    
    def _transistor_saturation(self, left: float, right: float) -> tuple:
        """Transistor saturation model - aggressive, odd harmonics"""
        saturation_amount = self.parameters['saturation']
        
        # Transistor saturation curve (hard clipping character)
        def transistor_curve(x):
            # Hard saturation with odd harmonics
            gain = 1 + saturation_amount * 3
            driven = x * gain
            
            if abs(driven) < 0.5:
                return driven
            elif abs(driven) < 1.0:
                # Soft clipping region
                return np.sign(driven) * (0.5 + (abs(driven) - 0.5) * 0.3)
            else:
                # Hard limiting
                return np.sign(driven) * 0.65
        
        left_sat = transistor_curve(left)
        right_sat = transistor_curve(right)
        
        # Add odd harmonics
        color = self.parameters['color']
        if color > 0:
            left_sat += np.sign(left) * (left ** 3) * color * 0.1
            right_sat += np.sign(right) * (right ** 3) * color * 0.1
        
        return left_sat, right_sat
    
    def _apply_warmth(self, left: float, right: float) -> tuple:
        """Apply low-frequency warmth/emphasis"""
        warmth = self.parameters['warmth']
        
        # Simple bass boost
        bass_boost = 1.0 + warmth * 0.3
        return left * bass_boost, right * bass_boost
    
    def _apply_presence(self, left: float, right: float) -> tuple:
        """Apply high-frequency presence/emphasis"""
        presence = self.parameters['presence']
        
        # Simple treble boost
        treble_boost = 1.0 + presence * 0.2
        return left * treble_boost, right * treble_boost
    
    def _apply_emphasis(self, sample: float, channel: int) -> float:
        """Apply pre-emphasis filter (high-frequency boost)"""
        alpha = 0.1  # Emphasis coefficient
        
        if channel == 0:  # Left
            emphasized = sample - self.emphasis_state_l * alpha
            self.emphasis_state_l = sample
        else:  # Right  
            emphasized = sample - self.emphasis_state_r * alpha
            self.emphasis_state_r = sample
        
        return emphasized
    
    def _apply_deemphasis(self, sample: float, channel: int) -> float:
        """Apply de-emphasis filter (high-frequency cut)"""
        alpha = 0.1  # De-emphasis coefficient
        
        if channel == 0:  # Left
            self.deemphasis_state_l = self.deemphasis_state_l * (1 - alpha) + sample * alpha
            return self.deemphasis_state_l
        else:  # Right
            self.deemphasis_state_r = self.deemphasis_state_r * (1 - alpha) + sample * alpha
            return self.deemphasis_state_r
    
    def get_drum_presets(self) -> Dict[str, Dict[str, float]]:
        """Get drum-optimized saturation presets"""
        return {
            'kick_thump': {
                'model': 0.0,      # Tube
                'drive': 2.0,
                'saturation': 0.4,
                'warmth': 0.7,
                'output_gain': -3.0,
                'wet_dry_mix': 0.6
            },
            'snare_snap': {
                'model': 2.0,      # Transistor
                'drive': 3.0,
                'saturation': 0.5,
                'presence': 0.6,
                'bias': 0.2,
                'wet_dry_mix': 0.5
            },
            'vintage_tape': {
                'model': 1.0,      # Tape
                'drive': 1.5,
                'saturation': 0.6,
                'warmth': 0.5,
                'dynamics': 0.4,
                'wet_dry_mix': 0.7
            },
            'subtle_warmth': {
                'model': 0.0,      # Tube
                'drive': 1.2,
                'saturation': 0.2,
                'warmth': 0.4,
                'color': 0.3,
                'wet_dry_mix': 0.3
            }
        }
    
    def get_meter_data(self) -> Dict[str, float]:
        """Get saturation meter data"""
        return {
            'saturation_amount_db': self.saturation_amount_db,
            'current_model': self.current_model,
            'harmonic_content': self.harmonic_content
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get saturation information"""
        info = super().get_info()
        info['meters'] = self.get_meter_data()
        info['current_model'] = self.current_model
        info['available_presets'] = list(self.get_drum_presets().keys())
        return info