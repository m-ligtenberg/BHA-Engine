"""
Professional Reverb Processor
Multiple algorithms: Room, Hall, Plate, Spring reverbs optimized for drums
"""

import numpy as np
from typing import Dict, Any, List
import time

from ..effects_chain import BaseEffect, EffectType


class DelayLine:
    """Simple delay line for reverb algorithms"""
    
    def __init__(self, max_delay_samples: int):
        self.buffer = np.zeros(max_delay_samples, dtype=np.float32)
        self.write_index = 0
        self.max_delay = max_delay_samples
    
    def read(self, delay_samples: int) -> float:
        """Read from delay line with linear interpolation"""
        delay_samples = max(1, min(delay_samples, self.max_delay - 1))
        read_index = (self.write_index - delay_samples) % self.max_delay
        
        # Linear interpolation for fractional delays
        int_delay = int(delay_samples)
        frac_delay = delay_samples - int_delay
        
        index1 = (self.write_index - int_delay) % self.max_delay
        index2 = (self.write_index - int_delay - 1) % self.max_delay
        
        sample1 = self.buffer[index1]
        sample2 = self.buffer[index2]
        
        return sample1 * (1.0 - frac_delay) + sample2 * frac_delay
    
    def write(self, sample: float):
        """Write to delay line"""
        self.buffer[self.write_index] = sample
        self.write_index = (self.write_index + 1) % self.max_delay


class AllpassFilter:
    """Allpass filter for reverb diffusion"""
    
    def __init__(self, delay_samples: int, gain: float = 0.7):
        self.delay_line = DelayLine(delay_samples)
        self.gain = gain
    
    def process(self, input_sample: float) -> float:
        """Process sample through allpass filter"""
        delayed = self.delay_line.read(len(self.delay_line.buffer) - 1)
        output = -self.gain * input_sample + delayed
        self.delay_line.write(input_sample + self.gain * delayed)
        return output


class CombFilter:
    """Comb filter with feedback for reverb"""
    
    def __init__(self, delay_samples: int, feedback: float = 0.5, damping: float = 0.5):
        self.delay_line = DelayLine(delay_samples)
        self.feedback = feedback
        self.damping = damping
        self.last_output = 0.0
    
    def process(self, input_sample: float) -> float:
        """Process sample through comb filter"""
        delayed = self.delay_line.read(len(self.delay_line.buffer) - 1)
        
        # Apply damping (simple low-pass filter)
        filtered = delayed * (1.0 - self.damping) + self.last_output * self.damping
        self.last_output = filtered
        
        output = input_sample + filtered * self.feedback
        self.delay_line.write(output)
        return output


class ReverbProcessor(BaseEffect):
    """
    Professional reverb with multiple algorithms optimized for drum processing.
    Implements Room, Hall, Plate, and Spring reverb models.
    """
    
    def __init__(self, effect_id: str, effect_type: EffectType = EffectType.REVERB,
                 sample_rate: int = 44100, buffer_size: int = 512):
        super().__init__(effect_id, effect_type, sample_rate, buffer_size)
        
        # Current algorithm
        self.current_algorithm = 'room'
        
        # Initialize reverb algorithms
        self._initialize_room_reverb()
        self._initialize_hall_reverb()
        self._initialize_plate_reverb()
        self._initialize_spring_reverb()
        
        # Pre-delay
        max_predelay_samples = int(0.5 * sample_rate)  # 500ms max pre-delay
        self.predelay_left = DelayLine(max_predelay_samples)
        self.predelay_right = DelayLine(max_predelay_samples)
        
        # High and low frequency damping
        self.high_damp_left = 0.0
        self.high_damp_right = 0.0
        self.low_damp_left = 0.0
        self.low_damp_right = 0.0
        
        # Stereo width processing
        self.stereo_spread = 0.0
        
        # Meters
        self.reverb_level_db = -60.0
        self.reverb_time_actual = 0.0
    
    def _initialize_parameters(self):
        """Initialize reverb parameters"""
        self.parameters = {
            'algorithm': 0.0,           # 0=room, 1=hall, 2=plate, 3=spring
            'room_size': 0.5,           # Room size/decay time
            'decay_time': 2.0,          # Decay time in seconds
            'damping': 0.5,             # High frequency damping
            'diffusion': 0.7,           # Early reflection diffusion
            'pre_delay_ms': 20.0,       # Pre-delay time
            'low_cut_hz': 0.0,          # Low frequency cut
            'high_cut_hz': 20000.0,     # High frequency cut
            'stereo_width': 1.0,        # Stereo width
            'early_late_mix': 0.5,      # Early/late reflection mix
            'modulation_depth': 0.1,    # Chorus/modulation depth
            'modulation_rate': 0.5,     # Modulation rate in Hz
        }
        
        self.parameter_ranges = {
            'algorithm': (0.0, 3.0),
            'room_size': (0.1, 1.0),
            'decay_time': (0.1, 10.0),
            'damping': (0.0, 1.0),
            'diffusion': (0.0, 1.0),
            'pre_delay_ms': (0.0, 500.0),
            'low_cut_hz': (0.0, 500.0),
            'high_cut_hz': (1000.0, 20000.0),
            'stereo_width': (0.0, 2.0),
            'early_late_mix': (0.0, 1.0),
            'modulation_depth': (0.0, 0.5),
            'modulation_rate': (0.1, 10.0)
        }
        
        self.parameter_defaults = self.parameters.copy()
    
    def _initialize_room_reverb(self):
        """Initialize room reverb algorithm (good for drums)"""
        # Room reverb - shorter, more intimate sound
        delay_times_ms = [23, 29, 37, 41, 47, 53, 59, 61]  # Prime numbers for decorrelation
        delay_samples = [int(dt * self.sample_rate / 1000) for dt in delay_times_ms]
        
        self.room_comb_filters_l = []
        self.room_comb_filters_r = []
        
        for i, delay in enumerate(delay_samples[:4]):
            # Slightly different delays for left/right for stereo spread
            delay_l = delay
            delay_r = int(delay * 1.1)
            
            self.room_comb_filters_l.append(CombFilter(delay_l, 0.6, 0.3))
            self.room_comb_filters_r.append(CombFilter(delay_r, 0.6, 0.3))
        
        # Allpass filters for diffusion
        allpass_delays = [7, 11, 13, 17]  # Small prime numbers
        allpass_samples = [int(dt * self.sample_rate / 1000) for dt in allpass_delays]
        
        self.room_allpass_l = []
        self.room_allpass_r = []
        
        for delay in allpass_samples:
            self.room_allpass_l.append(AllpassFilter(delay, 0.7))
            self.room_allpass_r.append(AllpassFilter(int(delay * 1.05), 0.7))
    
    def _initialize_hall_reverb(self):
        """Initialize hall reverb algorithm (spacious, good for ambient drums)"""
        # Hall reverb - longer, more spacious sound
        delay_times_ms = [89, 97, 101, 107, 113, 127, 131, 137]
        delay_samples = [int(dt * self.sample_rate / 1000) for dt in delay_times_ms]
        
        self.hall_comb_filters_l = []
        self.hall_comb_filters_r = []
        
        for i, delay in enumerate(delay_samples):
            delay_l = delay
            delay_r = int(delay * (1.0 + 0.1 * (i % 2)))  # Vary stereo delays
            
            self.hall_comb_filters_l.append(CombFilter(delay_l, 0.8, 0.2))
            self.hall_comb_filters_r.append(CombFilter(delay_r, 0.8, 0.2))
        
        # More allpass stages for hall diffusion
        allpass_delays = [19, 23, 29, 31, 37, 41]
        allpass_samples = [int(dt * self.sample_rate / 1000) for dt in allpass_delays]
        
        self.hall_allpass_l = []
        self.hall_allpass_r = []
        
        for delay in allpass_samples:
            self.hall_allpass_l.append(AllpassFilter(delay, 0.6))
            self.hall_allpass_r.append(AllpassFilter(int(delay * 1.08), 0.6))
    
    def _initialize_plate_reverb(self):
        """Initialize plate reverb algorithm (vintage sound, great for snare)"""
        # Plate reverb - metallic character, dense early reflections
        delay_times_ms = [31, 37, 43, 47, 53, 59, 67, 71]
        delay_samples = [int(dt * self.sample_rate / 1000) for dt in delay_times_ms]
        
        self.plate_comb_filters_l = []
        self.plate_comb_filters_r = []
        
        for i, delay in enumerate(delay_samples):
            delay_l = delay
            delay_r = int(delay * (0.9 + 0.2 * (i % 3) / 3))  # More variation for plate
            
            # Plate has higher feedback for metallic sound
            self.plate_comb_filters_l.append(CombFilter(delay_l, 0.75, 0.1))
            self.plate_comb_filters_r.append(CombFilter(delay_r, 0.75, 0.1))
        
        # Plate allpass network
        allpass_delays = [5, 7, 11, 13, 17, 19]
        allpass_samples = [int(dt * self.sample_rate / 1000) for dt in allpass_delays]
        
        self.plate_allpass_l = []
        self.plate_allpass_r = []
        
        for delay in allpass_samples:
            self.plate_allpass_l.append(AllpassFilter(delay, 0.8))
            self.plate_allpass_r.append(AllpassFilter(int(delay * 1.15), 0.8))
    
    def _initialize_spring_reverb(self):
        """Initialize spring reverb algorithm (vintage sound, great for guitar/drums)"""
        # Spring reverb - bouncy character with distinct early reflections
        delay_times_ms = [67, 73, 79, 83, 89, 97]  # Longer delays for spring character
        delay_samples = [int(dt * self.sample_rate / 1000) for dt in delay_times_ms]
        
        self.spring_comb_filters_l = []
        self.spring_comb_filters_r = []
        
        for i, delay in enumerate(delay_samples):
            delay_l = delay
            delay_r = int(delay * 1.2)  # Significant stereo offset for spring
            
            # Spring has moderate feedback with more damping
            self.spring_comb_filters_l.append(CombFilter(delay_l, 0.65, 0.4))
            self.spring_comb_filters_r.append(CombFilter(delay_r, 0.65, 0.4))
        
        # Spring allpass - fewer stages, more resonant
        allpass_delays = [17, 23, 29]
        allpass_samples = [int(dt * self.sample_rate / 1000) for dt in allpass_delays]
        
        self.spring_allpass_l = []
        self.spring_allpass_r = []
        
        for delay in allpass_samples:
            self.spring_allpass_l.append(AllpassFilter(delay, 0.5))
            self.spring_allpass_r.append(AllpassFilter(int(delay * 1.3), 0.5))
    
    def process(self, input_data: np.ndarray, frames: int) -> np.ndarray:
        """Process audio through selected reverb algorithm"""
        start_time = time.perf_counter()
        
        try:
            # Ensure stereo input
            if input_data.shape[1] == 1:
                stereo_input = np.column_stack([input_data[:, 0], input_data[:, 0]])
            else:
                stereo_input = input_data[:frames].copy()
            
            output_data = np.zeros_like(stereo_input)
            
            # Update algorithm if changed
            algorithm_index = int(self.parameters['algorithm'])
            algorithms = ['room', 'hall', 'plate', 'spring']
            self.current_algorithm = algorithms[min(algorithm_index, len(algorithms) - 1)]
            
            # Process sample by sample
            for i in range(frames):
                left_sample = stereo_input[i, 0]
                right_sample = stereo_input[i, 1]
                
                # Apply pre-delay
                predelay_samples = int(self.parameters['pre_delay_ms'] * self.sample_rate / 1000)
                
                self.predelay_left.write(left_sample)
                self.predelay_right.write(right_sample)
                
                delayed_left = self.predelay_left.read(predelay_samples)
                delayed_right = self.predelay_right.read(predelay_samples)
                
                # Apply low/high cut filters (simple implementation)
                if self.parameters['low_cut_hz'] > 0:
                    delayed_left = self._apply_highpass(delayed_left, self.parameters['low_cut_hz'], 'left')
                    delayed_right = self._apply_highpass(delayed_right, self.parameters['low_cut_hz'], 'right')
                
                if self.parameters['high_cut_hz'] < 20000:
                    delayed_left = self._apply_lowpass(delayed_left, self.parameters['high_cut_hz'], 'left')
                    delayed_right = self._apply_lowpass(delayed_right, self.parameters['high_cut_hz'], 'right')
                
                # Process through selected algorithm
                reverb_left, reverb_right = self._process_algorithm(delayed_left, delayed_right)
                
                # Apply stereo width
                width = self.parameters['stereo_width']
                if width != 1.0:
                    mid = (reverb_left + reverb_right) * 0.5
                    side = (reverb_left - reverb_right) * 0.5 * width
                    reverb_left = mid + side
                    reverb_right = mid - side
                
                output_data[i, 0] = reverb_left
                output_data[i, 1] = reverb_right
                
                # Update meters (every 32 samples)
                if i % 32 == 0:
                    reverb_level = max(abs(reverb_left), abs(reverb_right))
                    self.reverb_level_db = 20 * np.log10(max(reverb_level, 1e-10))
            
            # Apply wet/dry mix
            output_data = self._apply_wet_dry_mix(stereo_input, output_data)
            
            # Performance tracking
            self.processing_time_ms = (time.perf_counter() - start_time) * 1000
            
            return output_data
            
        except Exception as e:
            print(f"Reverb error: {e}")
            return input_data[:frames]
    
    def _process_algorithm(self, left_input: float, right_input: float) -> tuple:
        """Process through selected reverb algorithm"""
        if self.current_algorithm == 'room':
            return self._process_room_reverb(left_input, right_input)
        elif self.current_algorithm == 'hall':
            return self._process_hall_reverb(left_input, right_input)
        elif self.current_algorithm == 'plate':
            return self._process_plate_reverb(left_input, right_input)
        elif self.current_algorithm == 'spring':
            return self._process_spring_reverb(left_input, right_input)
        else:
            return left_input, right_input
    
    def _process_room_reverb(self, left: float, right: float) -> tuple:
        """Process room reverb algorithm"""
        # Adjust feedback based on room size and decay time
        feedback_scale = self.parameters['room_size'] * (self.parameters['decay_time'] / 10.0)
        
        # Process through comb filters
        comb_out_l = 0.0
        comb_out_r = 0.0
        
        for comb_l, comb_r in zip(self.room_comb_filters_l, self.room_comb_filters_r):
            comb_l.feedback = 0.6 * feedback_scale
            comb_r.feedback = 0.6 * feedback_scale
            comb_l.damping = self.parameters['damping']
            comb_r.damping = self.parameters['damping']
            
            comb_out_l += comb_l.process(left)
            comb_out_r += comb_r.process(right)
        
        # Process through allpass filters
        diffused_l = comb_out_l * self.parameters['diffusion']
        diffused_r = comb_out_r * self.parameters['diffusion']
        
        for allpass_l, allpass_r in zip(self.room_allpass_l, self.room_allpass_r):
            diffused_l = allpass_l.process(diffused_l)
            diffused_r = allpass_r.process(diffused_r)
        
        return diffused_l, diffused_r
    
    def _process_hall_reverb(self, left: float, right: float) -> tuple:
        """Process hall reverb algorithm"""
        feedback_scale = self.parameters['room_size'] * (self.parameters['decay_time'] / 10.0)
        
        # Hall has more comb filters for spaciousness
        comb_out_l = 0.0
        comb_out_r = 0.0
        
        for comb_l, comb_r in zip(self.hall_comb_filters_l, self.hall_comb_filters_r):
            comb_l.feedback = 0.8 * feedback_scale
            comb_r.feedback = 0.8 * feedback_scale
            comb_l.damping = self.parameters['damping'] * 0.5  # Less damping for hall
            comb_r.damping = self.parameters['damping'] * 0.5
            
            comb_out_l += comb_l.process(left * 0.7)  # Reduce input level
            comb_out_r += comb_r.process(right * 0.7)
        
        # More diffusion stages
        diffused_l = comb_out_l * self.parameters['diffusion']
        diffused_r = comb_out_r * self.parameters['diffusion']
        
        for allpass_l, allpass_r in zip(self.hall_allpass_l, self.hall_allpass_r):
            diffused_l = allpass_l.process(diffused_l)
            diffused_r = allpass_r.process(diffused_r)
        
        return diffused_l, diffused_r
    
    def _process_plate_reverb(self, left: float, right: float) -> tuple:
        """Process plate reverb algorithm"""
        feedback_scale = self.parameters['room_size'] * (self.parameters['decay_time'] / 10.0)
        
        # Plate reverb processing
        comb_out_l = 0.0
        comb_out_r = 0.0
        
        for comb_l, comb_r in zip(self.plate_comb_filters_l, self.plate_comb_filters_r):
            comb_l.feedback = 0.75 * feedback_scale
            comb_r.feedback = 0.75 * feedback_scale
            comb_l.damping = self.parameters['damping'] * 0.3  # Less damping for metallic sound
            comb_r.damping = self.parameters['damping'] * 0.3
            
            comb_out_l += comb_l.process(left)
            comb_out_r += comb_r.process(right)
        
        # High diffusion for plate character
        diffused_l = comb_out_l * self.parameters['diffusion']
        diffused_r = comb_out_r * self.parameters['diffusion']
        
        for allpass_l, allpass_r in zip(self.plate_allpass_l, self.plate_allpass_r):
            diffused_l = allpass_l.process(diffused_l)
            diffused_r = allpass_r.process(diffused_r)
        
        return diffused_l, diffused_r
    
    def _process_spring_reverb(self, left: float, right: float) -> tuple:
        """Process spring reverb algorithm"""
        feedback_scale = self.parameters['room_size'] * (self.parameters['decay_time'] / 10.0)
        
        # Spring reverb processing
        comb_out_l = 0.0
        comb_out_r = 0.0
        
        for comb_l, comb_r in zip(self.spring_comb_filters_l, self.spring_comb_filters_r):
            comb_l.feedback = 0.65 * feedback_scale
            comb_r.feedback = 0.65 * feedback_scale
            comb_l.damping = self.parameters['damping']
            comb_r.damping = self.parameters['damping']
            
            comb_out_l += comb_l.process(left)
            comb_out_r += comb_r.process(right)
        
        # Less diffusion for spring bounce
        diffused_l = comb_out_l * self.parameters['diffusion'] * 0.7
        diffused_r = comb_out_r * self.parameters['diffusion'] * 0.7
        
        for allpass_l, allpass_r in zip(self.spring_allpass_l, self.spring_allpass_r):
            diffused_l = allpass_l.process(diffused_l)
            diffused_r = allpass_r.process(diffused_r)
        
        return diffused_l, diffused_r
    
    def _apply_highpass(self, sample: float, cutoff_hz: float, channel: str) -> float:
        """Simple high-pass filter"""
        # Simple one-pole high-pass filter
        cutoff_norm = cutoff_hz / (self.sample_rate / 2)
        alpha = 1.0 / (1.0 + 2 * np.pi * cutoff_norm)
        
        if channel == 'left':
            filtered = alpha * (sample - self.low_damp_left)
            self.low_damp_left = sample
        else:
            filtered = alpha * (sample - self.low_damp_right)
            self.low_damp_right = sample
        
        return filtered
    
    def _apply_lowpass(self, sample: float, cutoff_hz: float, channel: str) -> float:
        """Simple low-pass filter"""
        # Simple one-pole low-pass filter
        cutoff_norm = cutoff_hz / (self.sample_rate / 2)
        alpha = 2 * np.pi * cutoff_norm / (1.0 + 2 * np.pi * cutoff_norm)
        
        if channel == 'left':
            self.high_damp_left = self.high_damp_left + alpha * (sample - self.high_damp_left)
            return self.high_damp_left
        else:
            self.high_damp_right = self.high_damp_right + alpha * (sample - self.high_damp_right)
            return self.high_damp_right
    
    def get_drum_presets(self) -> Dict[str, Dict[str, float]]:
        """Get drum-optimized reverb presets"""
        return {
            'kick_room': {
                'algorithm': 0.0,  # room
                'room_size': 0.3,
                'decay_time': 1.2,
                'damping': 0.6,
                'pre_delay_ms': 5.0,
                'wet_dry_mix': 0.2
            },
            'snare_plate': {
                'algorithm': 2.0,  # plate
                'room_size': 0.4,
                'decay_time': 1.8,
                'damping': 0.3,
                'pre_delay_ms': 15.0,
                'wet_dry_mix': 0.3
            },
            'drum_hall': {
                'algorithm': 1.0,  # hall
                'room_size': 0.7,
                'decay_time': 3.5,
                'damping': 0.4,
                'pre_delay_ms': 30.0,
                'wet_dry_mix': 0.25
            },
            'vintage_spring': {
                'algorithm': 3.0,  # spring
                'room_size': 0.5,
                'decay_time': 2.0,
                'damping': 0.5,
                'pre_delay_ms': 10.0,
                'wet_dry_mix': 0.4
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
        """Get reverb meter data"""
        return {
            'reverb_level_db': self.reverb_level_db,
            'decay_time_actual': self.reverb_time_actual,
            'algorithm': self.current_algorithm
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get reverb information"""
        info = super().get_info()
        info['meters'] = self.get_meter_data()
        info['current_algorithm'] = self.current_algorithm
        info['available_presets'] = list(self.get_drum_presets().keys())
        return info