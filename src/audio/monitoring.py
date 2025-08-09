"""
Professional Audio Monitoring and VU Metering System
High-precision level monitoring, spectrum analysis, and comprehensive metering
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from threading import RLock, Thread
from enum import Enum
from PyQt6.QtCore import QObject, pyqtSignal, QTimer
import time
from collections import deque


class MeterType(Enum):
    """Types of audio meters supported"""
    PEAK = "peak"
    RMS = "rms"
    VU = "vu"
    PPM = "ppm"
    LOUDNESS = "loudness"
    SPECTRUM = "spectrum"
    PHASE = "phase"
    CORRELATION = "correlation"


class MeterBallistics(Enum):
    """Meter response characteristics"""
    DIGITAL_PEAK = "digital_peak"     # Instant peak, slow release
    VU_ANALOG = "vu_analog"           # Classic VU meter ballistics
    PPM_BBC = "ppm_bbc"               # BBC PPM standard
    PPM_EBU = "ppm_ebu"               # EBU PPM standard
    CUSTOM = "custom"                 # User-defined response


class ProfessionalMeter:
    """
    High-precision audio meter with professional ballistics and calibration.
    Supports multiple meter types and standards.
    """
    
    def __init__(self, meter_type: MeterType, ballistics: MeterBallistics = MeterBallistics.DIGITAL_PEAK,
                 sample_rate: int = 44100, update_rate_hz: float = 30.0):
        self.meter_type = meter_type
        self.ballistics = ballistics
        self.sample_rate = sample_rate
        self.update_rate_hz = update_rate_hz
        
        # Meter state
        self.current_level = 0.0
        self.peak_level = 0.0
        self.rms_level = 0.0
        self.peak_hold_level = 0.0
        self.peak_hold_time = 0.0
        self.over_count = 0
        
        # Ballistics parameters
        self.attack_time_s = 0.0
        self.release_time_s = 0.3
        self.integration_time_s = 0.3
        self.peak_hold_time_s = 1.5
        
        # Configure ballistics
        self._configure_ballistics()
        
        # Sample buffers for RMS calculation
        self.rms_buffer_size = int(self.integration_time_s * sample_rate)
        self.rms_buffer = deque(maxlen=self.rms_buffer_size)
        
        # Peak detection
        self.peak_detector_alpha = 0.0
        self.rms_detector_alpha = 0.0
        
        # Calculate filter coefficients
        self._calculate_filter_coefficients()
        
        # Spectrum analyzer (if needed)
        self.fft_size = 1024
        self.spectrum_buffer = np.zeros(self.fft_size, dtype=np.float32)
        self.spectrum_result = np.zeros(self.fft_size // 2, dtype=np.float32)
        self.window = np.hanning(self.fft_size)
        
        # Calibration
        self.reference_level_dbfs = -20.0  # dBFS for 0 VU
        self.calibration_offset_db = 0.0
    
    def _configure_ballistics(self):
        """Configure meter ballistics based on standard"""
        if self.ballistics == MeterBallistics.DIGITAL_PEAK:
            self.attack_time_s = 0.0      # Instant attack
            self.release_time_s = 1.5     # Slow release
            self.integration_time_s = 0.0 # No integration
            
        elif self.ballistics == MeterBallistics.VU_ANALOG:
            self.attack_time_s = 0.3      # VU meter rise time
            self.release_time_s = 0.3     # VU meter fall time
            self.integration_time_s = 0.3 # VU integration
            
        elif self.ballistics == MeterBallistics.PPM_BBC:
            self.attack_time_s = 0.01     # BBC PPM attack (10ms)
            self.release_time_s = 1.5     # BBC PPM decay
            self.integration_time_s = 0.01
            
        elif self.ballistics == MeterBallistics.PPM_EBU:
            self.attack_time_s = 0.01     # EBU PPM attack
            self.release_time_s = 1.7     # EBU PPM decay
            self.integration_time_s = 0.01
    
    def _calculate_filter_coefficients(self):
        """Calculate filter coefficients for ballistics"""
        update_period = 1.0 / self.update_rate_hz
        
        # Attack coefficient
        if self.attack_time_s > 0:
            self.peak_detector_alpha = np.exp(-update_period / self.attack_time_s)
        else:
            self.peak_detector_alpha = 0.0
        
        # Release coefficient
        release_alpha = np.exp(-update_period / self.release_time_s)
        self.release_coefficient = release_alpha
        
        # RMS coefficient
        if self.integration_time_s > 0:
            self.rms_detector_alpha = np.exp(-update_period / self.integration_time_s)
        else:
            self.rms_detector_alpha = 0.0
    
    def process_samples(self, audio_samples: np.ndarray) -> Dict[str, float]:
        """Process audio samples and update meter readings"""
        if len(audio_samples.shape) == 2:
            # Stereo - use maximum of both channels
            samples = np.max(np.abs(audio_samples), axis=1)
        else:
            # Mono
            samples = np.abs(audio_samples)
        
        if len(samples) == 0:
            return self.get_readings()
        
        # Peak detection
        current_peak = np.max(samples)
        
        # Update peak level with ballistics
        if current_peak > self.peak_level:
            # Attack
            self.peak_level = current_peak * (1 - self.peak_detector_alpha) + self.peak_level * self.peak_detector_alpha
        else:
            # Release
            self.peak_level = self.peak_level * self.release_coefficient
        
        # Peak hold
        if current_peak > self.peak_hold_level:
            self.peak_hold_level = current_peak
            self.peak_hold_time = time.time()
        elif time.time() - self.peak_hold_time > self.peak_hold_time_s:
            self.peak_hold_level *= 0.99  # Gradual decay
        
        # RMS calculation
        for sample in samples:
            self.rms_buffer.append(sample ** 2)
        
        if len(self.rms_buffer) > 0:
            rms_squared = np.mean(list(self.rms_buffer))
            current_rms = np.sqrt(rms_squared)
            
            # RMS ballistics
            self.rms_level = current_rms * (1 - self.rms_detector_alpha) + self.rms_level * self.rms_detector_alpha
        
        # Over detection (clipping)
        if current_peak >= 1.0:
            self.over_count += 1
        
        # Update current level based on meter type
        if self.meter_type == MeterType.PEAK:
            self.current_level = self.peak_level
        elif self.meter_type == MeterType.RMS:
            self.current_level = self.rms_level
        elif self.meter_type == MeterType.VU:
            self.current_level = self.rms_level  # VU meters show RMS
        elif self.meter_type == MeterType.PPM:
            self.current_level = self.peak_level
        
        return self.get_readings()
    
    def get_readings(self) -> Dict[str, float]:
        """Get current meter readings in various formats"""
        readings = {
            'level_linear': self.current_level,
            'level_db': self._linear_to_db(self.current_level),
            'peak_linear': self.peak_level,
            'peak_db': self._linear_to_db(self.peak_level),
            'rms_linear': self.rms_level,
            'rms_db': self._linear_to_db(self.rms_level),
            'peak_hold_db': self._linear_to_db(self.peak_hold_level),
            'over_count': self.over_count
        }
        
        # Add calibrated VU reading
        if self.meter_type == MeterType.VU:
            readings['vu_reading'] = readings['level_db'] - self.reference_level_dbfs + self.calibration_offset_db
        
        return readings
    
    def _linear_to_db(self, linear_value: float) -> float:
        """Convert linear value to dB"""
        if linear_value <= 0:
            return -96.0  # -96dB floor
        return 20 * np.log10(linear_value)
    
    def reset_peak_hold(self):
        """Reset peak hold value"""
        self.peak_hold_level = 0.0
        self.peak_hold_time = time.time()
    
    def reset_over_count(self):
        """Reset over/clipping counter"""
        self.over_count = 0
    
    def set_calibration(self, reference_dbfs: float, offset_db: float = 0.0):
        """Set meter calibration"""
        self.reference_level_dbfs = reference_dbfs
        self.calibration_offset_db = offset_db


class SpectrumAnalyzer:
    """Real-time spectrum analyzer with professional features"""
    
    def __init__(self, fft_size: int = 2048, sample_rate: int = 44100, 
                 overlap: float = 0.5, averaging: int = 4):
        self.fft_size = fft_size
        self.sample_rate = sample_rate
        self.overlap = overlap
        self.averaging = averaging
        
        # FFT parameters
        self.hop_size = int(fft_size * (1.0 - overlap))
        self.window = np.hanning(fft_size)
        
        # Buffers
        self.input_buffer = np.zeros(fft_size, dtype=np.float32)
        self.buffer_index = 0
        
        # Spectrum averaging
        self.spectrum_history = deque(maxlen=averaging)
        self.current_spectrum = np.zeros(fft_size // 2, dtype=np.float32)
        
        # Frequency bins
        self.freq_bins = np.fft.fftfreq(fft_size, 1.0 / sample_rate)[:fft_size // 2]
    
    def process_samples(self, audio_samples: np.ndarray) -> np.ndarray:
        """Process audio samples and return spectrum"""
        if len(audio_samples.shape) == 2:
            # Convert stereo to mono for analysis
            samples = np.mean(audio_samples, axis=1)
        else:
            samples = audio_samples
        
        spectrum_updated = False
        
        # Process samples
        for sample in samples:
            self.input_buffer[self.buffer_index] = sample
            self.buffer_index += 1
            
            # When buffer is full, perform FFT
            if self.buffer_index >= self.fft_size:
                spectrum = self._calculate_spectrum()
                self.spectrum_history.append(spectrum)
                spectrum_updated = True
                
                # Shift buffer for overlap
                shift_amount = self.hop_size
                self.input_buffer[:-shift_amount] = self.input_buffer[shift_amount:]
                self.buffer_index -= shift_amount
        
        # Update averaged spectrum
        if spectrum_updated and len(self.spectrum_history) > 0:
            self.current_spectrum = np.mean(list(self.spectrum_history), axis=0)
        
        return self.current_spectrum
    
    def _calculate_spectrum(self) -> np.ndarray:
        """Calculate FFT spectrum"""
        # Apply window
        windowed = self.input_buffer * self.window
        
        # Calculate FFT
        fft_result = np.fft.fft(windowed)
        
        # Convert to magnitude spectrum in dB
        magnitude = np.abs(fft_result[:self.fft_size // 2])
        spectrum_db = 20 * np.log10(magnitude + 1e-10)  # Avoid log(0)
        
        return spectrum_db
    
    def get_spectrum_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get frequency bins and current spectrum"""
        return self.freq_bins, self.current_spectrum


class PhaseCorrelationMeter:
    """Professional phase correlation meter"""
    
    def __init__(self, sample_rate: int = 44100, integration_time: float = 0.3):
        self.sample_rate = sample_rate
        self.integration_time = integration_time
        
        # Integration buffer
        self.buffer_size = int(integration_time * sample_rate)
        self.left_buffer = deque(maxlen=self.buffer_size)
        self.right_buffer = deque(maxlen=self.buffer_size)
        
        # Current readings
        self.correlation = 0.0
        self.phase_difference = 0.0
        self.stereo_width = 0.0
    
    def process_stereo_samples(self, left_samples: np.ndarray, right_samples: np.ndarray) -> Dict[str, float]:
        """Process stereo samples and calculate correlation"""
        for left, right in zip(left_samples, right_samples):
            self.left_buffer.append(left)
            self.right_buffer.append(right)
        
        if len(self.left_buffer) >= self.buffer_size:
            left_array = np.array(list(self.left_buffer))
            right_array = np.array(list(self.right_buffer))
            
            # Calculate correlation coefficient
            correlation_matrix = np.corrcoef(left_array, right_array)
            self.correlation = correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0
            
            # Calculate phase difference (simplified)
            cross_correlation = np.correlate(left_array, right_array, mode='full')
            max_corr_index = np.argmax(np.abs(cross_correlation))
            delay_samples = max_corr_index - len(right_array) + 1
            self.phase_difference = (delay_samples / self.sample_rate) * 360.0  # Convert to degrees
            
            # Calculate stereo width
            sum_signal = left_array + right_array
            diff_signal = left_array - right_array
            
            sum_energy = np.mean(sum_signal ** 2)
            diff_energy = np.mean(diff_signal ** 2)
            
            if sum_energy > 0:
                self.stereo_width = diff_energy / sum_energy
            else:
                self.stereo_width = 0.0
        
        return {
            'correlation': self.correlation,
            'phase_difference_degrees': self.phase_difference,
            'stereo_width': self.stereo_width,
            'mono_compatibility': abs(self.correlation)  # Closer to 1 = better mono compatibility
        }


class AudioMonitoringSystem(QObject):
    """
    Comprehensive professional audio monitoring system.
    Integrates meters, spectrum analysis, and phase correlation.
    """
    
    # Signals for GUI updates
    level_update = pyqtSignal(str, dict)      # track_id, meter_readings
    spectrum_update = pyqtSignal(str, np.ndarray, np.ndarray)  # track_id, freq_bins, spectrum
    phase_update = pyqtSignal(str, dict)      # track_id, phase_data
    overload_detected = pyqtSignal(str, float)  # track_id, level_db
    
    def __init__(self, sample_rate: int = 44100, update_rate_hz: float = 30.0):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.update_rate_hz = update_rate_hz
        
        # Thread safety
        self._lock = RLock()
        
        # Meter instances per track
        self.track_meters: Dict[str, Dict[MeterType, ProfessionalMeter]] = {}
        self.track_spectrum_analyzers: Dict[str, SpectrumAnalyzer] = {}
        self.track_phase_meters: Dict[str, PhaseCorrelationMeter] = {}
        
        # Global monitoring settings
        self.overload_threshold_db = -0.1  # dBFS
        self.monitoring_enabled = True
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._emit_updates)
        self.update_timer.start(int(1000 / update_rate_hz))
        
        # Performance monitoring
        self.processing_load = 0.0
        self.max_processing_time = 0.0
    
    def add_track_monitoring(self, track_id: str, meter_types: List[MeterType] = None,
                           enable_spectrum: bool = True, enable_phase: bool = True):
        """Add monitoring for a track"""
        with self._lock:
            if meter_types is None:
                meter_types = [MeterType.PEAK, MeterType.RMS, MeterType.VU]
            
            # Create meters
            self.track_meters[track_id] = {}
            for meter_type in meter_types:
                ballistics = MeterBallistics.VU_ANALOG if meter_type == MeterType.VU else MeterBallistics.DIGITAL_PEAK
                meter = ProfessionalMeter(meter_type, ballistics, self.sample_rate, self.update_rate_hz)
                self.track_meters[track_id][meter_type] = meter
            
            # Create spectrum analyzer
            if enable_spectrum:
                self.track_spectrum_analyzers[track_id] = SpectrumAnalyzer(
                    fft_size=2048, sample_rate=self.sample_rate)
            
            # Create phase meter
            if enable_phase:
                self.track_phase_meters[track_id] = PhaseCorrelationMeter(self.sample_rate)
    
    def remove_track_monitoring(self, track_id: str):
        """Remove monitoring for a track"""
        with self._lock:
            self.track_meters.pop(track_id, None)
            self.track_spectrum_analyzers.pop(track_id, None)
            self.track_phase_meters.pop(track_id, None)
    
    def process_track_audio(self, track_id: str, audio_data: np.ndarray) -> Dict[str, Any]:
        """Process audio data for a track and update all monitors"""
        if not self.monitoring_enabled or track_id not in self.track_meters:
            return {}
        
        start_time = time.perf_counter()
        results = {}
        
        try:
            with self._lock:
                # Process through meters
                if track_id in self.track_meters:
                    meter_results = {}
                    for meter_type, meter in self.track_meters[track_id].items():
                        readings = meter.process_samples(audio_data)
                        meter_results[meter_type.value] = readings
                        
                        # Check for overloads
                        if readings['peak_db'] > self.overload_threshold_db:
                            self.overload_detected.emit(track_id, readings['peak_db'])
                    
                    results['meters'] = meter_results
                
                # Process through spectrum analyzer
                if track_id in self.track_spectrum_analyzers:
                    spectrum = self.track_spectrum_analyzers[track_id].process_samples(audio_data)
                    results['spectrum'] = spectrum
                
                # Process through phase meter (if stereo)
                if track_id in self.track_phase_meters and len(audio_data.shape) == 2:
                    phase_data = self.track_phase_meters[track_id].process_stereo_samples(
                        audio_data[:, 0], audio_data[:, 1])
                    results['phase'] = phase_data
                
                # Performance tracking
                processing_time = (time.perf_counter() - start_time) * 1000
                self.processing_load = processing_time
                self.max_processing_time = max(self.max_processing_time, processing_time)
                
                return results
                
        except Exception as e:
            print(f"Monitoring error for track {track_id}: {e}")
            return {}
    
    def _emit_updates(self):
        """Emit periodic updates for GUI"""
        if not self.monitoring_enabled:
            return
        
        try:
            with self._lock:
                # Emit meter updates
                for track_id, meters in self.track_meters.items():
                    meter_data = {}
                    for meter_type, meter in meters.items():
                        meter_data[meter_type.value] = meter.get_readings()
                    
                    self.level_update.emit(track_id, meter_data)
                
                # Emit spectrum updates
                for track_id, analyzer in self.track_spectrum_analyzers.items():
                    freq_bins, spectrum = analyzer.get_spectrum_data()
                    self.spectrum_update.emit(track_id, freq_bins, spectrum)
                
                # Emit phase updates
                for track_id, phase_meter in self.track_phase_meters.items():
                    if hasattr(phase_meter, 'correlation'):  # Only if data is available
                        phase_data = {
                            'correlation': phase_meter.correlation,
                            'phase_difference_degrees': phase_meter.phase_difference,
                            'stereo_width': phase_meter.stereo_width
                        }
                        self.phase_update.emit(track_id, phase_data)
                        
        except Exception as e:
            print(f"Monitoring update error: {e}")
    
    def reset_peak_holds(self, track_id: str = None):
        """Reset peak hold values"""
        with self._lock:
            if track_id:
                if track_id in self.track_meters:
                    for meter in self.track_meters[track_id].values():
                        meter.reset_peak_hold()
            else:
                # Reset all tracks
                for meters in self.track_meters.values():
                    for meter in meters.values():
                        meter.reset_peak_hold()
    
    def reset_over_counts(self, track_id: str = None):
        """Reset over/clipping counters"""
        with self._lock:
            if track_id:
                if track_id in self.track_meters:
                    for meter in self.track_meters[track_id].values():
                        meter.reset_over_count()
            else:
                # Reset all tracks
                for meters in self.track_meters.values():
                    for meter in meters.values():
                        meter.reset_over_count()
    
    def set_overload_threshold(self, threshold_db: float):
        """Set overload detection threshold"""
        self.overload_threshold_db = threshold_db
    
    def enable_monitoring(self, enabled: bool = True):
        """Enable/disable monitoring system"""
        self.monitoring_enabled = enabled
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'processing_load_ms': self.processing_load,
            'max_processing_time_ms': self.max_processing_time,
            'active_tracks': len(self.track_meters),
            'active_spectrum_analyzers': len(self.track_spectrum_analyzers),
            'active_phase_meters': len(self.track_phase_meters),
            'update_rate_hz': self.update_rate_hz
        }
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.max_processing_time = 0.0
        self.processing_load = 0.0