"""
MIDI Testing Suite - Comprehensive testing for all MIDI components
"""

import mido
import time
from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from typing import List, Dict, Any, Callable
import random
import threading


class MockRhythmWolf:
    """Mock Rhythm Wolf device for testing"""
    
    def __init__(self):
        self.current_pattern = 1
        self.is_playing = False
        self.current_bpm = 120.0
        self.parameters = {
            'kick_volume': 64,
            'kick_tune': 64,
            'snare_volume': 64,
            'bass_cutoff': 64,
            'tempo': 120
        }
        
    def generate_note_message(self, voice: str, velocity: int = 100):
        """Generate note on message for testing"""
        note_map = {
            'kick': 36,
            'snare': 38,
            'percussion': 40,
            'open_hihat': 42,
            'closed_hihat': 44
        }
        
        if voice in note_map:
            return mido.Message('note_on', 
                              channel=10,
                              note=note_map[voice], 
                              velocity=velocity)
        return None
        
    def generate_transport_message(self, command: str):
        """Generate transport message for testing"""
        if command == 'play':
            self.is_playing = True
            return mido.Message('start')
        elif command == 'stop':
            self.is_playing = False
            return mido.Message('stop')
        elif command == 'continue':
            self.is_playing = True
            return mido.Message('continue')
        return None
        
    def generate_cc_message(self, parameter: str, value: int):
        """Generate CC message for testing"""
        cc_map = {
            'kick_volume': (7, 10),
            'kick_tune': (74, 10),
            'snare_volume': (7, 10),
            'bass_cutoff': (71, 1),
            'tempo': (120, 1)
        }
        
        if parameter in cc_map:
            cc, channel = cc_map[parameter]
            self.parameters[parameter] = value
            return mido.Message('control_change',
                              channel=channel,
                              control=cc,
                              value=value)
        return None
        
    def generate_bpm_sequence(self, start_bmp: float, end_bpm: float, steps: int = 20):
        """Generate sequence of BPM changes for testing"""
        messages = []
        for i in range(steps):
            progress = i / (steps - 1)
            bpm = start_bmp + (end_bpm - start_bmp) * progress
            cc_value = int(((bpm - 60) / 140) * 127)  # Convert BPM to CC value
            msg = self.generate_cc_message('tempo', cc_value)
            if msg:
                messages.append((msg, i * 100))  # 100ms intervals
        return messages


class MIDIPerformanceTester(QObject):
    """Test MIDI performance and latency"""
    
    test_completed = pyqtSignal(str, dict)  # test_name, results
    
    def __init__(self, midi_handler):
        super().__init__()
        self.midi_handler = midi_handler
        self.mock_device = MockRhythmWolf()
        self.test_results = {}
        
    def test_message_throughput(self, message_count: int = 1000):
        """Test MIDI message processing throughput"""
        print(f"🧪 Testing message throughput ({message_count} messages)...")
        
        messages = []
        for i in range(message_count):
            # Generate random drum hits
            voice = random.choice(['kick', 'snare', 'percussion', 'open_hihat', 'closed_hihat'])
            velocity = random.randint(60, 127)
            msg = self.mock_device.generate_note_message(voice, velocity)
            if msg:
                messages.append(msg)
                
        # Measure processing time
        start_time = time.time()
        for msg in messages:
            self.midi_handler._on_midi_message(msg, 'rhythm_wolf')
        end_time = time.time()
        
        processing_time = (end_time - start_time) * 1000  # milliseconds
        throughput = message_count / (processing_time / 1000)  # messages per second
        avg_latency = processing_time / message_count  # ms per message
        
        results = {
            'message_count': message_count,
            'total_time_ms': processing_time,
            'throughput_msg_per_sec': throughput,
            'average_latency_ms': avg_latency,
            'target_throughput': 1000,  # Target: 1000 msg/sec
            'target_latency': 1.0,      # Target: <1ms per message
            'passed': throughput >= 1000 and avg_latency <= 1.0
        }
        
        self.test_results['throughput'] = results
        self.test_completed.emit('throughput', results)
        
        print(f"✅ Throughput test: {throughput:.0f} msg/sec, {avg_latency:.2f}ms avg latency")
        return results
        
    def test_bpm_sync_accuracy(self, bpm_changes: int = 50):
        """Test BPM sync accuracy and smoothing"""
        print(f"🧪 Testing BPM sync accuracy ({bmp_changes} changes)...")
        
        from ..midi.bpm_sync import RealTimeBPMSync
        from ..core.daw_engine import DAWEngine
        
        # Create test components
        daw = DAWEngine()
        bpm_sync = RealTimeBPMSync(daw)
        
        # Test data: gradual BPM changes
        test_bpms = []
        for i in range(bpm_changes):
            bpm = 80 + (i / bpm_changes) * 80  # 80-160 BPM range
            test_bpms.append(bpm)
            
        # Measure accuracy
        start_time = time.time()
        processed_bpms = []
        
        for target_bpm in test_bpms:
            cc_value = int(((target_bpm - 60) / 140) * 127)
            bpm_sync.handle_bpm_change(cc_value)
            actual_bpm = bpm_sync.get_current_bpm()
            processed_bpms.append(actual_bpm)
            
        processing_time = (time.time() - start_time) * 1000
        
        # Calculate accuracy metrics
        errors = [abs(target - actual) for target, actual in zip(test_bpms, processed_bpms)]
        avg_error = sum(errors) / len(errors)
        max_error = max(errors)
        
        results = {
            'bpm_changes': bpm_changes,
            'processing_time_ms': processing_time,
            'average_error_bpm': avg_error,
            'max_error_bmp': max_error,
            'target_avg_error': 0.5,     # Target: <0.5 BPM average error
            'target_max_error': 2.0,     # Target: <2.0 BPM max error
            'passed': avg_error <= 0.5 and max_error <= 2.0
        }
        
        self.test_results['bpm_accuracy'] = results
        self.test_completed.emit('bpm_accuracy', results)
        
        print(f"✅ BPM sync test: {avg_error:.2f} avg error, {max_error:.2f} max error")
        return results
        
    def test_auto_record_reliability(self, transport_cycles: int = 100):
        """Test auto-record reliability"""
        print(f"🧪 Testing auto-record reliability ({transport_cycles} cycles)...")
        
        from ..midi.auto_record import RhythmWolfAutoRecord
        from ..core.daw_engine import DAWEngine
        
        # Create test components
        daw = DAWEngine()
        auto_record = RhythmWolfAutoRecord(daw)
        
        # Track results
        successful_starts = 0
        successful_stops = 0
        errors = 0
        
        start_time = time.time()
        
        for i in range(transport_cycles):
            try:
                # Test start
                start_msg = self.mock_device.generate_transport_message('play')
                auto_record.handle_hardware_transport(start_msg)
                
                if daw.is_recording():
                    successful_starts += 1
                    
                # Small delay
                time.sleep(0.001)
                
                # Test stop
                stop_msg = self.mock_device.generate_transport_message('stop')
                auto_record.handle_hardware_transport(stop_msg)
                
                if not daw.is_recording():
                    successful_stops += 1
                    
            except Exception as e:
                errors += 1
                print(f"Auto-record error: {e}")
                
        processing_time = (time.time() - start_time) * 1000
        
        start_reliability = successful_starts / transport_cycles * 100
        stop_reliability = successful_stops / transport_cycles * 100
        overall_reliability = (successful_starts + successful_stops) / (transport_cycles * 2) * 100
        
        results = {
            'transport_cycles': transport_cycles,
            'processing_time_ms': processing_time,
            'start_reliability_%': start_reliability,
            'stop_reliability_%': stop_reliability,
            'overall_reliability_%': overall_reliability,
            'error_count': errors,
            'target_reliability': 99.0,  # Target: 99% reliability
            'passed': overall_reliability >= 99.0 and errors == 0
        }
        
        self.test_results['auto_record'] = results
        self.test_completed.emit('auto_record', results)
        
        print(f"✅ Auto-record test: {overall_reliability:.1f}% reliability, {errors} errors")
        return results
        
    def test_parameter_sync_performance(self, parameter_changes: int = 200):
        """Test bidirectional parameter synchronization"""
        print(f"🧪 Testing parameter sync ({parameter_changes} changes)...")
        
        from ..hardware.devices.akai.rhythm_wolf import RhythmWolfInterface
        
        # Create mock device interface
        rw = RhythmWolfInterface()
        rw.is_connected = True  # Mock connection
        
        # Test parameter changes
        parameters = ['kick_volume', 'kick_tune', 'snare_volume', 'bass_cutoff']
        sync_successes = 0
        sync_errors = 0
        
        start_time = time.time()
        
        for i in range(parameter_changes):
            param = random.choice(parameters)
            value = random.randint(0, 127)
            
            try:
                # Update software parameter
                rw.update_software_parameter(param, value)
                
                # Check if it matches
                actual_value = rw.get_parameter_value(param)
                
                if actual_value == value:
                    sync_successes += 1
                else:
                    sync_errors += 1
                    
            except Exception as e:
                sync_errors += 1
                print(f"Sync error: {e}")
                
        processing_time = (time.time() - start_time) * 1000
        sync_accuracy = sync_successes / parameter_changes * 100
        
        results = {
            'parameter_changes': parameter_changes,
            'processing_time_ms': processing_time,
            'sync_successes': sync_successes,
            'sync_errors': sync_errors,
            'sync_accuracy_%': sync_accuracy,
            'target_accuracy': 100.0,  # Target: 100% accuracy
            'passed': sync_accuracy == 100.0
        }
        
        self.test_results['parameter_sync'] = results
        self.test_completed.emit('parameter_sync', results)
        
        print(f"✅ Parameter sync test: {sync_accuracy:.1f}% accuracy")
        return results
        
    def test_stress_conditions(self, duration_seconds: int = 10):
        """Test system under stress conditions"""
        print(f"🧪 Running stress test ({duration_seconds} seconds)...")
        
        start_time = time.time()
        message_count = 0
        error_count = 0
        
        def stress_thread():
            nonlocal message_count, error_count
            
            while time.time() - start_time < duration_seconds:
                try:
                    # Generate rapid drum hits
                    voice = random.choice(['kick', 'snare', 'percussion'])
                    msg = self.mock_device.generate_note_message(voice)
                    if msg:
                        self.midi_handler._on_midi_message(msg, 'rhythm_wolf')
                        message_count += 1
                        
                    # Generate parameter changes
                    if random.random() < 0.1:  # 10% chance
                        param = random.choice(['kick_volume', 'bass_cutoff'])
                        msg = self.mock_device.generate_cc_message(param, random.randint(0, 127))
                        if msg:
                            self.midi_handler._on_midi_message(msg, 'rhythm_wolf')
                            message_count += 1
                            
                    # Generate transport commands occasionally  
                    if random.random() < 0.01:  # 1% chance
                        cmd = random.choice(['play', 'stop'])
                        msg = self.mock_device.generate_transport_message(cmd)
                        if msg:
                            self.midi_handler._on_midi_message(msg, 'rhythm_wolf')
                            message_count += 1
                            
                    time.sleep(0.001)  # 1ms between messages = 1000 msg/sec
                    
                except Exception as e:
                    error_count += 1
                    if error_count < 5:  # Don't spam errors
                        print(f"Stress test error: {e}")
                        
        # Run stress test in separate thread
        thread = threading.Thread(target=stress_thread)
        thread.start()
        thread.join()
        
        actual_duration = time.time() - start_time
        avg_msg_rate = message_count / actual_duration
        error_rate = error_count / message_count * 100 if message_count > 0 else 100
        
        results = {
            'duration_seconds': actual_duration,
            'total_messages': message_count,
            'average_msg_rate': avg_msg_rate,
            'error_count': error_count,
            'error_rate_%': error_rate,
            'target_msg_rate': 500,      # Target: 500+ msg/sec sustained
            'target_error_rate': 1.0,    # Target: <1% error rate
            'passed': avg_msg_rate >= 500 and error_rate <= 1.0
        }
        
        self.test_results['stress'] = results
        self.test_completed.emit('stress', results)
        
        print(f"✅ Stress test: {avg_msg_rate:.0f} msg/sec, {error_rate:.2f}% error rate")
        return results
        
    def run_all_tests(self):
        """Run complete test suite"""
        print("🧪 Running complete MIDI test suite...")
        
        all_results = {}
        test_start = time.time()
        
        # Run all tests
        tests = [
            ('throughput', lambda: self.test_message_throughput(1000)),
            ('bpm_accuracy', lambda: self.test_bpm_sync_accuracy(50)),
            ('auto_record', lambda: self.test_auto_record_reliability(50)),
            ('parameter_sync', lambda: self.test_parameter_sync_performance(100)),
            ('stress', lambda: self.test_stress_conditions(5))
        ]
        
        passed_tests = 0
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                all_results[test_name] = result
                if result.get('passed', False):
                    passed_tests += 1
                    print(f"✅ {test_name} PASSED")
                else:
                    print(f"❌ {test_name} FAILED")
            except Exception as e:
                print(f"❌ {test_name} ERROR: {e}")
                all_results[test_name] = {'error': str(e), 'passed': False}
                
        total_time = time.time() - test_start
        
        # Overall summary
        summary = {
            'total_tests': len(tests),
            'passed_tests': passed_tests,
            'failed_tests': len(tests) - passed_tests,
            'pass_rate_%': passed_tests / len(tests) * 100,
            'total_time_seconds': total_time,
            'overall_passed': passed_tests == len(tests)
        }
        
        all_results['summary'] = summary
        
        print(f"\n🧪 Test Suite Complete:")
        print(f"   Passed: {passed_tests}/{len(tests)} tests ({summary['pass_rate_%']:.1f}%)")
        print(f"   Total time: {total_time:.2f} seconds")
        
        if summary['overall_passed']:
            print("✅ ALL TESTS PASSED - MIDI system ready for production!")
        else:
            print("❌ SOME TESTS FAILED - Review issues before deployment")
            
        return all_results


class MIDIMockDataGenerator:
    """Generate realistic MIDI test data"""
    
    @staticmethod
    def generate_drum_pattern(bars: int = 4, bmp: int = 120):
        """Generate realistic drum pattern"""
        messages = []
        ms_per_beat = 60000 / bpm  # milliseconds per beat
        
        for bar in range(bars):
            for beat in range(4):  # 4/4 time
                timestamp = bar * 4 * ms_per_beat + beat * ms_per_beat
                
                # Kick on beats 1 and 3
                if beat in [0, 2]:
                    messages.append((
                        mido.Message('note_on', channel=10, note=36, velocity=100),
                        timestamp
                    ))
                    
                # Snare on beats 2 and 4
                if beat in [1, 3]:
                    messages.append((
                        mido.Message('note_on', channel=10, note=38, velocity=90),
                        timestamp
                    ))
                    
                # Hi-hat on every beat
                messages.append((
                    mido.Message('note_on', channel=10, note=44, velocity=70),
                    timestamp
                ))
                
        return messages
        
    @staticmethod
    def generate_parameter_automation(parameter: str, duration_ms: int = 8000):
        """Generate parameter automation curve"""
        messages = []
        cc_map = {'kick_volume': 7, 'bass_cutoff': 71, 'tempo': 120}
        
        if parameter not in cc_map:
            return messages
            
        cc = cc_map[parameter]
        points = 50
        
        for i in range(points):
            progress = i / (points - 1)
            timestamp = progress * duration_ms
            
            # Generate automation curve (sine wave)
            import math
            value = int(64 + 32 * math.sin(progress * math.pi * 2))
            
            messages.append((
                mido.Message('control_change', channel=1, control=cc, value=value),
                timestamp
            ))
            
        return messages