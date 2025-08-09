"""
MIDI Integration Example - Demonstrates how all MIDI components work together
"""

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QObject
import sys

# Import our MIDI components
from .midi_handler import CoreMIDIHandler
from ..hardware.hardware_manager import HardwareManager
from ..core.daw_engine import DAWEngine
from .midi_testing import MIDIPerformanceTester, MockRhythmWolf


class RhythmWolfMiniDAWExample(QObject):
    """Example showing complete MIDI integration"""
    
    def __init__(self):
        super().__init__()
        
        print("🎛️ Initializing Rhythm Wolf Mini-DAW MIDI System...")
        
        # Core components
        self.daw_engine = DAWEngine()
        self.hardware_manager = HardwareManager(self.daw_engine)
        
        # Connect signals for monitoring
        self.hardware_manager.device_connected.connect(self.on_device_connected)
        self.hardware_manager.device_disconnected.connect(self.on_device_disconnected)
        self.hardware_manager.hardware_event.connect(self.on_hardware_event)
        
        # Testing component
        self.tester = MIDIPerformanceTester(self.hardware_manager.midi_handler)
        self.tester.test_completed.connect(self.on_test_completed)
        
    def start_system(self):
        """Start the complete MIDI system"""
        print("🚀 Starting MIDI system...")
        
        # Start hardware management
        self.hardware_manager.start()
        
        print("✅ MIDI system started successfully!")
        print("   - Automatic device detection enabled")
        print("   - Auto-recording ready")
        print("   - BPM sync active") 
        print("   - Hardware control mapping active")
        
    def stop_system(self):
        """Clean shutdown of MIDI system"""
        print("🛑 Shutting down MIDI system...")
        self.hardware_manager.stop()
        print("✅ MIDI system stopped cleanly")
        
    def on_device_connected(self, device_type: str, device_info: dict):
        """Handle device connection"""
        print(f"🔌 Device Connected: {device_type}")
        print(f"   Port: {device_info.get('midi_port', 'Unknown')}")
        print(f"   Features: {device_info.get('capabilities', {})}")
        
        # Demonstrate device interaction
        if device_type == 'rhythm_wolf':
            self.demonstrate_rhythm_wolf_features(device_type)
            
    def on_device_disconnected(self, device_type: str):
        """Handle device disconnection"""
        print(f"🔌 Device Disconnected: {device_type}")
        
    def on_hardware_event(self, device_type: str, event_type: str, data):
        """Handle hardware events"""
        if event_type == 'note_triggered':
            print(f"🥁 {device_type} Note: Ch{data['channel']} Note{data['note']} Vel{data['velocity']}")
            
        elif event_type == 'parameter_changed':
            print(f"🎛️ {device_type} Parameter: {data['param']} = {data['value']}")
            
        elif event_type == 'tempo_changed':
            print(f"🎵 {device_type} BPM: {data:.1f}")
            
        elif event_type == 'recording_started':
            print(f"🔴 {device_type} Auto-Recording Started!")
            
        elif event_type == 'recording_stopped':
            print(f"⏹️ {device_type} Auto-Recording Stopped!")
            
    def demonstrate_rhythm_wolf_features(self, device_type: str):
        """Demonstrate Rhythm Wolf specific features"""
        print(f"🎛️ Demonstrating {device_type} features...")
        
        # Test parameter control
        print("   Setting kick volume to 100...")
        self.hardware_manager.set_device_parameter(device_type, 'kick_volume', 100)
        
        # Test note triggering
        print("   Triggering kick drum...")
        self.hardware_manager.trigger_device_note(device_type, 'kick', 120)
        
        # Enable features
        print("   Enabling auto-record and BPM sync...")
        self.hardware_manager.enable_auto_record(device_type, True)
        self.hardware_manager.enable_bpm_sync(device_type, True)
        
        # Get device parameters
        params = self.hardware_manager.get_device_parameters(device_type)
        if params:
            print(f"   Current parameters: {len(params)} controls active")
            
    def run_performance_tests(self):
        """Run comprehensive performance tests"""
        print("🧪 Running MIDI performance tests...")
        
        # Run all tests
        results = self.tester.run_all_tests()
        
        return results
        
    def on_test_completed(self, test_name: str, results: dict):
        """Handle test completion"""
        status = "✅ PASSED" if results.get('passed', False) else "❌ FAILED"
        print(f"🧪 Test '{test_name}' completed: {status}")
        
    def demonstrate_mock_device(self):
        """Demonstrate using mock device for testing"""
        print("🤖 Demonstrating mock device...")
        
        mock = MockRhythmWolf()
        
        # Generate test messages
        print("   Generating drum pattern...")
        for voice in ['kick', 'snare', 'kick', 'snare']:
            msg = mock.generate_note_message(voice)
            if msg:
                print(f"      Mock {voice}: {msg}")
                
        # Generate parameter changes
        print("   Generating parameter changes...")
        for param in ['kick_volume', 'bass_cutoff']:
            msg = mock.generate_cc_message(param, 80)
            if msg:
                print(f"      Mock {param}: {msg}")
                
        # Generate transport commands
        print("   Generating transport commands...")
        for cmd in ['play', 'stop']:
            msg = mock.generate_transport_message(cmd)
            if msg:
                print(f"      Mock {cmd}: {msg}")
                
    def get_system_status(self):
        """Get comprehensive system status"""
        connected_devices = self.hardware_manager.get_connected_devices()
        performance_stats = self.hardware_manager.get_performance_stats()
        
        status = {
            'connected_devices': connected_devices,
            'daw_playing': self.daw_engine.is_playing(),
            'daw_recording': self.daw_engine.is_recording(),
            'master_tempo': self.daw_engine.get_master_tempo(),
            'performance_stats': performance_stats
        }
        
        return status


def main():
    """Main example function"""
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
        
    # Create example system
    example = RhythmWolfMiniDAWExample()
    
    try:
        # Start the system
        example.start_system()
        
        # Demonstrate mock device (useful for testing without hardware)
        example.demonstrate_mock_device()
        
        # Run performance tests
        print("\n" + "="*60)
        test_results = example.run_performance_tests()
        print("="*60)
        
        # Display system status
        status = example.get_system_status()
        print(f"\n📊 System Status:")
        print(f"   Connected devices: {len(status['connected_devices'])}")
        print(f"   DAW playing: {status['daw_playing']}")
        print(f"   DAW recording: {status['daw_recording']}")
        print(f"   Master tempo: {status['master_tempo']} BPM")
        
        # Show performance stats
        perf = status['performance_stats']
        print(f"   MIDI messages processed: {perf['total_midi_messages']}")
        print(f"   Active devices: {perf['connected_devices']}")
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Clean shutdown
        example.stop_system()
        
    print("👋 Example complete!")
    

if __name__ == '__main__':
    main()