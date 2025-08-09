"""
Hardware Manager - Centralized coordination for all hardware devices
"""

from PyQt6.QtCore import QObject, pyqtSignal, QTimer
from typing import Dict, List, Optional, Any, Type
import time

from .devices.akai.rhythm_wolf import RhythmWolfInterface
from ..midi.midi_handler import CoreMIDIHandler
from ..midi.auto_record import RhythmWolfAutoRecord, AdvancedAutoRecord
from ..midi.bpm_sync import RealTimeBPMSync


class HardwareManager(QObject):
    """Centralized hardware management and coordination"""
    
    # High-level signals
    device_connected = pyqtSignal(str, dict)      # device_type, device_info
    device_disconnected = pyqtSignal(str)         # device_type
    hardware_event = pyqtSignal(str, str, object) # device_type, event_type, data
    sync_status_changed = pyqtSignal(bool)        # all_devices_synced
    
    def __init__(self, daw_engine):
        super().__init__()
        
        self.daw = daw_engine
        
        # Core components
        self.midi_handler = CoreMIDIHandler()
        
        # Device registry
        self.connected_devices: Dict[str, Any] = {}
        self.device_classes = {
            'rhythm_wolf': RhythmWolfInterface,
            # Future devices can be added here
            # 'td3': BehringerTD3Interface,
            # 'rd6': BehringerRD6Interface,
        }
        
        # Feature managers (per device type)
        self.auto_record_managers: Dict[str, Any] = {}
        self.bpm_sync_managers: Dict[str, Any] = {}
        
        # Global settings
        self.global_auto_record = True
        self.global_bpm_sync = True
        self.master_sync_device = None  # Device that controls master tempo
        
        # Performance monitoring
        self.total_midi_messages = 0
        self.device_error_counts: Dict[str, int] = {}
        self.last_activity_times: Dict[str, float] = {}
        
        # Health monitoring
        self.health_check_timer = QTimer()
        self.health_check_timer.timeout.connect(self._check_device_health)
        self.health_check_timer.start(5000)  # Check every 5 seconds
        
        # Connect core MIDI handler
        self.midi_handler.device_connected.connect(self._on_device_connected)
        self.midi_handler.device_disconnected.connect(self._on_device_disconnected)
        self.midi_handler.midi_message.connect(self._on_midi_message)
        
    def start(self):
        """Start hardware management"""
        print("🎛️ Starting Hardware Manager...")
        self.midi_handler.start()
        
    def stop(self):
        """Stop hardware management and cleanup"""
        print("🛑 Stopping Hardware Manager...")
        self.health_check_timer.stop()
        
        # Clean shutdown of all devices
        for device_type, device in self.connected_devices.items():
            try:
                if hasattr(device, 'close_connection'):
                    device.close_connection()
            except Exception as e:
                print(f"Error closing {device_type}: {e}")
                
        self.connected_devices.clear()
        self.auto_record_managers.clear()
        self.bpm_sync_managers.clear()
        
        self.midi_handler.stop()
        
    def _on_device_connected(self, device_type: str):
        """Handle device connection"""
        if device_type not in self.device_classes:
            print(f"⚠️ Unknown device type: {device_type}")
            return
            
        try:
            # Create device interface
            device_class = self.device_classes[device_type]
            device = device_class()
            
            # Initialize connection
            if device.detect_device() and device.initialize_connection():
                self.connected_devices[device_type] = device
                
                # Setup device-specific features
                self._setup_device_features(device_type, device)
                
                # Connect device signals
                self._connect_device_signals(device_type, device)
                
                # Get device info and emit signal
                device_info = device.get_device_info()
                self.device_connected.emit(device_type, device_info)
                
                print(f"✅ {device_type} fully initialized")
                
                # Track activity
                self.last_activity_times[device_type] = time.time()
                
            else:
                print(f"❌ Failed to initialize {device_type}")
                
        except Exception as e:
            print(f"❌ Error connecting {device_type}: {e}")
            self.device_error_counts[device_type] = self.device_error_counts.get(device_type, 0) + 1
            
    def _on_device_disconnected(self, device_type: str):
        """Handle device disconnection"""
        if device_type in self.connected_devices:
            device = self.connected_devices[device_type]
            
            # Cleanup device-specific features
            if device_type in self.auto_record_managers:
                del self.auto_record_managers[device_type]
                
            if device_type in self.bpm_sync_managers:
                del self.bpm_sync_managers[device_type]
                
            # Close device connection
            try:
                if hasattr(device, 'close_connection'):
                    device.close_connection()
            except:
                pass
                
            del self.connected_devices[device_type]
            
            # Update master sync device if needed
            if self.master_sync_device == device_type:
                self.master_sync_device = None
                self._select_new_sync_master()
                
            self.device_disconnected.emit(device_type)
            print(f"🔌 {device_type} disconnected")
            
    def _setup_device_features(self, device_type: str, device):
        """Setup device-specific features like auto-record and BPM sync"""
        
        # Auto-record feature
        if device_type == 'rhythm_wolf':
            if self.global_auto_record:
                auto_record = AdvancedAutoRecord(self.daw)
                
                # Connect to device transport signals
                device.transport_changed.connect(auto_record.handle_hardware_transport)
                
                # Connect status signals
                auto_record.recording_started.connect(
                    lambda: self.hardware_event.emit(device_type, 'recording_started', None)
                )
                auto_record.recording_stopped.connect(
                    lambda: self.hardware_event.emit(device_type, 'recording_stopped', None)
                )
                
                self.auto_record_managers[device_type] = auto_record
                
        # BPM sync feature
        if self.global_bpm_sync:
            bpm_sync = RealTimeBPMSync(self.daw)
            
            # Connect to device BPM signals
            device.bpm_changed.connect(bpm_sync.handle_bmp_change)
            
            # Connect tempo change signals
            bpm_sync.tempo_changed.connect(
                lambda bpm: self.hardware_event.emit(device_type, 'tempo_changed', bpm)
            )
            
            self.bpm_sync_managers[device_type] = bpm_sync
            
            # Set as master sync device if none exists
            if not self.master_sync_device:
                self.master_sync_device = device_type
                print(f"🎵 {device_type} set as master sync device")
                
    def _connect_device_signals(self, device_type: str, device):
        """Connect device signals to hardware manager"""
        
        # Parameter changes
        if hasattr(device, 'parameter_changed'):
            device.parameter_changed.connect(
                lambda param, value: self.hardware_event.emit(device_type, 'parameter_changed', {'param': param, 'value': value})
            )
            
        # Note triggers
        if hasattr(device, 'note_triggered'):
            device.note_triggered.connect(
                lambda ch, note, vel: self.hardware_event.emit(device_type, 'note_triggered', {'channel': ch, 'note': note, 'velocity': vel})
            )
            
        # Pattern changes
        if hasattr(device, 'pattern_changed'):
            device.pattern_changed.connect(
                lambda pattern: self.hardware_event.emit(device_type, 'pattern_changed', pattern)
            )
            
    def _on_midi_message(self, message, device_type: str):
        """Handle incoming MIDI messages"""
        self.total_midi_messages += 1
        self.last_activity_times[device_type] = time.time()
        
        # Forward to specific device if connected
        if device_type in self.connected_devices:
            device = self.connected_devices[device_type]
            try:
                device.handle_incoming_midi(message)
            except Exception as e:
                print(f"Error handling MIDI for {device_type}: {e}")
                self.device_error_counts[device_type] = self.device_error_counts.get(device_type, 0) + 1
                
    def _check_device_health(self):
        """Monitor device health and connectivity"""
        current_time = time.time()
        
        for device_type in list(self.connected_devices.keys()):
            last_activity = self.last_activity_times.get(device_type, current_time)
            
            # Check for devices that haven't been active
            if current_time - last_activity > 30:  # 30 seconds of inactivity
                print(f"⚠️ {device_type} seems inactive (no MIDI for {current_time - last_activity:.1f}s)")
                
            # Check error rates
            error_count = self.device_error_counts.get(device_type, 0)
            if error_count > 10:
                print(f"🚨 {device_type} has high error count: {error_count}")
                
    def _select_new_sync_master(self):
        """Select a new master sync device"""
        # Prefer Rhythm Wolf for master sync
        if 'rhythm_wolf' in self.connected_devices:
            self.master_sync_device = 'rhythm_wolf'
        elif self.connected_devices:
            # Use first available device
            self.master_sync_device = list(self.connected_devices.keys())[0]
        else:
            self.master_sync_device = None
            
        if self.master_sync_device:
            print(f"🎵 {self.master_sync_device} selected as new master sync device")
            
    # Public API methods
    
    def get_connected_devices(self) -> List[str]:
        """Get list of connected device types"""
        return list(self.connected_devices.keys())
        
    def is_device_connected(self, device_type: str) -> bool:
        """Check if specific device type is connected"""
        return device_type in self.connected_devices
        
    def get_device(self, device_type: str) -> Optional[Any]:
        """Get specific device interface"""
        return self.connected_devices.get(device_type)
        
    def send_to_device(self, device_type: str, message, priority: int = 0) -> bool:
        """Send MIDI message to specific device"""
        return self.midi_handler.send_message(device_type, message, priority)
        
    def trigger_device_note(self, device_type: str, voice: str, velocity: int = 127) -> bool:
        """Trigger a note on a specific device"""
        device = self.connected_devices.get(device_type)
        if device and hasattr(device, 'trigger_drum_voice'):
            return device.trigger_drum_voice(voice, velocity)
        return False
        
    def set_device_parameter(self, device_type: str, parameter: str, value: int) -> bool:
        """Set parameter on specific device"""
        device = self.connected_devices.get(device_type)
        if device and hasattr(device, 'update_software_parameter'):
            device.update_software_parameter(parameter, value)
            return True
        return False
        
    def get_device_parameters(self, device_type: str) -> Optional[Dict[str, int]]:
        """Get all parameters from specific device"""
        device = self.connected_devices.get(device_type)
        if device and hasattr(device, 'get_all_parameters'):
            return device.get_all_parameters()
        return None
        
    def enable_auto_record(self, device_type: str, enabled: bool):
        """Enable/disable auto-record for specific device"""
        if device_type in self.auto_record_managers:
            self.auto_record_managers[device_type].set_auto_record_enabled(enabled)
            
    def enable_bpm_sync(self, device_type: str, enabled: bool):
        """Enable/disable BPM sync for specific device"""
        if device_type in self.bpm_sync_managers:
            self.bpm_sync_managers[device_type].set_bmp_sync_enabled(enabled)
            
    def set_master_sync_device(self, device_type: str):
        """Set specific device as master sync source"""
        if device_type in self.connected_devices:
            # Disable BPM sync for current master
            if self.master_sync_device and self.master_sync_device in self.bpm_sync_managers:
                self.bpm_sync_managers[self.master_sync_device].set_bpm_sync_enabled(False)
                
            # Set new master
            self.master_sync_device = device_type
            if device_type in self.bpm_sync_managers:
                self.bmp_sync_managers[device_type].set_bpm_sync_enabled(True)
                
            print(f"🎵 {device_type} set as master sync device")
            
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = {
            'total_midi_messages': self.total_midi_messages,
            'connected_devices': len(self.connected_devices),
            'device_types': list(self.connected_devices.keys()),
            'master_sync_device': self.master_sync_device,
            'device_errors': self.device_error_counts.copy(),
            'midi_handler_stats': self.midi_handler.get_performance_stats()
        }
        
        # Add per-device stats
        for device_type, device in self.connected_devices.items():
            if hasattr(device, 'get_device_info'):
                stats[f'{device_type}_info'] = device.get_device_info()
                
        return stats
        
    def reset_error_counts(self):
        """Reset all device error counts"""
        self.device_error_counts.clear()
        
        # Reset individual component error counts
        for auto_record in self.auto_record_managers.values():
            if hasattr(auto_record, 'reset_error_count'):
                auto_record.reset_error_count()
                
        print("✅ All error counts reset")