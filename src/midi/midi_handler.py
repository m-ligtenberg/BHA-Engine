"""
Core MIDI Handler - Professional-grade MIDI I/O with low-latency threading
"""

import mido
import time
from typing import Dict, List, Optional, Callable, Any
from threading import Lock, RLock
from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal, QMutex, QMutexLocker

class MIDIDeviceManager(QObject):
    """Manages MIDI device detection, connection, and monitoring"""
    
    device_connected = pyqtSignal(str, str)  # device_name, port_name
    device_disconnected = pyqtSignal(str)    # device_name
    devices_refreshed = pyqtSignal(list)     # list of available devices
    
    def __init__(self):
        super().__init__()
        self.connected_devices: Dict[str, Dict[str, Any]] = {}
        self.device_patterns = {
            'rhythm_wolf': [
                "Rhythm Wolf", "RHYTHM WOLF", "Akai Rhythm Wolf", "RW"
            ],
            'td3': [
                "TD-3", "BEHRINGER TD-3", "Behringer TD-3"
            ],
            'rd6': [
                "RD-6", "BEHRINGER RD-6", "Behringer RD-6" 
            ]
        }
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_devices)
        self.mutex = QMutex()
        
    def start_monitoring(self, interval_ms: int = 2000):
        """Start automatic device monitoring"""
        self.refresh_timer.start(interval_ms)
        self.refresh_devices()
        
    def stop_monitoring(self):
        """Stop device monitoring"""
        self.refresh_timer.stop()
        
    def refresh_devices(self):
        """Scan for available MIDI devices"""
        with QMutexLocker(self.mutex):
            try:
                input_ports = mido.get_input_names()
                output_ports = mido.get_output_names()
                
                current_devices = {}
                
                # Check each device pattern
                for device_type, patterns in self.device_patterns.items():
                    for port_name in input_ports:
                        for pattern in patterns:
                            if pattern.lower() in port_name.lower():
                                has_output = port_name in output_ports
                                
                                device_info = {
                                    'type': device_type,
                                    'input_port': port_name,
                                    'output_port': port_name if has_output else None,
                                    'bidirectional': has_output
                                }
                                current_devices[device_type] = device_info
                                
                                # Emit connection signal if new
                                if device_type not in self.connected_devices:
                                    self.device_connected.emit(device_type, port_name)
                
                # Check for disconnected devices
                for device_type in list(self.connected_devices.keys()):
                    if device_type not in current_devices:
                        self.device_disconnected.emit(device_type)
                        
                self.connected_devices = current_devices
                self.devices_refreshed.emit(list(current_devices.keys()))
                
            except Exception as e:
                print(f"Error refreshing MIDI devices: {e}")
                
    def get_device_info(self, device_type: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific device"""
        with QMutexLocker(self.mutex):
            return self.connected_devices.get(device_type)
            
    def is_device_connected(self, device_type: str) -> bool:
        """Check if a device is currently connected"""
        with QMutexLocker(self.mutex):
            return device_type in self.connected_devices


class MIDIMessageBuffer:
    """Lock-free circular buffer for MIDI messages"""
    
    def __init__(self, size: int = 1024):
        self.size = size
        self.buffer = [None] * size
        self.write_index = 0
        self.read_index = 0
        self.lock = RLock()
        
    def put(self, message) -> bool:
        """Add message to buffer. Returns False if buffer is full."""
        with self.lock:
            next_write = (self.write_index + 1) % self.size
            if next_write == self.read_index:
                return False  # Buffer full
                
            self.buffer[self.write_index] = message
            self.write_index = next_write
            return True
            
    def get(self):
        """Get next message from buffer. Returns None if empty."""
        with self.lock:
            if self.read_index == self.write_index:
                return None  # Buffer empty
                
            message = self.buffer[self.read_index]
            self.read_index = (self.read_index + 1) % self.size
            return message
            
    def get_all(self) -> List:
        """Get all pending messages"""
        messages = []
        while True:
            msg = self.get()
            if msg is None:
                break
            messages.append(msg)
        return messages
        
    def clear(self):
        """Clear all messages from buffer"""
        with self.lock:
            self.read_index = self.write_index


class HighPriorityMIDIThread(QThread):
    """High-priority thread for MIDI input processing"""
    
    message_received = pyqtSignal(object, str)  # message, device_type
    error_occurred = pyqtSignal(str, str)       # error_msg, device_type
    
    def __init__(self, device_type: str, input_port_name: str):
        super().__init__()
        self.device_type = device_type
        self.input_port_name = input_port_name
        self.input_port = None
        self.running = False
        self.message_buffer = MIDIMessageBuffer(2048)
        self.error_count = 0
        self.max_errors = 10
        
    def run(self):
        """Main MIDI input loop - runs at high priority"""
        self.running = True
        self.error_count = 0
        
        try:
            # Open MIDI input port
            self.input_port = mido.open_input(
                self.input_port_name,
                callback=self._midi_callback
            )
            
            print(f"✅ MIDI Input Thread Started: {self.input_port_name}")
            
            # Process buffered messages at regular intervals
            while self.running:
                messages = self.message_buffer.get_all()
                for message in messages:
                    if self.running:
                        self.message_received.emit(message, self.device_type)
                
                # Small sleep to prevent excessive CPU usage
                self.msleep(1)  # 1ms sleep = 1000Hz processing rate
                
        except Exception as e:
            self.error_occurred.emit(f"MIDI thread error: {e}", self.device_type)
        finally:
            self._cleanup()
            
    def _midi_callback(self, message):
        """Low-latency MIDI callback - minimal processing"""
        if not self.message_buffer.put(message):
            # Buffer overflow - drop oldest messages
            self.message_buffer.clear()
            self.message_buffer.put(message)
            
    def _cleanup(self):
        """Clean up resources"""
        if self.input_port:
            try:
                self.input_port.close()
            except:
                pass
            self.input_port = None
            
    def stop(self):
        """Stop MIDI processing"""
        self.running = False
        self.quit()
        self.wait(5000)  # Wait up to 5 seconds
        self._cleanup()


class MIDIOutputManager(QObject):
    """Manages MIDI output with message queuing and error handling"""
    
    message_sent = pyqtSignal(object, str)     # message, device_type
    send_error = pyqtSignal(str, str)          # error_msg, device_type
    
    def __init__(self):
        super().__init__()
        self.output_ports: Dict[str, mido.ports.BaseOutput] = {}
        self.send_queues: Dict[str, List] = {}
        self.mutex = QMutex()
        
    def add_device(self, device_type: str, port_name: str) -> bool:
        """Add output device"""
        with QMutexLocker(self.mutex):
            try:
                if device_type not in self.output_ports:
                    port = mido.open_output(port_name)
                    self.output_ports[device_type] = port
                    self.send_queues[device_type] = []
                    print(f"✅ MIDI Output Added: {port_name}")
                    return True
            except Exception as e:
                self.send_error.emit(f"Failed to open output: {e}", device_type)
                return False
                
    def remove_device(self, device_type: str):
        """Remove output device"""
        with QMutexLocker(self.mutex):
            if device_type in self.output_ports:
                try:
                    self.output_ports[device_type].close()
                except:
                    pass
                del self.output_ports[device_type]
                del self.send_queues[device_type]
                
    def send_message(self, device_type: str, message, priority: int = 0) -> bool:
        """Send MIDI message with optional priority"""
        with QMutexLocker(self.mutex):
            if device_type not in self.output_ports:
                return False
                
            try:
                port = self.output_ports[device_type]
                if priority > 0:
                    # High priority - send immediately
                    port.send(message)
                else:
                    # Normal priority - queue for batch sending
                    self.send_queues[device_type].append(message)
                    
                self.message_sent.emit(message, device_type)
                return True
                
            except Exception as e:
                self.send_error.emit(f"Send failed: {e}", device_type)
                return False
                
    def flush_queues(self):
        """Send all queued messages"""
        with QMutexLocker(self.mutex):
            for device_type, queue in self.send_queues.items():
                if queue and device_type in self.output_ports:
                    port = self.output_ports[device_type]
                    try:
                        for message in queue:
                            port.send(message)
                        queue.clear()
                    except Exception as e:
                        self.send_error.emit(f"Queue flush failed: {e}", device_type)
                        queue.clear()


class CoreMIDIHandler(QObject):
    """Core MIDI handler coordinating all MIDI I/O operations"""
    
    # High-level signals
    device_connected = pyqtSignal(str)      # device_type
    device_disconnected = pyqtSignal(str)   # device_type
    midi_message = pyqtSignal(object, str)  # message, device_type
    
    def __init__(self):
        super().__init__()
        
        # Core components
        self.device_manager = MIDIDeviceManager()
        self.output_manager = MIDIOutputManager()
        
        # Active input threads
        self.input_threads: Dict[str, HighPriorityMIDIThread] = {}
        
        # Message handlers
        self.message_handlers: Dict[str, List[Callable]] = {}
        
        # Performance monitoring
        self.message_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
        # Connect device manager signals
        self.device_manager.device_connected.connect(self._on_device_connected)
        self.device_manager.device_disconnected.connect(self._on_device_disconnected)
        
        # Flush timer for queued messages
        self.flush_timer = QTimer()
        self.flush_timer.timeout.connect(self.output_manager.flush_queues)
        self.flush_timer.start(10)  # Flush every 10ms for low latency
        
    def start(self):
        """Start MIDI handling"""
        print("🎹 Starting MIDI Handler...")
        self.device_manager.start_monitoring()
        
    def stop(self):
        """Stop MIDI handling"""
        print("🛑 Stopping MIDI Handler...")
        self.device_manager.stop_monitoring()
        self.flush_timer.stop()
        
        # Stop all input threads
        for thread in self.input_threads.values():
            thread.stop()
        self.input_threads.clear()
        
    def _on_device_connected(self, device_type: str, port_name: str):
        """Handle device connection"""
        print(f"🔌 Device connected: {device_type} ({port_name})")
        
        # Start input thread
        if device_type not in self.input_threads:
            thread = HighPriorityMIDIThread(device_type, port_name)
            thread.message_received.connect(self._on_midi_message)
            thread.error_occurred.connect(self._on_thread_error)
            thread.start()
            
            self.input_threads[device_type] = thread
            
        # Setup output
        device_info = self.device_manager.get_device_info(device_type)
        if device_info and device_info['bidirectional']:
            self.output_manager.add_device(device_type, port_name)
            
        self.device_connected.emit(device_type)
        
    def _on_device_disconnected(self, device_type: str):
        """Handle device disconnection"""
        print(f"🔌 Device disconnected: {device_type}")
        
        # Stop input thread
        if device_type in self.input_threads:
            self.input_threads[device_type].stop()
            del self.input_threads[device_type]
            
        # Remove output
        self.output_manager.remove_device(device_type)
        
        self.device_disconnected.emit(device_type)
        
    def _on_midi_message(self, message, device_type: str):
        """Handle incoming MIDI message"""
        self.message_count += 1
        
        # Forward to registered handlers
        if device_type in self.message_handlers:
            for handler in self.message_handlers[device_type]:
                try:
                    handler(message)
                except Exception as e:
                    print(f"Message handler error: {e}")
                    self.error_count += 1
                    
        # Emit general signal
        self.midi_message.emit(message, device_type)
        
    def _on_thread_error(self, error_msg: str, device_type: str):
        """Handle thread errors"""
        print(f"❌ MIDI Thread Error ({device_type}): {error_msg}")
        self.error_count += 1
        
    def register_message_handler(self, device_type: str, handler: Callable):
        """Register a message handler for a device type"""
        if device_type not in self.message_handlers:
            self.message_handlers[device_type] = []
        self.message_handlers[device_type].append(handler)
        
    def unregister_message_handler(self, device_type: str, handler: Callable):
        """Unregister a message handler"""
        if device_type in self.message_handlers:
            try:
                self.message_handlers[device_type].remove(handler)
            except ValueError:
                pass
                
    def send_message(self, device_type: str, message, priority: int = 0) -> bool:
        """Send MIDI message to device"""
        return self.output_manager.send_message(device_type, message, priority)
        
    def is_device_connected(self, device_type: str) -> bool:
        """Check if device is connected"""
        return self.device_manager.is_device_connected(device_type)
        
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        runtime = time.time() - self.start_time
        return {
            'runtime_seconds': runtime,
            'messages_per_second': self.message_count / max(runtime, 1),
            'error_rate': self.error_count / max(self.message_count, 1),
            'total_messages': self.message_count,
            'total_errors': self.error_count
        }