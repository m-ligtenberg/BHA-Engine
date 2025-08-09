import mido
from PyQt6.QtCore import QObject, QThread, pyqtSignal, QMutex, QTimer
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Any
import time

class BaseHardware(ABC):
    """Abstract base class for all hardware integrations"""

    def __init__(self, device_name, midi_port=None):
        self.device_name = device_name
        self.midi_port = midi_port
        self.is_connected = False
        self.capabilities = {}
        self.parameter_mappings = {}

    @abstractmethod
    def detect_device(self):
        """Auto-detect if device is connected"""
        pass

    @abstractmethod  
    def initialize_connection(self):
        """Establish communication with device"""
        pass

    @abstractmethod
    def get_control_mappings(self):
        """Return dict of available controls and their MIDI mappings"""
        pass

    @abstractmethod
    def handle_incoming_midi(self, message):
        """Process MIDI messages from device"""
        pass

    @abstractmethod
    def send_parameter_change(self, parameter, value):
        """Send parameter changes to device"""
        pass

class RhythmWolfInterface(BaseHardware, QObject):
    """Akai Rhythm Wolf hardware interface with complete MIDI specification"""

    # Signals for hardware events
    parameter_changed = pyqtSignal(str, int)  # parameter_name, value
    note_triggered = pyqtSignal(int, int, int)  # channel, note, velocity
    transport_changed = pyqtSignal(str)  # play/stop/continue
    bpm_changed = pyqtSignal(float)  # BPM value
    pattern_changed = pyqtSignal(int)  # pattern_number
    sync_mode_changed = pyqtSignal(str)  # sync_mode

    def __init__(self):
        BaseHardware.__init__(self, "Akai Rhythm Wolf")
        QObject.__init__(self)

        self.capabilities = {
            'type': 'drum_machine',
            'channels': [1, 10],  # Bass synth on 1, drums on 10
            'has_sequencer': True,
            'has_knobs': True,
            'has_transport': True,
            'sync_modes': ['internal', 'midi', 'din_sync']
        }

        # MIDI port settings
        self.input_port = None
        self.output_port = None
        self.port_mutex = QMutex()
        
        # Hardware state tracking
        self.hardware_parameters = {}
        self.software_parameters = {}
        self.sync_enabled = True
        self.last_sync_time = 0
        
        # Pattern and sync state
        self.current_pattern = 1
        self.sync_mode = 'internal'
        self.is_playing = False
        
        # Performance tracking
        self.message_latency_ms = 0
        self.sync_errors = 0
        
        # Bidirectional sync timer
        self.sync_timer = QTimer()
        self.sync_timer.timeout.connect(self._sync_parameters)
        
        # Initialize parameter mappings
        self._initialize_parameter_state()

    def detect_device(self):
        """Auto-detect Rhythm Wolf connection"""
        available_ports = mido.get_input_names()

        # Look for Rhythm Wolf in available ports
        rhythm_wolf_patterns = [
            "Rhythm Wolf",
            "RHYTHM WOLF", 
            "Akai Rhythm Wolf",
            "RW"  # Some systems might abbreviate
        ]

        for port_name in available_ports:
            for pattern in rhythm_wolf_patterns:
                if pattern.lower() in port_name.lower():
                    self.midi_port = port_name
                    return True

        return False

    def initialize_connection(self):
        """Establish MIDI connection with Rhythm Wolf"""
        try:
            with self.port_mutex:
                if self.midi_port:
                    # Open input port for receiving MIDI
                    self.input_port = mido.open_input(self.midi_port)

                    # Try to open output port (same name usually)
                    output_ports = mido.get_output_names()
                    if self.midi_port in output_ports:
                        self.output_port = mido.open_output(self.midi_port)

                    self.is_connected = True
                    print(f"✅ Connected to Rhythm Wolf: {self.midi_port}")
                    return True

        except Exception as e:
            print(f"❌ Failed to connect to Rhythm Wolf: {e}")
            self.is_connected = False
            return False

    def get_control_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Complete Rhythm Wolf MIDI CC mappings with extended specification"""
        return {
            # Kick Drum (Channel 10, Note 36/C2)
            'kick_volume': {'cc': 7, 'range': (0, 127), 'channel': 10, 'note': 36, 'type': 'continuous'},
            'kick_tune': {'cc': 74, 'range': (0, 127), 'channel': 10, 'note': 36, 'type': 'continuous'},
            'kick_attack': {'cc': 73, 'range': (0, 127), 'channel': 10, 'note': 36, 'type': 'continuous'},
            'kick_decay': {'cc': 72, 'range': (0, 127), 'channel': 10, 'note': 36, 'type': 'continuous'},

            # Snare Drum (Channel 10, Note 38/D2)
            'snare_volume': {'cc': 7, 'range': (0, 127), 'channel': 10, 'note': 38, 'type': 'continuous'},
            'snare_tune': {'cc': 74, 'range': (0, 127), 'channel': 10, 'note': 38, 'type': 'continuous'},
            'snare_noise': {'cc': 71, 'range': (0, 127), 'channel': 10, 'note': 38, 'type': 'continuous'},
            'snare_decay': {'cc': 72, 'range': (0, 127), 'channel': 10, 'note': 38, 'type': 'continuous'},

            # Percussion (Channel 10, Note 40/E2)
            'perc_volume': {'cc': 7, 'range': (0, 127), 'channel': 10, 'note': 40, 'type': 'continuous'},
            'perc_tune': {'cc': 74, 'range': (0, 127), 'channel': 10, 'note': 40, 'type': 'continuous'},
            'perc_high': {'cc': 75, 'range': (0, 127), 'channel': 10, 'note': 40, 'type': 'continuous'},
            'perc_low': {'cc': 76, 'range': (0, 127), 'channel': 10, 'note': 40, 'type': 'continuous'},
            'perc_noise': {'cc': 71, 'range': (0, 127), 'channel': 10, 'note': 40, 'type': 'continuous'},
            'perc_decay': {'cc': 72, 'range': (0, 127), 'channel': 10, 'note': 40, 'type': 'continuous'},

            # Open Hi-Hat (Channel 10, Note 42/F#2)
            'open_hihat_volume': {'cc': 7, 'range': (0, 127), 'channel': 10, 'note': 42, 'type': 'continuous'},
            'open_hihat_tune': {'cc': 74, 'range': (0, 127), 'channel': 10, 'note': 42, 'type': 'continuous'},
            'open_hihat_decay': {'cc': 72, 'range': (0, 127), 'channel': 10, 'note': 42, 'type': 'continuous'},

            # Closed Hi-Hat (Channel 10, Note 44/G#2)
            'closed_hihat_volume': {'cc': 7, 'range': (0, 127), 'channel': 10, 'note': 44, 'type': 'continuous'},
            'closed_hihat_tune': {'cc': 74, 'range': (0, 127), 'channel': 10, 'note': 44, 'type': 'continuous'},

            # Bass Synth (Channel 1)
            'bass_volume': {'cc': 7, 'range': (0, 127), 'channel': 1, 'type': 'continuous'},
            'bass_tune': {'cc': 74, 'range': (0, 127), 'channel': 1, 'type': 'continuous'},
            'bass_cutoff': {'cc': 71, 'range': (0, 127), 'channel': 1, 'type': 'continuous'},
            'bass_resonance': {'cc': 76, 'range': (0, 127), 'channel': 1, 'type': 'continuous'},
            'bass_env_amount': {'cc': 77, 'range': (0, 127), 'channel': 1, 'type': 'continuous'},
            'bass_decay': {'cc': 72, 'range': (0, 127), 'channel': 1, 'type': 'continuous'},
            'bass_accent': {'cc': 96, 'range': (0, 127), 'channel': 1, 'type': 'continuous'},

            # Global Controls
            'howl': {'cc': 94, 'range': (0, 127), 'channel': 1, 'type': 'continuous'},
            'main_volume': {'cc': 7, 'range': (0, 127), 'channel': 1, 'type': 'continuous'},
            'tempo': {'cc': 120, 'range': (60, 200), 'channel': 1, 'type': 'continuous'},
            
            # Pattern and Transport Controls
            'pattern_select': {'cc': 32, 'range': (0, 15), 'channel': 1, 'type': 'discrete'},
            'sync_mode': {'cc': 127, 'range': (0, 2), 'channel': 1, 'type': 'discrete'},  # 0=internal, 1=midi, 2=din
        }

    def handle_incoming_midi(self, message):
        """Process MIDI messages from Rhythm Wolf"""
        if message.type == 'note_on':
            self.note_triggered.emit(message.channel, message.note, message.velocity)

        elif message.type == 'control_change':
            # Handle BPM changes specially
            if message.control in [120, 74]:  # Tempo knob
                # Convert CC to BPM
                bpm = 60 + (message.value / 127.0) * 140
                self.bpm_changed.emit(bpm)
            else:
                # Regular parameter change
                param_name = self.cc_to_parameter_name(message.control, message.channel)
                if param_name:
                    self.parameter_changed.emit(param_name, message.value)

        elif message.type in ['start', 'stop', 'continue']:
            self.transport_changed.emit(message.type)
            self.is_playing = (message.type in ['start', 'continue'])
            
        elif message.type == 'program_change':
            if message.channel == 1:  # Pattern change
                self.current_pattern = message.program + 1
                self.pattern_changed.emit(self.current_pattern)

    def cc_to_parameter_name(self, cc_number, channel):
        """Convert CC number and channel to parameter name"""
        mappings = self.get_control_mappings()

        for param_name, mapping in mappings.items():
            if mapping['cc'] == cc_number and mapping['channel'] == channel:
                return param_name

        return None

    def send_parameter_change(self, parameter, value):
        """Send parameter changes to Rhythm Wolf"""
        if not self.is_connected or not self.output_port:
            return False

        mappings = self.get_control_mappings()

        if parameter in mappings:
            mapping = mappings[parameter]
            cc_msg = mido.Message('control_change', 
                                channel=mapping['channel'],
                                control=mapping['cc'],
                                value=int(value))
            try:
                self.output_port.send(cc_msg)
                return True
            except Exception as e:
                print(f"Failed to send CC: {e}")
                return False

        return False
        
    def _initialize_parameter_state(self):
        """Initialize parameter state tracking"""
        mappings = self.get_control_mappings()
        for param_name in mappings:
            self.hardware_parameters[param_name] = 64  # Default middle value
            self.software_parameters[param_name] = 64
    
    def enable_bidirectional_sync(self, enabled: bool = True):
        """Enable/disable bidirectional parameter synchronization"""
        self.sync_enabled = enabled
        if enabled:
            self.sync_timer.start(50)  # Sync every 50ms
        else:
            self.sync_timer.stop()
    
    def _sync_parameters(self):
        """Synchronize parameters between hardware and software"""
        if not self.sync_enabled or not self.is_connected:
            return
            
        current_time = time.time() * 1000  # milliseconds
        
        # Only sync if enough time has passed since last sync
        if current_time - self.last_sync_time < 25:  # 40Hz max sync rate
            return
            
        self.last_sync_time = current_time
        
        # Check for parameter differences and sync
        for param_name in self.software_parameters:
            hw_value = self.hardware_parameters.get(param_name, 64)
            sw_value = self.software_parameters.get(param_name, 64)
            
            # If software parameter changed, send to hardware
            if abs(hw_value - sw_value) > 1:  # 1 CC value tolerance
                if self.send_parameter_change(param_name, sw_value):
                    self.hardware_parameters[param_name] = sw_value
                    
    def update_software_parameter(self, parameter: str, value: int):
        """Update software parameter value (will be synced to hardware)"""
        if parameter in self.software_parameters:
            self.software_parameters[parameter] = value
            
    def get_parameter_value(self, parameter: str) -> Optional[int]:
        """Get current parameter value"""
        return self.hardware_parameters.get(parameter)
        
    def get_all_parameters(self) -> Dict[str, int]:
        """Get all current parameter values"""
        return self.hardware_parameters.copy()
        
    def request_parameter_dump(self) -> bool:
        """Request full parameter dump from hardware"""
        if not self.is_connected or not self.output_port:
            return False
            
        try:
            # Send SysEx request for parameter dump (if supported)
            sysex_request = mido.Message('sysex', 
                                       data=[0x47, 0x7F, 0x29, 0x60, 0x00, 0x04, 0x00, 0x00])
            self.output_port.send(sysex_request)
            return True
        except Exception as e:
            print(f"Failed to request parameter dump: {e}")
            return False
            
    def send_pattern_change(self, pattern_number: int) -> bool:
        """Send pattern change to hardware"""
        if not self.is_connected or not self.output_port:
            return False
            
        if not 1 <= pattern_number <= 16:
            return False
            
        try:
            prog_change = mido.Message('program_change',
                                     channel=1,
                                     program=pattern_number - 1)
            self.output_port.send(prog_change)
            self.current_pattern = pattern_number
            return True
        except Exception as e:
            print(f"Failed to send pattern change: {e}")
            return False
            
    def send_transport_command(self, command: str) -> bool:
        """Send transport command to hardware"""
        if not self.is_connected or not self.output_port:
            return False
            
        try:
            if command == 'play':
                msg = mido.Message('start')
            elif command == 'stop':
                msg = mido.Message('stop')
            elif command == 'continue':
                msg = mido.Message('continue')
            else:
                return False
                
            self.output_port.send(msg)
            return True
        except Exception as e:
            print(f"Failed to send transport command: {e}")
            return False
            
    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information"""
        return {
            'device_name': self.device_name,
            'midi_port': self.midi_port,
            'is_connected': self.is_connected,
            'capabilities': self.capabilities,
            'current_pattern': self.current_pattern,
            'sync_mode': self.sync_mode,
            'is_playing': self.is_playing,
            'sync_enabled': self.sync_enabled,
            'message_latency_ms': self.message_latency_ms,
            'sync_errors': self.sync_errors
        }
        
    def get_note_mappings(self) -> Dict[str, int]:
        """Get drum voice note mappings"""
        return {
            'kick': 36,          # C2
            'snare': 38,         # D2
            'percussion': 40,    # E2
            'open_hihat': 42,    # F#2
            'closed_hihat': 44   # G#2
        }
        
    def trigger_drum_voice(self, voice: str, velocity: int = 127) -> bool:
        """Trigger a specific drum voice"""
        note_mappings = self.get_note_mappings()
        if voice not in note_mappings:
            return False
            
        if not self.is_connected or not self.output_port:
            return False
            
        try:
            note_on = mido.Message('note_on',
                                 channel=10,
                                 note=note_mappings[voice],
                                 velocity=velocity)
            
            note_off = mido.Message('note_off',
                                  channel=10,
                                  note=note_mappings[voice],
                                  velocity=0)
            
            self.output_port.send(note_on)
            # Schedule note off after short duration
            QTimer.singleShot(50, lambda: self.output_port.send(note_off))
            return True
            
        except Exception as e:
            print(f"Failed to trigger drum voice: {e}")
            return False
            
    def close_connection(self):
        """Clean shutdown of MIDI connection"""
        self.sync_timer.stop()
        
        with self.port_mutex:
            if self.input_port:
                try:
                    self.input_port.close()
                except:
                    pass
                self.input_port = None
                
            if self.output_port:
                try:
                    self.output_port.close()
                except:
                    pass
                self.output_port = None
                
        self.is_connected = False
        print(f"🔌 Disconnected from Rhythm Wolf")

class MIDIHandlerThread(QThread):
    """Dedicated thread for MIDI processing"""

    midi_received = pyqtSignal(object)  # MIDI message object

    def __init__(self, rhythm_wolf_interface):
        super().__init__()
        self.rhythm_wolf = rhythm_wolf_interface
        self.running = False

    def run(self):
        """Main MIDI processing loop"""
        self.running = True

        try:
            if self.rhythm_wolf.input_port:
                while self.running:
                    for message in self.rhythm_wolf.input_port.iter_pending():
                        if message:
                            self.midi_received.emit(message)
                            self.rhythm_wolf.handle_incoming_midi(message)

        except Exception as e:
            print(f"MIDI thread error: {e}")

    def stop(self):
        """Stop MIDI processing"""
        self.running = False
        self.quit()
        self.wait()