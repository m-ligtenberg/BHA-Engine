from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, 
                             QLabel, QFrame, QPushButton)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QPainter, QLinearGradient, QColor, QPen, QBrush, QFont
from gui.components.virtual_knobs import KnobGroup, RhythmWolfKnob

class HardwareLED(QWidget):
    """Individual LED indicator for hardware status"""
    
    def __init__(self, color="#00FF88", parent=None):
        super().__init__(parent)
        self.led_color = QColor(color)
        self.is_on = False
        self.is_blinking = False
        self.setFixedSize(12, 12)
        
        # Blinking animation
        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self.toggle_blink_state)
        
    def set_on(self, on=True):
        """Turn LED on/off"""
        self.is_on = on
        self.is_blinking = False
        self.blink_timer.stop()
        self.update()
        
    def set_blinking(self, blink=True):
        """Make LED blink"""
        self.is_blinking = blink
        if blink:
            self.blink_timer.start(500)  # 500ms blink rate
        else:
            self.blink_timer.stop()
            
    def toggle_blink_state(self):
        """Toggle blink state"""
        self.is_on = not self.is_on
        self.update()
        
    def paintEvent(self, event):
        """Paint LED with glow effect"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        center = self.rect().center()
        radius = 5
        
        if self.is_on:
            # Glowing LED effect
            glow_color = self.led_color
            painter.setBrush(QBrush(glow_color))
            painter.setPen(QPen(glow_color.lighter(150), 2))
        else:
            # Dim LED
            dim_color = QColor(self.led_color.red() // 4, 
                              self.led_color.green() // 4, 
                              self.led_color.blue() // 4)
            painter.setBrush(QBrush(dim_color))
            painter.setPen(QPen(dim_color, 1))
            
        painter.drawEllipse(center, radius, radius)

class DrumPadButton(QPushButton):
    """Virtual drum pad button matching Rhythm Wolf hardware"""
    
    pad_pressed = pyqtSignal(str)  # Emit drum voice name when pressed
    
    def __init__(self, voice_name, parent=None):
        super().__init__(voice_name, parent)
        self.voice_name = voice_name
        self.is_playing = False
        self.setFixedSize(80, 60)
        
        # Connect press to signal
        self.pressed.connect(lambda: self.pad_pressed.emit(self.voice_name))
        
        self.setStyleSheet("""
            DrumPadButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #4A5F7A, stop:1 #34495E);
                border: 3px solid #3D4F66;
                border-radius: 8px;
                font-weight: bold;
                font-size: 12px;
                color: #E8F4FD;
            }
            DrumPadButton:hover {
                border-color: #00D4FF;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #2E86AB, stop:1 #34495E);
            }
            DrumPadButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #00D4FF, stop:1 #2E86AB);
                border-color: #00FF88;
            }
        """)
        
    def set_playing(self, playing):
        """Visual feedback when pad is being played"""
        self.is_playing = playing
        if playing:
            self.setStyleSheet(self.styleSheet() + """
                DrumPadButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                              stop:0 #00FF88, stop:1 #00D4FF);
                    border-color: #FFD700;
                }
            """)
        else:
            # Reset to normal styling
            self.__init__(self.voice_name, self.parent())

class VoiceSection(QFrame):
    """Complete voice section with knobs and controls for one Rhythm Wolf voice"""
    
    knob_changed = pyqtSignal(str, str, int)  # voice, parameter, value
    
    def __init__(self, voice_config, parent=None):
        super().__init__(parent)
        self.voice_name = voice_config['name']
        self.voice_config = voice_config
        self.knobs = {}
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup voice section layout"""
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("""
            VoiceSection {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #1A2332, stop:1 #243447);
                border: 2px solid #3D4F66;
                border-radius: 10px;
                margin: 4px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        
        # Voice title and LED
        title_layout = QHBoxLayout()
        
        # Voice name
        voice_label = QLabel(self.voice_name.upper())
        voice_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        voice_label.setStyleSheet("""
            color: #00D4FF; 
            font-weight: bold; 
            font-size: 14px;
            text-shadow: 0 0 5px #00D4FF;
        """)
        title_layout.addWidget(voice_label)
        
        # Status LED
        self.status_led = HardwareLED("#00FF88")
        title_layout.addWidget(self.status_led)
        
        layout.addLayout(title_layout)
        
        # Drum pad (for drum voices)
        if self.voice_name != "BASS":
            self.drum_pad = DrumPadButton(self.voice_name)
            self.drum_pad.pad_pressed.connect(self.handle_pad_press)
            layout.addWidget(self.drum_pad)
        
        # Knobs grid
        knobs_layout = QGridLayout()
        knobs_layout.setSpacing(6)
        
        row, col = 0, 0
        for param in self.voice_config.get('parameters', []):
            knob = RhythmWolfKnob(
                label=param['name'], 
                cc_number=param.get('cc')
            )
            knob.value_changed.connect(
                lambda name, value, voice=self.voice_name: 
                self.knob_changed.emit(voice, name, value)
            )
            
            self.knobs[param['name']] = knob
            knobs_layout.addWidget(knob, row, col)
            
            col += 1
            if col >= 2:  # 2 knobs per row
                col = 0
                row += 1
                
        layout.addLayout(knobs_layout)
        
    def handle_pad_press(self, voice_name):
        """Handle drum pad press"""
        self.status_led.set_on(True)
        # LED will be turned off by parent when note ends
        
    def set_knob_value(self, param_name, value):
        """Set knob value from hardware sync"""
        if param_name in self.knobs:
            self.knobs[param_name].sync_to_hardware_value(value)
            
    def get_knob_values(self):
        """Get current knob values"""
        return {name: knob.value() for name, knob in self.knobs.items()}

class GlobalControlsSection(QFrame):
    """Global controls: Howl, Main Volume, Tempo"""
    
    control_changed = pyqtSignal(str, int)  # control_name, value
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.global_knobs = {}
        self.setup_ui()
        
    def setup_ui(self):
        """Setup global controls"""
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("""
            GlobalControlsSection {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #2F3F52, stop:1 #1A2332);
                border: 2px solid #00D4FF;
                border-radius: 10px;
                margin: 4px;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("GLOBAL CONTROLS")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            color: #00D4FF; 
            font-weight: bold; 
            font-size: 16px;
            text-shadow: 0 0 10px #00D4FF;
        """)
        layout.addWidget(title_label)
        
        # Control knobs
        controls_layout = QHBoxLayout()
        
        global_controls = [
            {'name': 'HOWL', 'cc': 94, 'color': '#F18F01'},
            {'name': 'MAIN', 'cc': 7, 'color': '#00FF88'},
            {'name': 'TEMPO', 'cc': 120, 'color': '#FFD700'}
        ]
        
        for control in global_controls:
            knob = RhythmWolfKnob(
                label=control['name'],
                cc_number=control['cc']
            )
            knob.value_changed.connect(
                lambda name, value: self.control_changed.emit(name, value)
            )
            
            self.global_knobs[control['name']] = knob
            controls_layout.addWidget(knob)
            
        layout.addLayout(controls_layout)

class RhythmWolfWidget(QWidget):
    """Complete hardware visualization widget for Rhythm Wolf"""

    # Signals for hardware integration
    knob_changed = pyqtSignal(str, str, int)     # voice, parameter, value
    pad_pressed = pyqtSignal(str)                # voice_name
    global_control_changed = pyqtSignal(str, int) # control_name, value

    def __init__(self, parent=None):
        super().__init__(parent)
        self.voice_sections = {}
        self.global_controls = None
        self.setup_ui()

    def setup_ui(self):
        """Setup complete hardware interface"""
        self.setFixedHeight(280)
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(8)
        
        # Main title with hardware status
        title_layout = QHBoxLayout()
        
        title_label = QLabel("🎛️ RHYTHM WOLF HARDWARE PANEL")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setProperty("class", "TitleLabel")
        title_layout.addWidget(title_label)
        
        # Hardware connection status
        self.connection_led = HardwareLED("#00FF88")
        self.connection_status = QLabel("CONNECTED")
        self.connection_status.setStyleSheet("color: #00FF88; font-weight: bold;")
        title_layout.addWidget(self.connection_status)
        title_layout.addWidget(self.connection_led)
        
        main_layout.addLayout(title_layout)
        
        # Voice sections layout
        voices_layout = QHBoxLayout()
        voices_layout.setSpacing(4)
        
        # Define Rhythm Wolf voice configurations
        voice_configs = [
            {
                'name': 'KICK',
                'parameters': [
                    {'name': 'VOL', 'cc': 36},
                    {'name': 'TUNE', 'cc': 37},
                    {'name': 'ATT', 'cc': 38},
                    {'name': 'DECAY', 'cc': 39}
                ]
            },
            {
                'name': 'SNARE', 
                'parameters': [
                    {'name': 'VOL', 'cc': 40},
                    {'name': 'TUNE', 'cc': 41},
                    {'name': 'NOISE', 'cc': 42},
                    {'name': 'DECAY', 'cc': 43}
                ]
            },
            {
                'name': 'PERC',
                'parameters': [
                    {'name': 'VOL', 'cc': 44},
                    {'name': 'HIGH', 'cc': 45},
                    {'name': 'NOISE', 'cc': 46},
                    {'name': 'LOW', 'cc': 47}
                ]
            },
            {
                'name': 'OH',  # Open Hi-Hat
                'parameters': [
                    {'name': 'VOL', 'cc': 48},
                    {'name': 'TUNE', 'cc': 49},
                    {'name': 'DECAY', 'cc': 50}
                ]
            },
            {
                'name': 'CH',  # Closed Hi-Hat
                'parameters': [
                    {'name': 'VOL', 'cc': 51},
                    {'name': 'TUNE', 'cc': 52}
                ]
            },
            {
                'name': 'BASS',
                'parameters': [
                    {'name': 'VOL', 'cc': 53},
                    {'name': 'FILTER', 'cc': 54},
                    {'name': 'DECAY', 'cc': 55},
                    {'name': 'TUNE', 'cc': 56}
                ]
            }
        ]
        
        # Create voice sections
        for voice_config in voice_configs:
            voice_section = VoiceSection(voice_config)
            voice_section.knob_changed.connect(self.knob_changed)
            voice_section.pad_pressed.connect(self.pad_pressed)
            
            self.voice_sections[voice_config['name']] = voice_section
            voices_layout.addWidget(voice_section)
            
        main_layout.addLayout(voices_layout)
        
        # Global controls
        self.global_controls = GlobalControlsSection()
        self.global_controls.control_changed.connect(self.global_control_changed)
        main_layout.addWidget(self.global_controls)

    def sync_with_hardware(self):
        """Update all knobs to match hardware positions"""
        # This will be called by the main application's update timer
        # In a real implementation, this would query the MIDI handler
        # for current hardware knob positions and update the GUI
        
        # Placeholder: simulate some knob movements for demo
        import random
        if hasattr(self, '_demo_counter'):
            self._demo_counter += 1
        else:
            self._demo_counter = 0
            
        # Every 100 updates, simulate a knob change
        if self._demo_counter % 100 == 0:
            voice = random.choice(list(self.voice_sections.keys()))
            voice_section = self.voice_sections[voice]
            if voice_section.knobs:
                knob_name = random.choice(list(voice_section.knobs.keys()))
                new_value = random.randint(0, 127)
                voice_section.set_knob_value(knob_name, new_value)
                
    def set_hardware_connected(self, connected=True):
        """Update hardware connection status"""
        self.connection_led.set_on(connected)
        if connected:
            self.connection_status.setText("CONNECTED")
            self.connection_status.setStyleSheet("color: #00FF88; font-weight: bold;")
        else:
            self.connection_status.setText("DISCONNECTED")  
            self.connection_status.setStyleSheet("color: #FF4757; font-weight: bold;")
            
    def trigger_pad_visual(self, voice_name, duration_ms=200):
        """Trigger visual feedback on drum pad"""
        if voice_name in self.voice_sections:
            voice_section = self.voice_sections[voice_name]
            if hasattr(voice_section, 'drum_pad'):
                voice_section.drum_pad.set_playing(True)
                # Timer to turn off visual feedback
                QTimer.singleShot(duration_ms, 
                                lambda: voice_section.drum_pad.set_playing(False))
                                
    def get_all_knob_values(self):
        """Get current values of all knobs for project save"""
        all_values = {}
        
        # Voice knobs
        for voice_name, voice_section in self.voice_sections.items():
            all_values[voice_name] = voice_section.get_knob_values()
            
        # Global controls  
        if self.global_controls:
            all_values['GLOBAL'] = {
                name: knob.value() 
                for name, knob in self.global_controls.global_knobs.items()
            }
            
        return all_values