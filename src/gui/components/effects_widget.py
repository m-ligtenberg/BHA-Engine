from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
                             QLabel, QFrame, QPushButton, QComboBox, QSlider,
                             QGraphicsDropShadowEffect, QScrollArea, QGroupBox)
from PyQt6.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve, QTimer
from PyQt6.QtGui import QPainter, QLinearGradient, QColor, QPen, QBrush, QFont
from gui.components.virtual_knobs import RhythmWolfKnob

class EffectSlot(QFrame):
    """Individual effect slot in the effects chain"""
    
    effect_changed = pyqtSignal(int, str)      # slot_number, effect_name
    parameter_changed = pyqtSignal(int, str, float)  # slot_number, param_name, value
    bypass_changed = pyqtSignal(int, bool)     # slot_number, bypassed
    
    def __init__(self, slot_number, parent=None):
        super().__init__(parent)
        self.slot_number = slot_number
        self.current_effect = None
        self.is_bypassed = False
        self.parameter_knobs = {}
        self.effect_parameters = {}
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup effect slot interface"""
        self.setFrameStyle(QFrame.Shape.Box)
        self.setFixedSize(200, 280)
        self.setStyleSheet("""
            EffectSlot {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #243447, stop:1 #1A2332);
                border: 2px solid #3D4F66;
                border-radius: 8px;
                margin: 2px;
            }
        """)
        
        # Add drop shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 60))
        shadow.setOffset(2, 2)
        self.setGraphicsEffect(shadow)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        
        # Slot header
        header_layout = QHBoxLayout()
        
        # Slot number
        slot_label = QLabel(f"SLOT {self.slot_number}")
        slot_label.setStyleSheet("color: #00D4FF; font-weight: bold; font-size: 12px;")
        header_layout.addWidget(slot_label)
        
        # Bypass button
        self.bypass_button = QPushButton("BYP")
        self.bypass_button.setCheckable(True)
        self.bypass_button.setFixedSize(40, 25)
        self.bypass_button.clicked.connect(self.toggle_bypass)
        self.bypass_button.setStyleSheet("""
            QPushButton {
                background-color: #2C3E50;
                border: 1px solid #3D4F66;
                border-radius: 4px;
                font-size: 10px;
                font-weight: bold;
                color: #E8F4FD;
            }
            QPushButton:checked {
                background-color: #F18F01;
                color: #000000;
            }
        """)
        header_layout.addWidget(self.bypass_button)
        
        layout.addLayout(header_layout)
        
        # Effect selector
        self.effect_combo = QComboBox()
        self.effect_combo.addItems([
            "None", "Compressor", "EQ 3-Band", "Reverb Hall", 
            "Reverb Room", "Reverb Plate", "Delay 1/4", "Delay 1/8", 
            "Saturation", "Filter LP", "Filter HP", "Chorus", "Flanger"
        ])
        self.effect_combo.currentTextChanged.connect(self.change_effect)
        self.effect_combo.setStyleSheet("""
            QComboBox {
                background-color: #34495E;
                border: 2px solid #3D4F66;
                border-radius: 4px;
                padding: 4px;
                font-size: 11px;
                color: #E8F4FD;
            }
            QComboBox:hover {
                border-color: #00D4FF;
            }
            QComboBox::drop-down {
                border: none;
                background: #2E86AB;
            }
            QComboBox::down-arrow {
                border: 2px solid #E8F4FD;
                border-radius: 2px;
            }
        """)
        layout.addWidget(self.effect_combo)
        
        # Effect display name
        self.effect_name_label = QLabel("EMPTY SLOT")
        self.effect_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.effect_name_label.setStyleSheet("""
            color: #A4B8D1; 
            font-weight: bold; 
            font-size: 14px;
            background-color: #1A2332;
            border-radius: 4px;
            padding: 6px;
        """)
        layout.addWidget(self.effect_name_label)
        
        # Parameters container
        self.params_widget = QWidget()
        self.params_layout = QGridLayout(self.params_widget)
        self.params_layout.setSpacing(4)
        layout.addWidget(self.params_widget)
        
        # Wet/Dry mix (always present)
        self.mix_knob = RhythmWolfKnob("MIX")
        self.mix_knob.setValue(50)  # 50% mix by default
        self.mix_knob.value_changed.connect(
            lambda name, value: self.parameter_changed.emit(self.slot_number, "mix", value / 127.0)
        )
        layout.addWidget(self.mix_knob)
        
    def toggle_bypass(self):
        """Toggle effect bypass"""
        self.is_bypassed = self.bypass_button.isChecked()
        self.bypass_changed.emit(self.slot_number, self.is_bypassed)
        
        # Visual feedback
        if self.is_bypassed:
            self.setStyleSheet(self.styleSheet().replace("#3D4F66", "#6B7B94"))
            self.effect_name_label.setText(f"{self.current_effect or 'EMPTY SLOT'} (BYPASSED)")
        else:
            self.setStyleSheet(self.styleSheet().replace("#6B7B94", "#3D4F66"))
            self.effect_name_label.setText(self.current_effect or "EMPTY SLOT")
            
    def change_effect(self, effect_name):
        """Change the current effect"""
        if effect_name == "None":
            self.current_effect = None
            self.effect_name_label.setText("EMPTY SLOT")
            self.clear_parameters()
        else:
            self.current_effect = effect_name
            self.effect_name_label.setText(effect_name.upper())
            self.setup_effect_parameters(effect_name)
            
        self.effect_changed.emit(self.slot_number, effect_name)
        
    def clear_parameters(self):
        """Clear all parameter knobs"""
        for knob in list(self.parameter_knobs.values()):
            knob.setParent(None)
            knob.deleteLater()
        self.parameter_knobs.clear()
        
    def setup_effect_parameters(self, effect_name):
        """Setup parameter knobs for the selected effect"""
        self.clear_parameters()
        
        # Define parameters for each effect
        effect_params = {
            "Compressor": [
                {"name": "THRESH", "default": 80, "tooltip": "Compression threshold"},
                {"name": "RATIO", "default": 64, "tooltip": "Compression ratio"},
                {"name": "ATTACK", "default": 20, "tooltip": "Attack time"},
                {"name": "RELEASE", "default": 40, "tooltip": "Release time"}
            ],
            "EQ 3-Band": [
                {"name": "HIGH", "default": 64, "tooltip": "High frequency gain"},
                {"name": "MID", "default": 64, "tooltip": "Mid frequency gain"},
                {"name": "LOW", "default": 64, "tooltip": "Low frequency gain"},
                {"name": "Q", "default": 50, "tooltip": "Mid frequency Q factor"}
            ],
            "Reverb Hall": [
                {"name": "SIZE", "default": 70, "tooltip": "Room size"},
                {"name": "DECAY", "default": 60, "tooltip": "Decay time"},
                {"name": "DAMP", "default": 40, "tooltip": "High frequency damping"},
                {"name": "PREDLY", "default": 30, "tooltip": "Pre-delay time"}
            ],
            "Reverb Room": [
                {"name": "SIZE", "default": 40, "tooltip": "Room size"},
                {"name": "DECAY", "default": 35, "tooltip": "Decay time"},
                {"name": "DAMP", "default": 50, "tooltip": "High frequency damping"},
                {"name": "EARLY", "default": 45, "tooltip": "Early reflections"}
            ],
            "Reverb Plate": [
                {"name": "SIZE", "default": 60, "tooltip": "Plate size"},
                {"name": "DECAY", "default": 55, "tooltip": "Decay time"},
                {"name": "BRIGHT", "default": 70, "tooltip": "High frequency content"},
                {"name": "PREDLY", "default": 20, "tooltip": "Pre-delay time"}
            ],
            "Delay 1/4": [
                {"name": "TIME", "default": 85, "tooltip": "Delay time (1/4 note)"},
                {"name": "FDBK", "default": 45, "tooltip": "Feedback amount"},
                {"name": "FILTER", "default": 64, "tooltip": "Filter cutoff"},
                {"name": "SYNC", "default": 127, "tooltip": "Tempo sync on/off"}
            ],
            "Delay 1/8": [
                {"name": "TIME", "default": 64, "tooltip": "Delay time (1/8 note)"},
                {"name": "FDBK", "default": 35, "tooltip": "Feedback amount"},
                {"name": "FILTER", "default": 80, "tooltip": "Filter cutoff"},
                {"name": "PING", "default": 30, "tooltip": "Ping-pong amount"}
            ],
            "Saturation": [
                {"name": "DRIVE", "default": 50, "tooltip": "Saturation drive"},
                {"name": "TONE", "default": 64, "tooltip": "Tonal character"},
                {"name": "OUTPUT", "default": 90, "tooltip": "Output level"},
                {"name": "TYPE", "default": 40, "tooltip": "Saturation type"}
            ],
            "Filter LP": [
                {"name": "CUTOFF", "default": 80, "tooltip": "Cutoff frequency"},
                {"name": "RESO", "default": 20, "tooltip": "Resonance"},
                {"name": "ENV", "default": 0, "tooltip": "Envelope amount"},
                {"name": "LFO", "default": 0, "tooltip": "LFO amount"}
            ],
            "Filter HP": [
                {"name": "CUTOFF", "default": 30, "tooltip": "Cutoff frequency"},
                {"name": "RESO", "default": 20, "tooltip": "Resonance"},
                {"name": "ENV", "default": 0, "tooltip": "Envelope amount"},
                {"name": "SLOPE", "default": 64, "tooltip": "Filter slope"}
            ],
            "Chorus": [
                {"name": "RATE", "default": 40, "tooltip": "LFO rate"},
                {"name": "DEPTH", "default": 35, "tooltip": "Modulation depth"},
                {"name": "DELAY", "default": 25, "tooltip": "Delay time"},
                {"name": "FDBK", "default": 15, "tooltip": "Feedback amount"}
            ],
            "Flanger": [
                {"name": "RATE", "default": 30, "tooltip": "LFO rate"},
                {"name": "DEPTH", "default": 60, "tooltip": "Modulation depth"},
                {"name": "FDBK", "default": 40, "tooltip": "Feedback amount"},
                {"name": "MANUAL", "default": 64, "tooltip": "Manual delay offset"}
            ]
        }
        
        if effect_name in effect_params:
            params = effect_params[effect_name]
            row, col = 0, 0
            
            for param in params:
                knob = RhythmWolfKnob(param["name"])
                knob.setValue(param["default"])
                knob.setToolTip(param["tooltip"])
                knob.value_changed.connect(
                    lambda name, value, param_name=param["name"].lower(): 
                    self.parameter_changed.emit(self.slot_number, param_name, value / 127.0)
                )
                
                self.parameter_knobs[param["name"]] = knob
                self.params_layout.addWidget(knob, row, col)
                
                col += 1
                if col >= 2:  # 2 knobs per row
                    col = 0
                    row += 1

class EffectsRackWidget(QWidget):
    """Professional effects rack with 6 effect slots per track"""
    
    effects_changed = pyqtSignal(str, list)  # track_name, effects_chain
    
    def __init__(self, track_name="Track", parent=None):
        super().__init__(parent)
        self.track_name = track_name
        self.effect_slots = []
        self.setup_ui()
        
    def setup_ui(self):
        """Setup effects rack interface"""
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        
        # Rack header
        header_layout = QHBoxLayout()
        
        # Track name
        track_label = QLabel(f"{self.track_name.upper()} EFFECTS RACK")
        track_label.setStyleSheet("""
            color: #00D4FF; 
            font-weight: bold; 
            font-size: 16px;
            text-shadow: 0 0 8px #00D4FF;
        """)
        header_layout.addWidget(track_label)
        
        # Master bypass
        self.master_bypass = QPushButton("MASTER BYPASS")
        self.master_bypass.setCheckable(True)
        self.master_bypass.setFixedSize(120, 30)
        self.master_bypass.clicked.connect(self.toggle_master_bypass)
        self.master_bypass.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #34495E, stop:1 #2C3E50);
                border: 2px solid #3D4F66;
                border-radius: 6px;
                font-weight: bold;
                color: #E8F4FD;
            }
            QPushButton:hover {
                border-color: #00D4FF;
            }
            QPushButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #F18F01, stop:1 #E67E22);
                color: #000000;
            }
        """)
        header_layout.addWidget(self.master_bypass)
        
        layout.addLayout(header_layout)
        
        # Effects chain display
        chain_display = QLabel("Signal Flow: INPUT → SLOT 1 → SLOT 2 → SLOT 3 → SLOT 4 → SLOT 5 → SLOT 6 → OUTPUT")
        chain_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chain_display.setStyleSheet("color: #A4B8D1; font-size: 11px; font-family: 'Courier New';")
        layout.addWidget(chain_display)
        
        # Scrollable effects slots
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setFixedHeight(320)
        
        slots_container = QWidget()
        slots_layout = QHBoxLayout(slots_container)
        slots_layout.setSpacing(4)
        
        # Create 6 effect slots
        for i in range(1, 7):
            slot = EffectSlot(i)
            slot.effect_changed.connect(self.handle_effect_changed)
            slot.parameter_changed.connect(self.handle_parameter_changed)
            slot.bypass_changed.connect(self.handle_bypass_changed)
            
            self.effect_slots.append(slot)
            slots_layout.addWidget(slot)
            
        scroll_area.setWidget(slots_container)
        layout.addWidget(scroll_area)
        
        # CPU usage meter
        cpu_layout = QHBoxLayout()
        cpu_label = QLabel("CPU Usage:")
        cpu_label.setStyleSheet("color: #A4B8D1; font-size: 11px;")
        cpu_layout.addWidget(cpu_label)
        
        self.cpu_meter = QSlider(Qt.Orientation.Horizontal)
        self.cpu_meter.setRange(0, 100)
        self.cpu_meter.setValue(15)  # Simulated CPU usage
        self.cpu_meter.setEnabled(False)  # Read-only display
        self.cpu_meter.setFixedWidth(200)
        cpu_layout.addWidget(self.cpu_meter)
        
        self.cpu_value_label = QLabel("15%")
        self.cpu_value_label.setStyleSheet("color: #00FF88; font-size: 11px; font-family: 'Courier New';")
        cpu_layout.addWidget(self.cpu_value_label)
        
        cpu_layout.addStretch()
        layout.addLayout(cpu_layout)
        
        # Start CPU monitoring timer
        self.cpu_timer = QTimer()
        self.cpu_timer.timeout.connect(self.update_cpu_usage)
        self.cpu_timer.start(1000)  # Update every second
        
    def toggle_master_bypass(self):
        """Toggle master bypass for all effects"""
        is_bypassed = self.master_bypass.isChecked()
        
        # Visual feedback - dim all slots if bypassed
        for slot in self.effect_slots:
            if is_bypassed:
                slot.setEnabled(False)
                slot.setStyleSheet(slot.styleSheet() + """
                    EffectSlot { background-color: #1A1A1A; }
                """)
            else:
                slot.setEnabled(True)
                # Reset stylesheet
                slot.setup_ui()
                
    def handle_effect_changed(self, slot_number, effect_name):
        """Handle effect change in a slot"""
        self.update_effects_chain()
        
    def handle_parameter_changed(self, slot_number, param_name, value):
        """Handle parameter change"""
        self.update_effects_chain()
        
    def handle_bypass_changed(self, slot_number, bypassed):
        """Handle bypass change"""
        self.update_effects_chain()
        
    def update_effects_chain(self):
        """Update the complete effects chain and emit signal"""
        effects_chain = []
        
        for slot in self.effect_slots:
            if slot.current_effect and slot.current_effect != "None" and not slot.is_bypassed:
                effect_data = {
                    'type': slot.current_effect,
                    'parameters': {name: knob.value() / 127.0 for name, knob in slot.parameter_knobs.items()},
                    'mix': slot.mix_knob.value() / 127.0
                }
                effects_chain.append(effect_data)
                
        self.effects_changed.emit(self.track_name, effects_chain)
        
    def update_cpu_usage(self):
        """Update CPU usage display (simulated)"""
        # Calculate CPU based on active effects
        active_effects = sum(1 for slot in self.effect_slots 
                           if slot.current_effect and slot.current_effect != "None" and not slot.is_bypassed)
        
        # Base CPU + effects overhead
        cpu_usage = 8 + (active_effects * 12)  # Rough simulation
        
        # Add some random variation
        import random
        cpu_usage += random.randint(-3, 3)
        cpu_usage = max(5, min(100, cpu_usage))
        
        self.cpu_meter.setValue(cpu_usage)
        self.cpu_value_label.setText(f"{cpu_usage}%")
        
        # Color coding for CPU usage
        if cpu_usage < 50:
            color = "#00FF88"  # Green
        elif cpu_usage < 80:
            color = "#FFD700"  # Yellow
        else:
            color = "#FF4757"  # Red
            
        self.cpu_value_label.setStyleSheet(f"color: {color}; font-size: 11px; font-family: 'Courier New';")
        
    def get_effects_preset(self):
        """Get current effects configuration for saving"""
        preset = {}
        for i, slot in enumerate(self.effect_slots):
            preset[f"slot_{i+1}"] = {
                'effect': slot.current_effect,
                'bypassed': slot.is_bypassed,
                'parameters': {name: knob.value() for name, knob in slot.parameter_knobs.items()},
                'mix': slot.mix_knob.value()
            }
        preset['master_bypassed'] = self.master_bypass.isChecked()
        return preset
        
    def load_effects_preset(self, preset):
        """Load effects configuration from preset"""
        for i, slot in enumerate(self.effect_slots):
            slot_key = f"slot_{i+1}"
            if slot_key in preset:
                slot_data = preset[slot_key]
                
                # Set effect type
                if slot_data.get('effect'):
                    effect_index = slot.effect_combo.findText(slot_data['effect'])
                    if effect_index >= 0:
                        slot.effect_combo.setCurrentIndex(effect_index)
                        
                # Set bypass state
                slot.bypass_button.setChecked(slot_data.get('bypassed', False))
                slot.toggle_bypass()
                
                # Set parameters
                parameters = slot_data.get('parameters', {})
                for param_name, value in parameters.items():
                    if param_name.upper() in slot.parameter_knobs:
                        slot.parameter_knobs[param_name.upper()].setValue(value)
                        
                # Set mix
                slot.mix_knob.setValue(slot_data.get('mix', 50))
                
        # Set master bypass
        self.master_bypass.setChecked(preset.get('master_bypassed', False))
        if preset.get('master_bypassed', False):
            self.toggle_master_bypass()