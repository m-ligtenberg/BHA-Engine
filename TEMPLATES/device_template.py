from bsa_engine.hardware.base_hardware import BaseHardware

class ExampleSynthDevice(BaseHardware):
    DEVICE_NAME = "Example Synth"

    def __init__(self):
        super().__init__()
        self.parameters = {
            "cutoff": 0.0,
            "resonance": 0.5,
            "volume": 1.0,
        }

    def detect_device(self):
        # Custom MIDI port/device detection logic.
        pass

    def connect(self):
        # Establish MIDI comms.
        pass

    def disconnect(self):
        # Clean up after finished.
        pass

    def handle_midi_message(self, msg):
        # React to incoming MIDI messages.
        pass

    def send_mapped_parameter(self, param_name, value):
        # Convert to appropriate MIDI CC and send.
        pass

# Register the device in bsa_engine.hardware.devices.__init__.py
