# BSA-Engine (Base Hardware Abstraction Engine)

BSA-Engine is a modular, hardware-agnostic music production engine designed to provide seamless integration and unified control for a wide variety of MIDI hardware devices. With a core abstraction layer, it empowers developers and musicians to connect, prototype, and create across multiple instruments and controllers—scaling easily as new hardware emerges.

## Features

- Universal, extensible hardware abstraction layer
- Modular device integration: add support for new gear with simple modules
- Automatic hardware detection and configuration
- Flexible device mapping via config files (YAML/JSON)
- Unified API for software/hardware interaction
- Suitable for use with UIs, scripting, or embedding in larger DAW projects
- Robust testing and rapid prototyping environment

## Installation

```

git clone https://github.com/<yourname>/bsa-engine.git
cd bsa-engine
pip install -e .

```

## Usage

1. Plug in your supported MIDI device(s)
2. Run your application (or scripts) using BSA-Engine
3. Access and control attached devices via the engine’s unified interface

Example:

```

from bsa_engine import Engine

engine = Engine()
engine.detect_devices()
for device in engine.devices:
device.set_parameter('filter', 0.75)

```

See [docs/USAGE.md](docs/USAGE.md) for details.
