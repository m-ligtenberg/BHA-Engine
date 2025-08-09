# Contributing to BSA-Engine

Welcome! Expanding hardware support and improving the engine depends on developers like you. Here’s how to contribute:

## Getting Started

1. Fork this repository and branch off `main`.
2. Add new device support by:
   - Creating a subclass of `BaseHardware` in `src/hardware/devices/`
   - Adding or editing device mapping files in `configs/devices/`
   - Writing or updating relevant tests in `tests/`
3. Ensure code follows [PEP8](https://www.python.org/dev/peps/pep-0008/) and project linting rules.
4. Add or update documentation as needed.

## Development Workflow

- Branch naming:  
  `feature/device-<name>`, `fix/<what>`, `doc/<topic>`
- Describe your pull request and its testing.  
- For nontrivial changes, open a GitHub Issue first to discuss!
- All PRs must pass CI checks.

## Testing

- Run unit tests: `pytest`
- Hardware integration? If possible, include mock/test data or scripts to simulate hardware comms.
- Document edge cases in your tests.

## Community

- Respectful, welcoming, and helpful discussions only.
- Use [GitHub Discussions](https://github.com/<yourname>/bsa-engine/discussions) or Discord (see docs).

---

Thanks for helping build the future of modular music-making!
