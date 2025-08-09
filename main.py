#!/usr/bin/env python3
"""
Rhythm Wolf Mini-DAW
Main application entry point
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from gui.main_application import main

if __name__ == "__main__":
    sys.exit(main())