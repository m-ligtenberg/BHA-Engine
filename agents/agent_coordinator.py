"""
Agent Coordinator - Manages the specialized development agents
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class AgentCoordinator:
    """Coordinates all specialized development agents"""
    
    def __init__(self):
        self.agents = {
            'hardware': None,     # Agent 1: Hardware Integration Specialist
            'audio': None,        # Agent 2: Audio Engine Developer  
            'ui': None,           # Agent 3: PyQt6 UI/UX Designer
            'samples': None,      # Agent 4: Sample & Project Manager
            'qa': None            # Agent 5: Quality Assurance Monitor
        }
        
        self.development_phases = {
            'phase_1': 'Core Infrastructure',
            'phase_2': 'Hardware Features', 
            'phase_3': 'Audio & Effects',
            'phase_4': 'Integration & Polish'
        }
        
        self.current_phase = 'phase_1'
        
    def get_current_priorities(self):
        """Get current development priorities by phase"""
        priorities = {
            'phase_1': [
                'DAW Engine orchestration',
                'Rhythm Wolf MIDI interface', 
                'Core MIDI I/O handling',
                'PyQt6 main window setup'
            ],
            'phase_2': [
                'Auto-recording system',
                'Real-time BPM synchronization', 
                'Hardware-synced GUI knobs',
                '16-step LED sequencer visualization'
            ],
            'phase_3': [
                'Multi-track recording engine',
                'Effects processing (EQ, compression, reverb, delay)',
                'Sample library integration',
                'Professional UI with blueprint theme'
            ],
            'phase_4': [
                'Performance optimization',
                'Cross-platform testing',
                'Documentation completion',
                'Quality assurance validation'
            ]
        }
        
        return priorities.get(self.current_phase, [])
        
    def advance_phase(self):
        """Advance to next development phase"""
        phases = list(self.development_phases.keys())
        current_index = phases.index(self.current_phase)
        
        if current_index < len(phases) - 1:
            self.current_phase = phases[current_index + 1]
            print(f"🚀 Advanced to {self.development_phases[self.current_phase]}")
        else:
            print("✅ All development phases completed!")
            
    def get_agent_assignments(self):
        """Get current agent work assignments"""
        if self.current_phase == 'phase_1':
            return {
                'hardware': 'Implement Rhythm Wolf MIDI interface and auto-record system',
                'audio': 'Create basic DAW engine and multi-track foundation',
                'ui': 'Enhance PyQt6 interface with blueprint theme and custom widgets',
                'samples': 'Design project management and pattern sequencer architecture',
                'qa': 'Set up testing framework and code quality monitoring'
            }
        elif self.current_phase == 'phase_2':
            return {
                'hardware': 'Complete auto-record and BPM sync features',
                'ui': 'Implement hardware-synced knobs and LED animations',
                'audio': 'Integrate MIDI with audio engine',
                'samples': 'Create pattern storage and retrieval system',
                'qa': 'Test hardware integration and real-time performance'
            }
        # Add more phases as needed...
        
        return {}

# Global coordinator instance
coordinator = AgentCoordinator()

def get_coordinator():
    return coordinator