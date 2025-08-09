"""
Professional Effects Chain System
Supports 6 effect slots per track with advanced routing and processing
"""

import numpy as np
from typing import Dict, List, Optional, Any, Type, Union
from threading import RLock
from enum import Enum
from abc import ABC, abstractmethod
from PyQt6.QtCore import QObject, pyqtSignal
import time


class EffectType(Enum):
    """Available effect types"""
    COMPRESSOR = "compressor"
    EQ = "eq" 
    REVERB = "reverb"
    DELAY = "delay"
    SATURATION = "saturation"
    FILTER = "filter"
    CHORUS = "chorus"
    FLANGER = "flanger"
    PHASER = "phaser"
    GATE = "gate"
    LIMITER = "limiter"
    DISTORTION = "distortion"


class BaseEffect(ABC):
    """
    Abstract base class for all audio effects.
    Provides common interface and functionality.
    """
    
    def __init__(self, effect_id: str, effect_type: EffectType, 
                 sample_rate: int = 44100, buffer_size: int = 512):
        self.effect_id = effect_id
        self.effect_type = effect_type
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        
        # Effect state
        self.enabled = True
        self.bypassed = False
        self.wet_dry_mix = 1.0  # 0.0 = dry, 1.0 = wet
        
        # Parameters (override in subclasses)
        self.parameters = {}
        self.parameter_ranges = {}
        self.parameter_defaults = {}
        
        # Performance monitoring
        self.processing_time_ms = 0.0
        self.cpu_usage_percent = 0.0
        
        # Initialize effect-specific parameters
        self._initialize_parameters()
    
    @abstractmethod
    def _initialize_parameters(self):
        """Initialize effect-specific parameters"""
        pass
    
    @abstractmethod
    def process(self, input_data: np.ndarray, frames: int) -> np.ndarray:
        """Process audio through this effect"""
        pass
    
    def set_parameter(self, param_name: str, value: float) -> bool:
        """Set effect parameter with validation"""
        if param_name not in self.parameters:
            return False
        
        # Validate parameter range
        if param_name in self.parameter_ranges:
            min_val, max_val = self.parameter_ranges[param_name]
            value = max(min_val, min(max_val, value))
        
        self.parameters[param_name] = value
        self._on_parameter_changed(param_name, value)
        return True
    
    def get_parameter(self, param_name: str) -> Optional[float]:
        """Get effect parameter value"""
        return self.parameters.get(param_name)
    
    def reset_parameters(self):
        """Reset all parameters to defaults"""
        for param, default_value in self.parameter_defaults.items():
            self.parameters[param] = default_value
    
    def _on_parameter_changed(self, param_name: str, value: float):
        """Override in subclasses to handle parameter changes"""
        pass
    
    def enable(self, enabled: bool = True):
        """Enable/disable effect"""
        self.enabled = enabled
    
    def bypass(self, bypassed: bool = True):
        """Bypass/unbypass effect"""
        self.bypassed = bypassed
    
    def set_wet_dry_mix(self, mix: float):
        """Set wet/dry mix ratio"""
        self.wet_dry_mix = max(0.0, min(1.0, mix))
    
    def _apply_wet_dry_mix(self, dry_signal: np.ndarray, wet_signal: np.ndarray) -> np.ndarray:
        """Apply wet/dry mixing"""
        return dry_signal * (1.0 - self.wet_dry_mix) + wet_signal * self.wet_dry_mix
    
    def get_info(self) -> Dict[str, Any]:
        """Get effect information"""
        return {
            'id': self.effect_id,
            'type': self.effect_type.value,
            'enabled': self.enabled,
            'bypassed': self.bypassed,
            'wet_dry_mix': self.wet_dry_mix,
            'parameters': self.parameters.copy(),
            'processing_time_ms': self.processing_time_ms,
            'cpu_usage_percent': self.cpu_usage_percent
        }


class EffectsChain(QObject):
    """
    Professional effects chain supporting 6 effect slots per track.
    Handles effect routing, processing, and management.
    """
    
    # Signals for GUI updates
    effect_added = pyqtSignal(int, str)      # slot, effect_id
    effect_removed = pyqtSignal(int, str)    # slot, effect_id
    effect_bypassed = pyqtSignal(int, bool)  # slot, bypassed
    parameter_changed = pyqtSignal(str, str, float)  # effect_id, param, value
    chain_processed = pyqtSignal(float)      # processing_time_ms
    
    def __init__(self, chain_id: str, max_slots: int = 6, 
                 sample_rate: int = 44100, buffer_size: int = 512):
        super().__init__()
        
        self.chain_id = chain_id
        self.max_slots = max_slots
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        
        # Effects chain slots
        self.effect_slots: List[Optional[BaseEffect]] = [None] * max_slots
        self.slot_enabled: List[bool] = [True] * max_slots
        
        # Chain settings
        self.chain_enabled = True
        self.chain_bypassed = False
        self.master_wet_dry = 1.0
        
        # Thread safety
        self._lock = RLock()
        
        # Performance monitoring
        self.total_processing_time = 0.0
        self.peak_processing_time = 0.0
        self.average_processing_time = 0.0
        self.processing_history = []
        self.max_history_size = 100
        
        # Audio buffers for processing
        self.temp_buffers = []
        for i in range(max_slots + 1):  # +1 for input buffer
            self.temp_buffers.append(np.zeros((buffer_size, 2), dtype=np.float32))
    
    def add_effect(self, slot: int, effect: BaseEffect) -> bool:
        """Add effect to specified slot"""
        with self._lock:
            if slot < 0 or slot >= self.max_slots:
                return False
            
            if self.effect_slots[slot] is not None:
                return False  # Slot already occupied
            
            # Configure effect for this chain
            effect.sample_rate = self.sample_rate
            effect.buffer_size = self.buffer_size
            
            self.effect_slots[slot] = effect
            self.effect_added.emit(slot, effect.effect_id)
            return True
    
    def remove_effect(self, slot: int) -> bool:
        """Remove effect from specified slot"""
        with self._lock:
            if slot < 0 or slot >= self.max_slots:
                return False
            
            effect = self.effect_slots[slot]
            if effect is None:
                return False
            
            effect_id = effect.effect_id
            self.effect_slots[slot] = None
            self.effect_removed.emit(slot, effect_id)
            return True
    
    def move_effect(self, from_slot: int, to_slot: int) -> bool:
        """Move effect from one slot to another"""
        with self._lock:
            if (from_slot < 0 or from_slot >= self.max_slots or 
                to_slot < 0 or to_slot >= self.max_slots):
                return False
            
            if self.effect_slots[from_slot] is None:
                return False
            
            if self.effect_slots[to_slot] is not None:
                return False  # Destination slot occupied
            
            # Move effect
            effect = self.effect_slots[from_slot]
            self.effect_slots[from_slot] = None
            self.effect_slots[to_slot] = effect
            
            self.effect_removed.emit(from_slot, effect.effect_id)
            self.effect_added.emit(to_slot, effect.effect_id)
            return True
    
    def swap_effects(self, slot1: int, slot2: int) -> bool:
        """Swap effects between two slots"""
        with self._lock:
            if (slot1 < 0 or slot1 >= self.max_slots or 
                slot2 < 0 or slot2 >= self.max_slots):
                return False
            
            # Swap effects
            effect1 = self.effect_slots[slot1]
            effect2 = self.effect_slots[slot2]
            
            self.effect_slots[slot1] = effect2
            self.effect_slots[slot2] = effect1
            
            # Emit signals for both slots
            if effect1:
                self.effect_removed.emit(slot1, effect1.effect_id)
            if effect2:
                self.effect_removed.emit(slot2, effect2.effect_id)
            if effect2:
                self.effect_added.emit(slot1, effect2.effect_id)
            if effect1:
                self.effect_added.emit(slot2, effect1.effect_id)
            
            return True
    
    def get_effect(self, slot: int) -> Optional[BaseEffect]:
        """Get effect in specified slot"""
        if slot < 0 or slot >= self.max_slots:
            return None
        return self.effect_slots[slot]
    
    def find_effect(self, effect_id: str) -> Optional[Tuple[int, BaseEffect]]:
        """Find effect by ID, return (slot, effect) or None"""
        for slot, effect in enumerate(self.effect_slots):
            if effect and effect.effect_id == effect_id:
                return slot, effect
        return None
    
    def enable_slot(self, slot: int, enabled: bool = True):
        """Enable/disable processing for specific slot"""
        if slot >= 0 and slot < self.max_slots:
            self.slot_enabled[slot] = enabled
    
    def bypass_effect(self, slot: int, bypassed: bool = True):
        """Bypass/unbypass effect in specific slot"""
        effect = self.get_effect(slot)
        if effect:
            effect.bypass(bypassed)
            self.effect_bypassed.emit(slot, bypassed)
    
    def set_effect_parameter(self, effect_id: str, param_name: str, value: float) -> bool:
        """Set parameter for specific effect"""
        result = self.find_effect(effect_id)
        if result:
            slot, effect = result
            if effect.set_parameter(param_name, value):
                self.parameter_changed.emit(effect_id, param_name, value)
                return True
        return False
    
    def process(self, input_data: np.ndarray, frames: int) -> np.ndarray:
        """
        Process audio through the entire effects chain.
        Optimized for real-time performance.
        """
        start_time = time.perf_counter()
        
        try:
            # If chain is bypassed or disabled, return input
            if self.chain_bypassed or not self.chain_enabled:
                return input_data
            
            # Start with input data
            current_buffer = 0
            self.temp_buffers[current_buffer][:frames] = input_data
            
            # Process through each slot
            for slot in range(self.max_slots):
                if not self.slot_enabled[slot]:
                    continue
                
                effect = self.effect_slots[slot]
                if effect is None or not effect.enabled or effect.bypassed:
                    continue
                
                # Process through effect
                input_buffer = self.temp_buffers[current_buffer]
                output_buffer = self.temp_buffers[(current_buffer + 1) % len(self.temp_buffers)]
                
                processed_audio = effect.process(input_buffer[:frames], frames)
                output_buffer[:frames] = processed_audio
                
                # Move to next buffer
                current_buffer = (current_buffer + 1) % len(self.temp_buffers)
            
            # Get final output
            output_data = self.temp_buffers[current_buffer][:frames].copy()
            
            # Apply master wet/dry mix
            if self.master_wet_dry < 1.0:
                output_data = self._apply_master_wet_dry(input_data, output_data)
            
            # Performance tracking
            processing_time = (time.perf_counter() - start_time) * 1000
            self._update_performance_stats(processing_time)
            
            return output_data
            
        except Exception as e:
            # Fail-safe: return input on error
            print(f"Effects chain error: {e}")
            return input_data
    
    def _apply_master_wet_dry(self, dry_signal: np.ndarray, wet_signal: np.ndarray) -> np.ndarray:
        """Apply master wet/dry mix"""
        return dry_signal * (1.0 - self.master_wet_dry) + wet_signal * self.master_wet_dry
    
    def _update_performance_stats(self, processing_time: float):
        """Update performance statistics"""
        self.total_processing_time = processing_time
        self.peak_processing_time = max(self.peak_processing_time, processing_time)
        
        # Update running average
        self.processing_history.append(processing_time)
        if len(self.processing_history) > self.max_history_size:
            self.processing_history.pop(0)
        
        self.average_processing_time = np.mean(self.processing_history)
        
        # Emit signal for GUI updates
        self.chain_processed.emit(processing_time)
    
    def enable_chain(self, enabled: bool = True):
        """Enable/disable entire effects chain"""
        self.chain_enabled = enabled
    
    def bypass_chain(self, bypassed: bool = True):
        """Bypass/unbypass entire effects chain"""
        self.chain_bypassed = bypassed
    
    def set_master_wet_dry(self, mix: float):
        """Set master wet/dry mix for entire chain"""
        self.master_wet_dry = max(0.0, min(1.0, mix))
    
    def clear_all_effects(self):
        """Remove all effects from the chain"""
        with self._lock:
            for slot in range(self.max_slots):
                if self.effect_slots[slot] is not None:
                    effect_id = self.effect_slots[slot].effect_id
                    self.effect_slots[slot] = None
                    self.effect_removed.emit(slot, effect_id)
    
    def get_chain_info(self) -> Dict[str, Any]:
        """Get comprehensive chain information"""
        effects_info = []
        for slot, effect in enumerate(self.effect_slots):
            if effect:
                info = effect.get_info()
                info['slot'] = slot
                info['slot_enabled'] = self.slot_enabled[slot]
                effects_info.append(info)
        
        return {
            'chain_id': self.chain_id,
            'chain_enabled': self.chain_enabled,
            'chain_bypassed': self.chain_bypassed,
            'master_wet_dry': self.master_wet_dry,
            'max_slots': self.max_slots,
            'active_effects': len([e for e in self.effect_slots if e is not None]),
            'effects': effects_info,
            'performance': {
                'total_processing_time_ms': self.total_processing_time,
                'peak_processing_time_ms': self.peak_processing_time,
                'average_processing_time_ms': self.average_processing_time
            }
        }
    
    def save_preset(self, name: str) -> Dict[str, Any]:
        """Save current chain configuration as preset"""
        preset = {
            'name': name,
            'chain_enabled': self.chain_enabled,
            'master_wet_dry': self.master_wet_dry,
            'effects': []
        }
        
        for slot, effect in enumerate(self.effect_slots):
            if effect:
                effect_data = {
                    'slot': slot,
                    'effect_type': effect.effect_type.value,
                    'effect_id': effect.effect_id,
                    'enabled': effect.enabled,
                    'bypassed': effect.bypassed,
                    'wet_dry_mix': effect.wet_dry_mix,
                    'parameters': effect.parameters.copy()
                }
                preset['effects'].append(effect_data)
        
        return preset
    
    def load_preset(self, preset: Dict[str, Any], effect_factory: 'EffectFactory') -> bool:
        """Load chain configuration from preset"""
        try:
            with self._lock:
                # Clear existing effects
                self.clear_all_effects()
                
                # Apply chain settings
                self.chain_enabled = preset.get('chain_enabled', True)
                self.master_wet_dry = preset.get('master_wet_dry', 1.0)
                
                # Load effects
                for effect_data in preset.get('effects', []):
                    slot = effect_data['slot']
                    effect_type = EffectType(effect_data['effect_type'])
                    
                    # Create effect using factory
                    effect = effect_factory.create_effect(effect_type, effect_data['effect_id'])
                    if effect:
                        # Configure effect
                        effect.enabled = effect_data.get('enabled', True)
                        effect.bypassed = effect_data.get('bypassed', False)
                        effect.wet_dry_mix = effect_data.get('wet_dry_mix', 1.0)
                        
                        # Set parameters
                        for param_name, value in effect_data.get('parameters', {}).items():
                            effect.set_parameter(param_name, value)
                        
                        # Add to chain
                        self.add_effect(slot, effect)
                
                return True
                
        except Exception as e:
            print(f"Error loading preset: {e}")
            return False
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.peak_processing_time = 0.0
        self.processing_history.clear()
        self.average_processing_time = 0.0
    
    def __str__(self) -> str:
        active_effects = [f"Slot {i}: {effect.effect_type.value}" 
                         for i, effect in enumerate(self.effect_slots) 
                         if effect is not None]
        return f"EffectsChain(id={self.chain_id}, effects=[{', '.join(active_effects)}])"


class EffectFactory:
    """
    Factory for creating audio effects.
    Manages effect types and instantiation.
    """
    
    def __init__(self):
        # Registry of available effects
        self._effect_registry: Dict[EffectType, Type[BaseEffect]] = {}
        
        # Register built-in effects (will be implemented in separate files)
        self._register_builtin_effects()
    
    def _register_builtin_effects(self):
        """Register built-in effect types"""
        # These will be imported from the processors module
        # For now, we'll register placeholders
        pass
    
    def register_effect(self, effect_type: EffectType, effect_class: Type[BaseEffect]):
        """Register a new effect type"""
        self._effect_registry[effect_type] = effect_class
    
    def create_effect(self, effect_type: EffectType, effect_id: str, 
                     sample_rate: int = 44100, buffer_size: int = 512) -> Optional[BaseEffect]:
        """Create an effect instance"""
        if effect_type not in self._effect_registry:
            return None
        
        effect_class = self._effect_registry[effect_type]
        try:
            return effect_class(effect_id, effect_type, sample_rate, buffer_size)
        except Exception as e:
            print(f"Error creating effect {effect_type}: {e}")
            return None
    
    def get_available_effects(self) -> List[EffectType]:
        """Get list of available effect types"""
        return list(self._effect_registry.keys())
    
    def get_effect_info(self, effect_type: EffectType) -> Optional[Dict[str, Any]]:
        """Get information about an effect type"""
        if effect_type not in self._effect_registry:
            return None
        
        effect_class = self._effect_registry[effect_type]
        # This would return metadata about the effect
        # For now, return basic info
        return {
            'type': effect_type.value,
            'name': effect_type.value.title(),
            'class': effect_class.__name__
        }


# Global effect factory instance
effect_factory = EffectFactory()