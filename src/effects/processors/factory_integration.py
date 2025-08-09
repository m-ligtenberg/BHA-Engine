"""
Effect Factory Integration
Registers all available effects with the factory system
"""

from ..effects_chain import effect_factory, EffectType
from .compressor import Compressor
from .eq import ThreeBandEQ
from .reverb import ReverbProcessor
from .delay import BPMSyncedDelay
from .saturation import AnalogSaturation


def register_all_effects():
    """Register all available effects with the global factory"""
    
    # Register compressor
    effect_factory.register_effect(EffectType.COMPRESSOR, Compressor)
    
    # Register EQ
    effect_factory.register_effect(EffectType.EQ, ThreeBandEQ)
    
    # Register reverb
    effect_factory.register_effect(EffectType.REVERB, ReverbProcessor)
    
    # Register delay
    effect_factory.register_effect(EffectType.DELAY, BPMSyncedDelay)
    
    # Register saturation
    effect_factory.register_effect(EffectType.SATURATION, AnalogSaturation)
    
    print("Audio effects registered successfully:")
    print(f"- Available effects: {len(effect_factory.get_available_effects())}")
    for effect_type in effect_factory.get_available_effects():
        print(f"  * {effect_type.value}")


# Auto-register effects when module is imported
register_all_effects()