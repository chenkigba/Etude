# etude/models/__init__.py

"""
Neural network models for the Etude project.

Main models:
    - EtudeDecoder: Core transformer model for piano cover generation
    - Demixed_DilatedTransformerModel: Beat detection model
    - Model_SPEC2MIDI: AMT-APC automatic music transcription model
    - HFT_Transformer: Time signature detection model
"""

from .etude_decoder import EtudeDecoder, EtudeDecoderConfig
from .beat_transformer import Demixed_DilatedTransformerModel
from .amt_apc import Model_SPEC2MIDI, Encoder_SPEC2MIDI, Decoder_SPEC2MIDI
from .hft_transformer import HFT_Transformer

__all__ = [
    "EtudeDecoder",
    "EtudeDecoderConfig",
    "Demixed_DilatedTransformerModel",
    "Model_SPEC2MIDI",
    "Encoder_SPEC2MIDI",
    "Decoder_SPEC2MIDI",
    "HFT_Transformer",
]
