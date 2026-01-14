# etude/utils/__init__.py

"""
Utility functions and tools for the Etude project.

Modules:
    - logger: Structured logging system
    - model_loader: Functions for loading pretrained models (import separately to avoid circular imports)
    - preprocess: Audio preprocessing utilities
    - download: Audio download utilities
    - training_utils: Training helpers (checkpoints, seeds, etc.)

Note:
    To avoid circular imports, use explicit imports for model_loader:
        from etude.utils.model_loader import load_etude_decoder
"""

from .logger import logger
from .download import download_audio_from_url
from .training_utils import set_seed, save_checkpoint, load_checkpoint

__all__ = [
    "logger",
    "download_audio_from_url",
    "set_seed",
    "save_checkpoint",
    "load_checkpoint",
]
