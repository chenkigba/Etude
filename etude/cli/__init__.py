# etude/cli/__init__.py

"""
CLI commands for the Etude project.

Available commands after installation:
    - etude-infer: Generate piano covers from audio
    - etude-prepare: Prepare training data
    - etude-train: Train the EtudeDecoder model
    - etude-evaluate: Run evaluation metrics
    - etude-separate: Run audio source separation (utility)
"""

from .infer import InferencePipeline

__all__ = [
    "InferencePipeline",
]
