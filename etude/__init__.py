# etude/__init__.py

"""
Etude: A controllable piano cover generation framework.

This package provides tools for generating piano covers from audio using
deep learning models based on the GPT-NeoX architecture.

Usage:
    # Configuration
    from etude.config import load_config, EtudeConfig

    # Data processing
    from etude.data import EtudeDataset, Vocab, TinyREMITokenizer

    # Models
    from etude.models import EtudeDecoder, EtudeDecoderConfig

    # Evaluation
    from etude.evaluation import EvaluationRunner, ReportGenerator

    # CLI pipelines
    from etude.cli.infer import InferencePipeline
"""

__version__ = "0.1.0"
__author__ = "Xiugapurin"

from .config import load_config, EtudeConfig

__all__ = [
    "__version__",
    "__author__",
    "load_config",
    "EtudeConfig",
]
