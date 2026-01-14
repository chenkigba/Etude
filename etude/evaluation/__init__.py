# etude/evaluation/__init__.py

"""
Evaluation framework for the Etude project.

Provides tools for evaluating generated piano covers using various metrics:
    - WPD (Wrong Pitch Duration): Measures pitch accuracy
    - RGC (Rhythmic Grove Consistency): Measures rhythmic consistency
    - IPE (Inter-Phrase Expression): Measures phrase expression consistency
"""

from .runner import EvaluationRunner
from .reporting import ReportGenerator

__all__ = [
    "EvaluationRunner",
    "ReportGenerator",
]
