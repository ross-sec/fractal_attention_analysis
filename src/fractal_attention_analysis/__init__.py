"""
Fractal-Attention Analysis (FAA) Framework for LLM Interpretability

A mathematical framework for analyzing transformer attention mechanisms using
fractal geometry and golden ratio transformations.

Copyright (c) 2025 Ross Technologies & Hooking LTD
Licensed under MIT License
"""

__version__ = "0.1.0"
__author__ = "Andre Ross, Leorah Ross, Eyal Atias"
__email__ = "devops.ross@gmail.com"
__license__ = "MIT"

from .core import FractalAttentionAnalyzer
from .fractal import FractalTransforms
from .metrics import AttentionMetrics
from .visualization import AttentionVisualizer
from .utils import ModelLoader, DeviceManager

__all__ = [
    "FractalAttentionAnalyzer",
    "FractalTransforms",
    "AttentionMetrics",
    "AttentionVisualizer",
    "ModelLoader",
    "DeviceManager",
]
