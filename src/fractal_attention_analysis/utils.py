"""
Utility classes for model loading and device management.

This module provides helper classes for managing LLM loading and GPU/CPU device allocation.
"""

import torch
import warnings
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM


class DeviceManager:
    """Manages device allocation and memory for model inference."""
    
    def __init__(self, prefer_gpu: bool = True, max_gpu_memory_gb: int = 20):
        """
        Initialize device manager.
        
        Args:
            prefer_gpu: Whether to prefer GPU if available
            max_gpu_memory_gb: Maximum GPU memory to use (reserves buffer)
        """
        self.prefer_gpu = prefer_gpu
        self.max_gpu_memory_gb = max_gpu_memory_gb
        self.device = self._select_device()
        
    def _select_device(self) -> torch.device:
        """Select the best available device."""
        if self.prefer_gpu and torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    
    def get_device_config(self, model_size_params: int) -> Dict[str, Any]:
        """
        Get device configuration based on model size.
        
        Args:
            model_size_params: Number of model parameters
            
        Returns:
            Dictionary with device configuration
        """
        config = {}
        
        if not torch.cuda.is_available():
            return config
            
        # For models > 500M parameters, use balanced device mapping
        if model_size_params > 500_000_000:
            config['device_map'] = 'balanced'
            config['max_memory'] = {
                0: f"{self.max_gpu_memory_gb}GB",
                "cpu": "32GB"
            }
            config['dtype'] = torch.float16
        else:
            config['device_map'] = 'cuda'
            
        return config
    
    @staticmethod
    def log_memory_usage(stage: str = ""):
        """Log current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory {stage}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


class ModelLoader:
    """Handles loading and configuration of transformer models."""
    
    def __init__(self, device_manager: Optional[DeviceManager] = None):
        """
        Initialize model loader.
        
        Args:
            device_manager: DeviceManager instance for device allocation
        """
        self.device_manager = device_manager or DeviceManager()
        
    def load_model(
        self,
        model_name: str,
        force_eager_attention: bool = True,
        **kwargs
    ) -> tuple:
        """
        Load a transformer model and tokenizer.
        
        Args:
            model_name: HuggingFace model identifier
            force_eager_attention: Force eager attention implementation
            **kwargs: Additional arguments for model loading
            
        Returns:
            Tuple of (model, tokenizer)
        """
        print(f"Loading model: {model_name}")
        
        # Load tokenizer
        tokenizer = self._load_tokenizer(model_name)
        
        # Prepare model kwargs
        model_kwargs = self._prepare_model_kwargs(
            model_name,
            force_eager_attention,
            **kwargs
        )
        
        # Load model
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # Apply model-specific fixes
            model = self._apply_model_fixes(model, model_name)
            
            model.eval()
            print(f"Model loaded successfully on device: {next(model.parameters()).device}")
            
            return model, tokenizer
            
        except Exception as e:
            print(f"Error loading model with device mapping: {e}")
            return self._fallback_load(model_name, force_eager_attention)
    
    def _load_tokenizer(self, model_name: str) -> AutoTokenizer:
        """Load and configure tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Handle model-specific tokenizer configurations
        if 'gemma' in model_name.lower():
            if tokenizer.model_max_length is None or tokenizer.model_max_length > 100000:
                tokenizer.model_max_length = 2048
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return tokenizer
    
    def _prepare_model_kwargs(
        self,
        model_name: str,
        force_eager_attention: bool,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare model loading arguments."""
        model_kwargs = kwargs.copy()
        
        # Estimate model size (rough heuristic)
        model_size = self._estimate_model_size(model_name)
        
        # Get device configuration
        if torch.cuda.is_available():
            device_config = self.device_manager.get_device_config(model_size)
            model_kwargs.update(device_config)
        
        # Force eager attention if requested
        if force_eager_attention:
            model_kwargs['attn_implementation'] = 'eager'
            
        return model_kwargs
    
    def _estimate_model_size(self, model_name: str) -> int:
        """Estimate model size from name (rough heuristic)."""
        name_lower = model_name.lower()
        
        if '1b' in name_lower or '1.0b' in name_lower:
            return 1_000_000_000
        elif '0.6b' in name_lower or '600m' in name_lower:
            return 600_000_000
        elif 'gpt2' in name_lower and 'large' not in name_lower:
            return 117_000_000
        else:
            return 500_000_000  # Default estimate
    
    def _apply_model_fixes(self, model, model_name: str):
        """Apply model-specific configuration fixes."""
        # Fix for Qwen models
        if 'qwen' in model_name.lower():
            if hasattr(model, 'config'):
                model.config._attn_implementation = "eager"
                model.config.attn_implementation = "eager"
                if hasattr(model, '_attn_implementation'):
                    model._attn_implementation = "eager"
        
        return model
    
    def _fallback_load(self, model_name: str, force_eager_attention: bool) -> tuple:
        """Fallback loading strategy if main loading fails."""
        print("Attempting CPU fallback loading...")
        
        tokenizer = self._load_tokenizer(model_name)
        
        model_kwargs = {}
        if force_eager_attention:
            model_kwargs['attn_implementation'] = 'eager'
            
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        if torch.cuda.is_available():
            model = model.to('cuda')
            print("Model moved to GPU (CPU fallback)")
        else:
            print("Model loaded on CPU")
            
        model.eval()
        return model, tokenizer


# Suppress specific warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(
    "ignore",
    message=".*sdpa.*attention.*does not support.*output_attentions.*",
    category=UserWarning
)

