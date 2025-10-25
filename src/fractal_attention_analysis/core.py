"""
Core Fractal-Attention Analysis implementation.

This module provides the main FractalAttentionAnalyzer class that integrates
all components of the FAA framework.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .fractal import FractalTransforms
from .metrics import AttentionMetrics
from .utils import DeviceManager, ModelLoader
from .visualization import AttentionVisualizer


class FractalAttentionAnalyzer:
    """
    Main class for Fractal-Attention Analysis of transformer models.

    This class provides a unified interface for analyzing attention mechanisms
    in transformer-based language models using fractal geometry and golden ratio
    transformations.

    Example:
        >>> analyzer = FractalAttentionAnalyzer("gpt2")
        >>> results = analyzer.analyze("Hello world, this is a test.")
        >>> print(f"Fractal dimension: {results['fractal_dimension']:.4f}")
    """

    def __init__(
        self,
        model_name: str,
        device_manager: Optional[DeviceManager] = None,
        force_eager_attention: bool = True,
        **model_kwargs,
    ) -> None:
        """
        Initialize the Fractal-Attention Analyzer.

        Args:
            model_name: HuggingFace model identifier (e.g., "gpt2", "meta-llama/Llama-3.2-1B")
            device_manager: Optional DeviceManager for GPU/CPU allocation
            force_eager_attention: Force eager attention implementation for compatibility
            **model_kwargs: Additional arguments passed to model loading
        """
        self.model_name = model_name

        # Initialize components
        self.device_manager = device_manager or DeviceManager()
        self.model_loader = ModelLoader(self.device_manager)
        self.fractal_transforms = FractalTransforms()
        self.metrics = AttentionMetrics()
        self.visualizer = AttentionVisualizer()

        # Load model and tokenizer
        self.model, self.tokenizer = self.model_loader.load_model(
            model_name, force_eager_attention=force_eager_attention, **model_kwargs
        )

    def analyze(
        self,
        text: str,
        layer_idx: int = -1,
        head_idx: int = 0,
        return_visualizations: bool = False,
        save_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Perform fractal-attention analysis on input text.

        Args:
            text: Input text to analyze
            layer_idx: Transformer layer index (-1 for last layer)
            head_idx: Attention head index
            return_visualizations: Whether to return matplotlib figures
            save_dir: Optional directory to save visualizations

        Returns:
            Dictionary containing analysis results including:
                - fractal_dimension: Computed fractal dimension
                - metrics: Dictionary of attention metrics
                - transformed_attention: Fractal-transformed attention matrix
                - tokens: List of tokens
                - analysis_time: Time taken for analysis
                - visualizations: Optional matplotlib figures
        """
        start_time = time.time()

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )

        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Extract attention weights
        attention_weights = self._extract_attention(inputs, layer_idx, head_idx)

        # Perform fractal analysis
        fractal_results = self._fractal_analysis(attention_weights)

        # Compute metrics
        metrics = self.metrics.compute_all_metrics(
            attention_weights, fractal_dimension=fractal_results["fractal_dimension"]
        )

        # Prepare results
        results = {
            "fractal_dimension": fractal_results["fractal_dimension"],
            "transformed_attention": fractal_results["transformed_attention"],
            "golden_ratio_scored": fractal_results["golden_ratio_scored"],
            "metrics": metrics,
            "tokens": [self.tokenizer.decode([token_id]) for token_id in inputs["input_ids"][0]],
            "analysis_time": time.time() - start_time,
            "model_name": self.model_name,
            "layer_idx": layer_idx,
            "head_idx": head_idx,
        }

        # Generate visualizations if requested
        if return_visualizations or save_dir:
            results["visualizations"] = self._generate_visualizations(
                attention_weights, fractal_results, results["tokens"], save_dir
            )

        return results

    def analyze_batch(self, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Analyze multiple texts in batch.

        Args:
            texts: List of input texts
            **kwargs: Arguments passed to analyze()

        Returns:
            List of analysis results
        """
        results = []
        for text in texts:
            result = self.analyze(text, **kwargs)
            results.append(result)
        return results

    def compare_models(self, other_model_name: str, text: str, **kwargs) -> Dict[str, Any]:
        """
        Compare attention patterns between this model and another.

        Args:
            other_model_name: Name of model to compare against
            text: Input text for comparison
            **kwargs: Arguments passed to analyze()

        Returns:
            Dictionary with comparison results
        """
        # Analyze with current model
        results_self = self.analyze(text, **kwargs)

        # Create temporary analyzer for other model
        other_analyzer = FractalAttentionAnalyzer(
            other_model_name, device_manager=self.device_manager
        )
        results_other = other_analyzer.analyze(text, **kwargs)

        # Compute comparison metrics
        comparison = {
            "model_1": {
                "name": self.model_name,
                "fractal_dimension": results_self["fractal_dimension"],
                "metrics": results_self["metrics"],
            },
            "model_2": {
                "name": other_model_name,
                "fractal_dimension": results_other["fractal_dimension"],
                "metrics": results_other["metrics"],
            },
            "dimension_difference": abs(
                results_self["fractal_dimension"] - results_other["fractal_dimension"]
            ),
        }

        return comparison

    def _extract_attention(
        self, inputs: Dict[str, torch.Tensor], layer_idx: int, head_idx: int
    ) -> np.ndarray:
        """Extract attention weights from model."""
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        if outputs.attentions is None or len(outputs.attentions) == 0:
            # Fallback: create synthetic attention
            seq_len = inputs["input_ids"].shape[1]
            return self._create_synthetic_attention(seq_len)

        # Get attention from specified layer
        layer_idx = min(layer_idx, len(outputs.attentions) - 1)
        attention_layer = outputs.attentions[layer_idx]

        if attention_layer is None:
            seq_len = inputs["input_ids"].shape[1]
            return self._create_synthetic_attention(seq_len)

        # Extract specific head
        attention_weights = attention_layer[0, head_idx].cpu().numpy()

        # Convert float16 to float32 if needed
        if attention_weights.dtype == np.float16:
            attention_weights = attention_weights.astype(np.float32)

        return attention_weights  # type: ignore[no-any-return]

    def _create_synthetic_attention(self, seq_len: int) -> np.ndarray:
        """Create synthetic attention for testing/fallback."""
        attention = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            # Decay with distance
            for j in range(seq_len):
                distance = abs(i - j)
                attention[i, j] = np.exp(-distance / seq_len)

        # Normalize
        attention = attention / attention.sum(axis=1, keepdims=True)
        return attention  # type: ignore[no-any-return]

    def _fractal_analysis(self, attention_weights: np.ndarray) -> Dict[str, Any]:
        """Perform fractal analysis on attention weights."""
        # Compute fractal dimension
        fractal_dimension = self.fractal_transforms.compute_fractal_dimension(attention_weights)

        # Apply fractal transformation
        transformed_attention = self.fractal_transforms.fractal_interpolation_function(
            attention_weights
        )

        # Apply golden ratio scoring
        golden_ratio_scored = self.fractal_transforms.golden_ratio_scoring(attention_weights)

        # Compute self-similarity
        self_similarity = self.fractal_transforms.compute_self_similarity(attention_weights)

        return {
            "fractal_dimension": fractal_dimension,
            "transformed_attention": transformed_attention,
            "golden_ratio_scored": golden_ratio_scored,
            "self_similarity": self_similarity,
        }

    def _generate_visualizations(
        self,
        attention_weights: np.ndarray,
        fractal_results: Dict[str, Any],
        tokens: List[str],
        save_dir: Optional[Path],
    ) -> Dict[str, Any]:
        """Generate visualization figures."""
        visualizations = {}

        # Attention matrix
        fig1 = self.visualizer.plot_attention_matrix(
            attention_weights,
            tokens=tokens,
            title=f"Attention Matrix - {self.model_name}",
            save_path=save_dir / "attention_matrix.png" if save_dir else None,
        )
        visualizations["attention_matrix"] = fig1

        # Fractal comparison
        fig2 = self.visualizer.plot_fractal_comparison(
            attention_weights,
            fractal_results["transformed_attention"],
            title="Fractal Transformation",
            save_path=save_dir / "fractal_comparison.png" if save_dir else None,
        )
        visualizations["fractal_comparison"] = fig2

        # Token importance
        token_scores = attention_weights.mean(axis=0)
        fig3 = self.visualizer.plot_token_importance(
            token_scores,
            tokens,
            title="Token Importance Scores",
            save_path=save_dir / "token_importance.png" if save_dir else None,
        )
        visualizations["token_importance"] = fig3

        return visualizations

    def export_results(
        self, results: Dict[str, Any], output_path: Path, format: str = "json"
    ) -> None:
        """
        Export analysis results to file.

        Args:
            results: Analysis results dictionary
            output_path: Path to save results
            format: Output format ('json', 'csv', or 'npz')
        """
        import json

        import pandas as pd

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            # Remove non-serializable items
            export_data = {
                k: v
                for k, v in results.items()
                if k not in ["visualizations", "transformed_attention", "golden_ratio_scored"]
            }
            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

        elif format == "csv":
            # Flatten metrics for CSV
            flat_data = {
                "model_name": results["model_name"],
                "fractal_dimension": results["fractal_dimension"],
                "analysis_time": results["analysis_time"],
                **results["metrics"],
            }
            df = pd.DataFrame([flat_data])
            df.to_csv(output_path, index=False)

        elif format == "npz":
            # Save numpy arrays
            np.savez(
                output_path,
                transformed_attention=results["transformed_attention"],
                golden_ratio_scored=results["golden_ratio_scored"],
                fractal_dimension=results["fractal_dimension"],
            )
        else:
            raise ValueError(f"Unsupported format: {format}")

    def __repr__(self) -> str:
        """String representation of analyzer."""
        return f"FractalAttentionAnalyzer(model='{self.model_name}')"
