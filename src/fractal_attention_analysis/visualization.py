"""
Visualization utilities for attention patterns and fractal analysis.

This module provides functions for visualizing attention matrices,
fractal patterns, and analysis results.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class AttentionVisualizer:
    """Visualizes attention patterns and fractal analysis results."""

    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        """
        Initialize visualizer.

        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except:
            pass  # Use default style if specified style not available

        self.figsize = (12, 8)
        self.cmap = "viridis"

    def plot_attention_matrix(
        self,
        attention_matrix: np.ndarray,
        tokens: Optional[List[str]] = None,
        title: str = "Attention Matrix",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot attention matrix as heatmap.

        Args:
            attention_matrix: Attention weights matrix
            tokens: Optional list of token strings
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        sns.heatmap(attention_matrix, cmap=self.cmap, ax=ax, cbar_kws={"label": "Attention Weight"})

        if tokens:
            ax.set_xticklabels(tokens, rotation=45, ha="right")
            ax.set_yticklabels(tokens, rotation=0)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Target Token", fontsize=12)
        ax.set_ylabel("Source Token", fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_fractal_comparison(
        self,
        original: np.ndarray,
        transformed: np.ndarray,
        title: str = "Fractal Transformation Comparison",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot side-by-side comparison of original and transformed attention.

        Args:
            original: Original attention matrix
            transformed: Transformed attention matrix
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Original
        sns.heatmap(original, cmap=self.cmap, ax=ax1, cbar_kws={"label": "Weight"})
        ax1.set_title("Original Attention", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Target Token")
        ax1.set_ylabel("Source Token")

        # Transformed
        sns.heatmap(transformed, cmap=self.cmap, ax=ax2, cbar_kws={"label": "Weight"})
        ax2.set_title("Fractal-Transformed Attention", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Target Token")
        ax2.set_ylabel("Source Token")

        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        title: str = "Metrics Comparison",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot bar chart comparing metrics across different analyses.

        Args:
            metrics_dict: Dictionary mapping analysis names to metric dictionaries
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        # Extract metric names
        metric_names = list(next(iter(metrics_dict.values())).keys())

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, metric_name in enumerate(metric_names[:6]):  # Plot up to 6 metrics
            ax = axes[idx]

            names = list(metrics_dict.keys())
            values = [metrics_dict[name].get(metric_name, 0) for name in names]

            ax.bar(names, values, color="steelblue", alpha=0.7)
            ax.set_title(metric_name.replace("_", " ").title(), fontsize=10, fontweight="bold")
            ax.set_ylabel("Value")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(axis="y", alpha=0.3)

        # Hide unused subplots
        for idx in range(len(metric_names), len(axes)):
            axes[idx].axis("off")

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_fractal_dimension_analysis(
        self,
        scales: List[int],
        counts: List[int],
        dimension: float,
        title: str = "Fractal Dimension Analysis",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot fractal dimension box-counting analysis.

        Args:
            scales: List of box sizes
            counts: List of box counts
            dimension: Computed fractal dimension
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        log_scales = np.log(scales)
        log_counts = np.log(counts)

        # Plot data points
        ax.scatter(log_scales, log_counts, s=100, alpha=0.6, label="Data")

        # Plot fitted line
        coeffs = np.polyfit(log_scales, log_counts, 1)
        fitted_line = coeffs[0] * log_scales + coeffs[1]
        ax.plot(log_scales, fitted_line, "r--", linewidth=2, label=f"Fit (D = {dimension:.4f})")

        ax.set_xlabel("log(Box Size)", fontsize=12)
        ax.set_ylabel("log(Box Count)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_multi_scale_analysis(
        self,
        multi_scale_results: List[Tuple[float, np.ndarray]],
        title: str = "Multi-Scale Fractal Analysis",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot multi-scale fractal analysis results.

        Args:
            multi_scale_results: List of (scale, matrix) tuples
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        num_scales = len(multi_scale_results)
        cols = min(3, num_scales)
        rows = (num_scales + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

        if num_scales == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, (scale, matrix) in enumerate(multi_scale_results):
            ax = axes[idx]
            sns.heatmap(matrix, cmap=self.cmap, ax=ax, cbar=False)
            ax.set_title(f"Scale: {scale:.2f}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide unused subplots
        for idx in range(num_scales, len(axes)):
            axes[idx].axis("off")

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_token_importance(
        self,
        token_scores: np.ndarray,
        tokens: List[str],
        top_k: int = 10,
        title: str = "Token Importance Scores",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot token importance scores.

        Args:
            token_scores: Importance scores for each token
            tokens: List of token strings
            top_k: Number of top tokens to display
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        # Get top-k tokens
        top_indices = np.argsort(token_scores)[-top_k:][::-1]
        top_tokens = [tokens[i] for i in top_indices]
        top_scores = token_scores[top_indices]

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_k))
        bars = ax.barh(range(top_k), top_scores, color=colors)

        ax.set_yticks(range(top_k))
        ax.set_yticklabels(top_tokens)
        ax.set_xlabel("Importance Score", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, top_scores)):
            ax.text(score, i, f" {score:.3f}", va="center", fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    @staticmethod
    def close_all() -> None:
        """Close all matplotlib figures."""
        plt.close("all")
