"""
Attention metrics and evaluation functions.

This module provides metrics for evaluating attention patterns including
entropy, sparsity, and interpretability scores.
"""

import numpy as np
from scipy.stats import entropy as scipy_entropy
from typing import Dict, Any, Optional


class AttentionMetrics:
    """Computes various metrics for attention analysis."""
    
    def __init__(self):
        """Initialize attention metrics calculator."""
        self.epsilon = 1e-10  # Small constant to avoid log(0)
        
    def compute_entropy(self, attention_weights: np.ndarray) -> float:
        """
        Compute Shannon entropy of attention distribution.
        
        Args:
            attention_weights: Attention matrix [seq_len, seq_len]
            
        Returns:
            Entropy value
        """
        # Flatten and normalize
        flat = attention_weights.flatten()
        flat = flat + self.epsilon  # Avoid log(0)
        flat = flat / flat.sum()
        
        return scipy_entropy(flat)
    
    def compute_sparsity(self, attention_weights: np.ndarray, threshold: float = 0.01) -> float:
        """
        Compute sparsity of attention matrix.
        
        Args:
            attention_weights: Attention matrix
            threshold: Threshold below which values are considered zero
            
        Returns:
            Sparsity ratio (0-1, higher = more sparse)
        """
        total_elements = attention_weights.size
        sparse_elements = np.sum(attention_weights < threshold)
        
        return sparse_elements / total_elements
    
    def compute_concentration(self, attention_weights: np.ndarray, top_k: int = 5) -> float:
        """
        Compute attention concentration (how much attention is on top-k tokens).
        
        Args:
            attention_weights: Attention matrix
            top_k: Number of top tokens to consider
            
        Returns:
            Concentration ratio (0-1)
        """
        # Average attention per token
        token_attention = attention_weights.mean(axis=0)
        
        # Sort and get top-k
        top_k_attention = np.sort(token_attention)[-top_k:].sum()
        total_attention = token_attention.sum()
        
        if total_attention == 0:
            return 0.0
        
        return top_k_attention / total_attention
    
    def compute_uniformity(self, attention_weights: np.ndarray) -> float:
        """
        Compute how uniform the attention distribution is.
        
        Args:
            attention_weights: Attention matrix
            
        Returns:
            Uniformity score (0-1, higher = more uniform)
        """
        # Flatten and normalize
        flat = attention_weights.flatten()
        flat = flat / (flat.sum() + self.epsilon)
        
        # Compare to uniform distribution
        uniform = np.ones_like(flat) / len(flat)
        
        # KL divergence from uniform
        kl_div = scipy_entropy(flat + self.epsilon, uniform + self.epsilon)
        
        # Convert to similarity (0-1)
        uniformity = np.exp(-kl_div)
        
        return uniformity
    
    def compute_entropy_reduction(
        self,
        original_attention: np.ndarray,
        transformed_attention: np.ndarray
    ) -> float:
        """
        Compute entropy reduction after transformation.
        
        Args:
            original_attention: Original attention matrix
            transformed_attention: Transformed attention matrix
            
        Returns:
            Entropy reduction (positive = reduction, negative = increase)
        """
        original_entropy = self.compute_entropy(original_attention)
        transformed_entropy = self.compute_entropy(transformed_attention)
        
        # Handle infinite entropy (perfect compression)
        if np.isinf(original_entropy) and not np.isinf(transformed_entropy):
            return float('inf')
        elif np.isinf(transformed_entropy):
            return float('-inf')
        
        return original_entropy - transformed_entropy
    
    def compute_sparsity_improvement(
        self,
        original_attention: np.ndarray,
        transformed_attention: np.ndarray,
        threshold: float = 0.01
    ) -> float:
        """
        Compute sparsity improvement after transformation.
        
        Args:
            original_attention: Original attention matrix
            transformed_attention: Transformed attention matrix
            threshold: Sparsity threshold
            
        Returns:
            Sparsity improvement (positive = more sparse)
        """
        original_sparsity = self.compute_sparsity(original_attention, threshold)
        transformed_sparsity = self.compute_sparsity(transformed_attention, threshold)
        
        return transformed_sparsity - original_sparsity
    
    def compute_interpretability_score(
        self,
        attention_weights: np.ndarray,
        fractal_dimension: float
    ) -> float:
        """
        Compute overall interpretability score.
        
        Combines multiple metrics into a single interpretability score.
        
        Args:
            attention_weights: Attention matrix
            fractal_dimension: Computed fractal dimension
            
        Returns:
            Interpretability score (0-1, higher = more interpretable)
        """
        # Component scores
        sparsity = self.compute_sparsity(attention_weights)
        concentration = self.compute_concentration(attention_weights)
        uniformity = 1 - self.compute_uniformity(attention_weights)  # Invert (less uniform = more interpretable)
        
        # Fractal complexity (closer to 2.0 = more interpretable)
        fractal_score = 1 - abs(fractal_dimension - 2.0) / 2.0
        fractal_score = max(0, min(1, fractal_score))
        
        # Weighted combination
        score = (
            0.3 * sparsity +
            0.3 * concentration +
            0.2 * uniformity +
            0.2 * fractal_score
        )
        
        return score
    
    def compute_attention_flow(self, attention_weights: np.ndarray) -> Dict[str, float]:
        """
        Compute attention flow metrics (forward, backward, self-attention).
        
        Args:
            attention_weights: Attention matrix [seq_len, seq_len]
            
        Returns:
            Dictionary with flow metrics
        """
        seq_len = attention_weights.shape[0]
        
        # Extract triangular parts
        upper_tri = np.triu(attention_weights, k=1)  # Forward attention
        lower_tri = np.tril(attention_weights, k=-1)  # Backward attention
        diagonal = np.diag(attention_weights)  # Self-attention
        
        total = attention_weights.sum()
        
        if total == 0:
            return {
                'forward_flow': 0.0,
                'backward_flow': 0.0,
                'self_attention': 0.0
            }
        
        return {
            'forward_flow': upper_tri.sum() / total,
            'backward_flow': lower_tri.sum() / total,
            'self_attention': diagonal.sum() / total
        }
    
    def compute_all_metrics(
        self,
        attention_weights: np.ndarray,
        fractal_dimension: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Compute all available metrics.
        
        Args:
            attention_weights: Attention matrix
            fractal_dimension: Optional fractal dimension
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'entropy': self.compute_entropy(attention_weights),
            'sparsity': self.compute_sparsity(attention_weights),
            'concentration': self.compute_concentration(attention_weights),
            'uniformity': self.compute_uniformity(attention_weights),
        }
        
        # Add flow metrics
        metrics.update(self.compute_attention_flow(attention_weights))
        
        # Add interpretability score if fractal dimension provided
        if fractal_dimension is not None:
            metrics['fractal_dimension'] = fractal_dimension
            metrics['interpretability_score'] = self.compute_interpretability_score(
                attention_weights,
                fractal_dimension
            )
        
        return metrics

