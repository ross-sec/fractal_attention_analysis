"""
Fractal transformation and interpolation functions.

This module implements the mathematical foundations of fractal analysis
including golden ratio transformations and fractal dimension calculations.
"""

import math
from typing import List, Optional, Tuple

import numpy as np


class FractalTransforms:
    """Implements fractal transformations and dimension calculations."""

    # Mathematical constants
    PHI = (1 + math.sqrt(5)) / 2  # Golden ratio

    # Fractal dimensions for different patterns
    D_SIERPINSKI = math.log(3) / math.log(2)  # ≈ 1.585
    D_KOCH = math.log(4) / math.log(3)  # ≈ 1.262
    D_DRAGON = 2.0  # Dragon curve
    D_NEURAL = (PHI**2) / 2  # Neural fractal dimension

    def __init__(self, phi: Optional[float] = None):
        """
        Initialize fractal transforms.

        Args:
            phi: Golden ratio value (defaults to (1+√5)/2)
        """
        self.phi = phi or self.PHI
        self.phi_inv = 1 / self.phi

    def golden_ratio_partition(self, attention_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Partition attention matrix using golden ratio.

        Args:
            attention_matrix: Input attention matrix [seq_len, seq_len]

        Returns:
            Tuple of (major_partition, minor_partition)
        """
        seq_len = attention_matrix.shape[0]
        split_idx = int(seq_len / self.phi)

        major = attention_matrix[:split_idx, :split_idx]
        minor = attention_matrix[split_idx:, split_idx:]

        return major, minor

    def fractal_interpolation_function(
        self, attention_weights: np.ndarray, scale_factor: float = 0.5
    ) -> np.ndarray:
        """
        Apply fractal interpolation to attention weights.

        Args:
            attention_weights: Input attention matrix
            scale_factor: Scaling factor for interpolation

        Returns:
            Transformed attention matrix
        """
        # Apply golden ratio scaling
        scaled = attention_weights * self.phi_inv

        # Apply fractal transformation using IFS (Iterated Function System)
        transformed = self._apply_ifs(scaled, scale_factor)

        return transformed

    def _apply_ifs(self, matrix: np.ndarray, scale: float) -> np.ndarray:
        """Apply Iterated Function System transformation."""
        # Affine transformation with golden ratio scaling
        w1 = scale * self.phi_inv
        w2 = scale * (1 - self.phi_inv)

        # Apply contractive mappings
        transformed = w1 * matrix + w2 * np.roll(matrix, 1, axis=0)

        return transformed

    def compute_fractal_dimension(
        self, attention_matrix: np.ndarray, scales: Optional[List[int]] = None
    ) -> float:
        """
        Compute fractal dimension using box-counting method.

        Args:
            attention_matrix: Input attention matrix
            scales: List of box sizes for counting (defaults to powers of 2)

        Returns:
            Estimated fractal dimension
        """
        if scales is None:
            scales = [2, 4, 8, 16, 32]

        # Binarize attention matrix
        threshold = np.median(attention_matrix)
        binary_matrix = (attention_matrix > threshold).astype(int)

        counts = []
        for scale in scales:
            count = self._box_count(binary_matrix, scale)
            if count > 0:
                counts.append((scale, count))

        if len(counts) < 2:
            return self.D_NEURAL  # Fallback to neural fractal dimension

        # Linear regression in log-log space
        log_scales = np.log([c[0] for c in counts])
        log_counts = np.log([c[1] for c in counts])

        # Fractal dimension = -slope
        coeffs = np.polyfit(log_scales, log_counts, 1)
        dimension = -coeffs[0]

        return dimension

    def _box_count(self, binary_matrix: np.ndarray, box_size: int) -> int:
        """Count non-empty boxes at given scale."""
        n, m = binary_matrix.shape

        # Ensure matrix is divisible by box_size
        n_boxes = n // box_size
        m_boxes = m // box_size

        if n_boxes == 0 or m_boxes == 0:
            return 0

        # Slice to make perfectly divisible
        sliced = binary_matrix[: n_boxes * box_size, : m_boxes * box_size]

        # Reshape and count non-empty boxes
        reshaped = sliced.reshape(n_boxes, box_size, m_boxes, box_size)
        box_sums = reshaped.sum(axis=(1, 3))

        return np.count_nonzero(box_sums)

    def multi_scale_analysis(
        self, attention_matrix: np.ndarray, num_scales: int = 5
    ) -> List[Tuple[float, np.ndarray]]:
        """
        Perform multi-scale fractal analysis.

        Args:
            attention_matrix: Input attention matrix
            num_scales: Number of scales to analyze

        Returns:
            List of (scale, transformed_matrix) tuples
        """
        results = []

        for i in range(num_scales):
            scale = self.phi**i
            transformed = self.fractal_interpolation_function(
                attention_matrix, scale_factor=1.0 / scale
            )
            results.append((scale, transformed))

        return results

    def compute_self_similarity(self, attention_matrix: np.ndarray, scale_factor: int = 2) -> float:
        """
        Compute self-similarity measure of attention patterns.

        Args:
            attention_matrix: Input attention matrix
            scale_factor: Downsampling factor

        Returns:
            Self-similarity score (0-1)
        """
        # Downsample matrix
        n = attention_matrix.shape[0]
        if n < scale_factor * 2:
            return 1.0

        downsampled = attention_matrix[::scale_factor, ::scale_factor]

        # Resize original to match downsampled
        target_size = downsampled.shape[0]
        resized = attention_matrix[
            : target_size * scale_factor : scale_factor, : target_size * scale_factor : scale_factor
        ]

        # Compute correlation
        if resized.size == 0 or downsampled.size == 0:
            return 1.0

        correlation = np.corrcoef(resized.flatten(), downsampled.flatten())[0, 1]

        return abs(correlation)

    def golden_ratio_scoring(self, attention_weights: np.ndarray) -> np.ndarray:
        """
        Apply golden ratio-based importance scoring.

        Args:
            attention_weights: Input attention weights

        Returns:
            Scored attention weights
        """
        # Apply golden ratio weighting
        seq_len = attention_weights.shape[0]
        golden_weights = np.array([self.phi ** (-i / seq_len) for i in range(seq_len)])

        # Normalize
        golden_weights = golden_weights / golden_weights.sum()

        # Apply to attention
        scored = attention_weights * golden_weights.reshape(-1, 1)

        return scored
