"""Tests for fractal transformation module."""

import numpy as np
import pytest
from fractal_attention_analysis.fractal import FractalTransforms


class TestFractalTransforms:
    """Test suite for FractalTransforms class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.transforms = FractalTransforms()
        self.test_matrix = np.random.rand(10, 10)
        
    def test_golden_ratio_value(self):
        """Test that golden ratio is correctly computed."""
        expected_phi = (1 + np.sqrt(5)) / 2
        assert np.isclose(self.transforms.phi, expected_phi)
        
    def test_golden_ratio_partition(self):
        """Test golden ratio partitioning of attention matrix."""
        major, minor = self.transforms.golden_ratio_partition(self.test_matrix)
        
        # Check that partitions are non-empty
        assert major.size > 0
        assert minor.size > 0
        
        # Check dimensions
        assert major.shape[0] == major.shape[1]
        assert minor.shape[0] == minor.shape[1]
        
    def test_fractal_interpolation(self):
        """Test fractal interpolation function."""
        transformed = self.transforms.fractal_interpolation_function(self.test_matrix)
        
        # Check output shape matches input
        assert transformed.shape == self.test_matrix.shape
        
        # Check output is finite
        assert np.all(np.isfinite(transformed))
        
    def test_fractal_dimension_range(self):
        """Test that computed fractal dimension is in valid range."""
        dimension = self.transforms.compute_fractal_dimension(self.test_matrix)
        
        # Fractal dimension should be between 0 and 3 for 2D patterns
        assert 0 < dimension < 3
        
    def test_fractal_dimension_with_custom_scales(self):
        """Test fractal dimension with custom scales."""
        scales = [2, 4, 8]
        dimension = self.transforms.compute_fractal_dimension(
            self.test_matrix,
            scales=scales
        )
        
        assert 0 < dimension < 3
        
    def test_self_similarity(self):
        """Test self-similarity computation."""
        similarity = self.transforms.compute_self_similarity(self.test_matrix)
        
        # Similarity should be between 0 and 1
        assert 0 <= similarity <= 1
        
    def test_golden_ratio_scoring(self):
        """Test golden ratio scoring."""
        scored = self.transforms.golden_ratio_scoring(self.test_matrix)
        
        # Check output shape
        assert scored.shape == self.test_matrix.shape
        
        # Check output is finite
        assert np.all(np.isfinite(scored))
        
    def test_multi_scale_analysis(self):
        """Test multi-scale fractal analysis."""
        results = self.transforms.multi_scale_analysis(self.test_matrix, num_scales=3)
        
        # Check we get correct number of scales
        assert len(results) == 3
        
        # Check each result has scale and matrix
        for scale, matrix in results:
            assert isinstance(scale, float)
            assert isinstance(matrix, np.ndarray)
            assert matrix.shape == self.test_matrix.shape
            
    def test_empty_matrix_handling(self):
        """Test handling of edge cases."""
        small_matrix = np.random.rand(2, 2)
        
        # Should not raise error
        dimension = self.transforms.compute_fractal_dimension(small_matrix)
        assert isinstance(dimension, float)
        
    def test_uniform_matrix(self):
        """Test with uniform attention matrix."""
        uniform = np.ones((10, 10)) * 0.5
        dimension = self.transforms.compute_fractal_dimension(uniform)
        
        # Uniform pattern should have low fractal dimension
        assert dimension > 0

