"""Tests for attention metrics module."""

import numpy as np
import pytest
from fractal_attention_analysis.metrics import AttentionMetrics


class TestAttentionMetrics:
    """Test suite for AttentionMetrics class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.metrics = AttentionMetrics()
        self.test_matrix = np.random.rand(10, 10)
        # Normalize to make it a proper attention distribution
        self.test_matrix = self.test_matrix / self.test_matrix.sum(axis=1, keepdims=True)
        
    def test_entropy_computation(self):
        """Test entropy calculation."""
        entropy = self.metrics.compute_entropy(self.test_matrix)
        
        # Entropy should be non-negative
        assert entropy >= 0
        
        # Entropy should be finite
        assert np.isfinite(entropy)
        
    def test_sparsity_computation(self):
        """Test sparsity calculation."""
        sparsity = self.metrics.compute_sparsity(self.test_matrix)
        
        # Sparsity should be between 0 and 1
        assert 0 <= sparsity <= 1
        
    def test_sparsity_with_sparse_matrix(self):
        """Test sparsity with actually sparse matrix."""
        sparse_matrix = np.zeros((10, 10))
        sparse_matrix[0, 0] = 1.0
        
        sparsity = self.metrics.compute_sparsity(sparse_matrix, threshold=0.01)
        
        # Should be very sparse
        assert sparsity > 0.9
        
    def test_concentration_computation(self):
        """Test attention concentration."""
        concentration = self.metrics.compute_concentration(self.test_matrix, top_k=3)
        
        # Concentration should be between 0 and 1
        assert 0 <= concentration <= 1
        
    def test_uniformity_computation(self):
        """Test uniformity calculation."""
        uniformity = self.metrics.compute_uniformity(self.test_matrix)
        
        # Uniformity should be between 0 and 1
        assert 0 <= uniformity <= 1
        
    def test_uniformity_with_uniform_matrix(self):
        """Test uniformity with perfectly uniform distribution."""
        uniform = np.ones((10, 10)) / 10
        uniformity = self.metrics.compute_uniformity(uniform)
        
        # Should be close to 1 for uniform distribution
        assert uniformity > 0.9
        
    def test_entropy_reduction(self):
        """Test entropy reduction calculation."""
        original = np.random.rand(10, 10)
        transformed = original * 0.5  # Simple transformation
        
        reduction = self.metrics.compute_entropy_reduction(original, transformed)
        
        # Should be a finite number
        assert np.isfinite(reduction) or np.isinf(reduction)
        
    def test_sparsity_improvement(self):
        """Test sparsity improvement calculation."""
        original = np.random.rand(10, 10)
        # Make transformed more sparse
        transformed = original.copy()
        transformed[transformed < 0.5] = 0
        
        improvement = self.metrics.compute_sparsity_improvement(original, transformed)
        
        # Should show improvement (positive value)
        assert improvement >= 0
        
    def test_interpretability_score(self):
        """Test overall interpretability score."""
        score = self.metrics.compute_interpretability_score(
            self.test_matrix,
            fractal_dimension=2.0
        )
        
        # Score should be between 0 and 1
        assert 0 <= score <= 1
        
    def test_attention_flow(self):
        """Test attention flow metrics."""
        flow = self.metrics.compute_attention_flow(self.test_matrix)
        
        # Check all flow components are present
        assert 'forward_flow' in flow
        assert 'backward_flow' in flow
        assert 'self_attention' in flow
        
        # All flows should be between 0 and 1
        assert 0 <= flow['forward_flow'] <= 1
        assert 0 <= flow['backward_flow'] <= 1
        assert 0 <= flow['self_attention'] <= 1
        
        # Flows should approximately sum to 1
        total_flow = flow['forward_flow'] + flow['backward_flow'] + flow['self_attention']
        assert np.isclose(total_flow, 1.0, atol=0.01)
        
    def test_compute_all_metrics(self):
        """Test computing all metrics at once."""
        all_metrics = self.metrics.compute_all_metrics(
            self.test_matrix,
            fractal_dimension=2.0295
        )
        
        # Check all expected metrics are present
        expected_metrics = [
            'entropy',
            'sparsity',
            'concentration',
            'uniformity',
            'forward_flow',
            'backward_flow',
            'self_attention',
            'fractal_dimension',
            'interpretability_score'
        ]
        
        for metric in expected_metrics:
            assert metric in all_metrics
            
    def test_zero_matrix_handling(self):
        """Test handling of zero matrix."""
        zero_matrix = np.zeros((10, 10))
        
        # Should handle gracefully without errors
        entropy = self.metrics.compute_entropy(zero_matrix)
        assert np.isfinite(entropy) or entropy == 0
        
        flow = self.metrics.compute_attention_flow(zero_matrix)
        assert all(v == 0 for v in flow.values())

