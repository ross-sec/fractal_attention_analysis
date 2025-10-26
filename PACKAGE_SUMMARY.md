# Fractal-Attention Analysis Package - Complete Summary

## 📦 Package Information

- **Name**: `fractal-attention-analysis`
- **Version**: 0.1.0
- **License**: MIT
- **Python**: >=3.8
- **Repository**: https://github.com/ross-sec/fractal_attention_analysis

## 🏗️ Package Structure

```
faa-pip/
├── .github/
│   └── workflows/
│       ├── ci.yml                 # Continuous Integration
│       └── publish.yml            # PyPI Publishing
├── src/
│   └── fractal_attention_analysis/
│       ├── __init__.py           # Package initialization
│       ├── core.py               # FractalAttentionAnalyzer (main class)
│       ├── fractal.py            # FractalTransforms (fractal math)
│       ├── metrics.py            # AttentionMetrics (evaluation)
│       ├── visualization.py      # AttentionVisualizer (plotting)
│       ├── utils.py              # ModelLoader, DeviceManager
│       └── cli.py                # Command-line interface
├── tests/
│   ├── __init__.py
│   ├── test_fractal.py           # Fractal transforms tests
│   └── test_metrics.py           # Metrics tests
├── examples/
│   ├── basic_usage.py            # Basic example
│   └── advanced_analysis.py      # Advanced features
├── docs/                         # Documentation (placeholder)
├── pyproject.toml                # Package metadata & dependencies
├── README.md                     # Main documentation
├── CONTRIBUTING.md               # Contribution guidelines
├── CHANGELOG.md                  # Version history
├── LICENSE                       # MIT License
├── MANIFEST.in                   # Package file inclusion
└── .gitignore                    # Git ignore rules
```

## 🎯 Core Components

### 1. FractalAttentionAnalyzer (core.py)
**Main class for LLM interpretability analysis**

```python
from fractal_attention_analysis import FractalAttentionAnalyzer

analyzer = FractalAttentionAnalyzer("gpt2")
results = analyzer.analyze("Your text here")
```

**Key Methods**:
- `analyze()` - Analyze single text
- `analyze_batch()` - Batch processing
- `compare_models()` - Compare two models
- `export_results()` - Save results (JSON/CSV/NPZ)

### 2. FractalTransforms (fractal.py)
**Fractal mathematics and transformations**

**Key Methods**:
- `compute_fractal_dimension()` - Box-counting method
- `fractal_interpolation_function()` - IFS transformation
- `golden_ratio_partition()` - φ-based partitioning
- `golden_ratio_scoring()` - Importance scoring
- `multi_scale_analysis()` - Multi-scale processing
- `compute_self_similarity()` - Self-similarity measure

### 3. AttentionMetrics (metrics.py)
**Comprehensive attention evaluation**

**Key Methods**:
- `compute_entropy()` - Shannon entropy
- `compute_sparsity()` - Sparsity ratio
- `compute_concentration()` - Top-k concentration
- `compute_uniformity()` - Distribution uniformity
- `compute_attention_flow()` - Forward/backward/self flow
- `compute_interpretability_score()` - Overall score
- `compute_all_metrics()` - All metrics at once

### 4. AttentionVisualizer (visualization.py)
**Rich matplotlib-based visualizations**

**Key Methods**:
- `plot_attention_matrix()` - Heatmap visualization
- `plot_fractal_comparison()` - Before/after comparison
- `plot_metrics_comparison()` - Bar chart comparisons
- `plot_fractal_dimension_analysis()` - Box-counting plot
- `plot_multi_scale_analysis()` - Multi-scale grid
- `plot_token_importance()` - Token importance bars

### 5. Utilities (utils.py)
**Model loading and device management**

**Classes**:
- `ModelLoader` - HuggingFace model loading with optimizations
- `DeviceManager` - GPU/CPU allocation and memory management

### 6. CLI (cli.py)
**Command-line interface**

**Commands**:
- `faa-analyze` - Analyze text
- `faa-compare` - Compare models

## 📚 Dependencies

### Core Dependencies
- `torch >= 2.0.0` - PyTorch for model inference
- `transformers >= 4.30.0` - HuggingFace Transformers
- `numpy >= 1.24.0` - Numerical operations
- `scipy >= 1.10.0` - Scientific computing
- `matplotlib >= 3.7.0` - Plotting
- `seaborn >= 0.12.0` - Statistical visualization
- `pandas >= 2.0.0` - Data manipulation
- `scikit-learn >= 1.3.0` - ML utilities

### Optional Dependencies
- `[dev]` - Development tools (pytest, black, flake8, mypy)
- `[docs]` - Documentation tools (sphinx)
- `[gpu]` - GPU utilities (GPUtil)
- `[all]` - All optional dependencies
