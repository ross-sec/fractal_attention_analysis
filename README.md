# Fractal-Attention Analysis (FAA) Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/fractal-attention-analysis.svg)](https://badge.fury.io/py/fractal-attention-analysis)

A mathematical framework for analyzing transformer attention mechanisms using fractal geometry and golden ratio transformations. FAA provides deep insights into how Large Language Models (LLMs) process and attend to information.


## 🌟 Features

- **Universal LLM Support**: Works with any HuggingFace transformer model
- **Fractal Dimension Analysis**: Compute fractal dimensions of attention patterns
- **Golden Ratio Transformations**: Apply φ-based transformations for enhanced interpretability
- **Comprehensive Metrics**: Entropy, sparsity, concentration, and custom interpretability scores
- **Rich Visualizations**: Beautiful matplotlib-based attention pattern visualizations
- **CLI Interface**: Easy-to-use command-line tools
- **Modular Design**: Clean OOP architecture for easy extension
- **GPU Acceleration**: Efficient CUDA support with automatic memory management

## 📊 Key Findings

Our research demonstrates:
- **Universal Fractal Signature**: Consistent fractal dimension (≈2.0295) across diverse architectures (GPT-2, Qwen, Llama, Gemma)
- **Architectural Independence**: Fractal patterns persist despite model size and design differences
- **Real-time Analysis**: Sub-second performance for practical deployment

## 🚀 Quick Start

### Installation

```bash
pip install fractal-attention-analysis
```

### Basic Usage

```python
from fractal_attention_analysis import FractalAttentionAnalyzer

# Initialize analyzer with any HuggingFace model
analyzer = FractalAttentionAnalyzer("gpt2")

# Analyze text
results = analyzer.analyze("The golden ratio appears in nature and mathematics.")

# Access results
print(f"Fractal Dimension: {results['fractal_dimension']:.4f}")
print(f"Metrics: {results['metrics']}")
```

### Command Line Interface

```bash
# Analyze text with GPT-2
faa-analyze --model gpt2 --text "Hello world"

# Analyze with visualization
faa-analyze --model meta-llama/Llama-3.2-1B \
            --text "AI is transforming the world" \
            --save-viz ./output

# Compare two models
faa-compare --model1 gpt2 --model2 distilgpt2 \
            --text "Test sentence"
```

## 📚 Documentation

### Core Components

#### FractalAttentionAnalyzer

Main class for performing fractal-attention analysis:

```python
analyzer = FractalAttentionAnalyzer(
    model_name="gpt2",              # HuggingFace model ID
    device_manager=None,            # Optional custom device manager
    force_eager_attention=True,     # Force eager attention for compatibility
)

# Analyze text
results = analyzer.analyze(
    text="Your input text",
    layer_idx=-1,                   # Layer to analyze (-1 = last)
    head_idx=0,                     # Attention head index
    return_visualizations=True,     # Generate plots
    save_dir=Path("./output")       # Save visualizations
)
```

#### FractalTransforms

Fractal transformation and dimension calculations:

```python
from fractal_attention_analysis import FractalTransforms

transforms = FractalTransforms()

# Compute fractal dimension
dimension = transforms.compute_fractal_dimension(attention_matrix)

# Apply fractal interpolation
transformed = transforms.fractal_interpolation_function(attention_matrix)

# Golden ratio scoring
scored = transforms.golden_ratio_scoring(attention_matrix)
```

#### AttentionMetrics

Comprehensive attention metrics:

```python
from fractal_attention_analysis import AttentionMetrics

metrics = AttentionMetrics()

# Compute all metrics
all_metrics = metrics.compute_all_metrics(
    attention_matrix,
    fractal_dimension=2.0295
)

# Individual metrics
entropy = metrics.compute_entropy(attention_matrix)
sparsity = metrics.compute_sparsity(attention_matrix)
concentration = metrics.compute_concentration(attention_matrix)
```

#### AttentionVisualizer

Visualization utilities:

```python
from fractal_attention_analysis import AttentionVisualizer

visualizer = AttentionVisualizer()

# Plot attention matrix
fig = visualizer.plot_attention_matrix(
    attention_matrix,
    tokens=["Hello", "world"],
    title="Attention Pattern"
)

# Plot fractal comparison
fig = visualizer.plot_fractal_comparison(
    original_attention,
    transformed_attention
)
```

### Advanced Usage

#### Batch Analysis

```python
texts = [
    "First sentence to analyze.",
    "Second sentence to analyze.",
    "Third sentence to analyze."
]

results = analyzer.analyze_batch(texts)
```

#### Model Comparison

```python
comparison = analyzer.compare_models(
    other_model_name="distilgpt2",
    text="Compare attention patterns"
)

print(f"Dimension difference: {comparison['dimension_difference']:.4f}")
```

#### Export Results

```python
# Export as JSON
analyzer.export_results(results, "output.json", format='json')

# Export as CSV
analyzer.export_results(results, "output.csv", format='csv')

# Export as NumPy archive
analyzer.export_results(results, "output.npz", format='npz')
```

## 🔬 Mathematical Foundation

The FAA framework is based on:

1. **Golden Ratio (φ)**: Used for optimal attention partitioning
   ```
   φ = (1 + √5) / 2 ≈ 1.618
   ```

2. **Fractal Dimension**: Computed using box-counting method
   ```
   D = lim(ε→0) [log N(ε) / log(1/ε)]
   ```

3. **Fractal Interpolation**: Iterated Function System (IFS) transformations
   ```
   F(x) = Σ wᵢ · fᵢ(x)
   ```

4. **Neural Fractal Dimension**: Theoretical dimension for neural attention
   ```
   D_neural = φ² / 2 ≈ 1.309
   ```

## 📈 Performance

- **Analysis Time**: 0.047-0.248s depending on model size
- **Memory Efficient**: Supports models up to 1B parameters on 24GB GPU
- **Universal**: Works with GPT, BERT, T5, LLaMA, Qwen, Gemma, and more

## 🛠️ Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/ross-sec/fractal_attention_analysis.git
cd fractal-attention-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=fractal_attention_analysis --cov-report=html

# Run specific test file
pytest tests/test_core.py
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
flake8 src/ tests/

# Type check
mypy src/
```

## 📖 Citation

If you use FAA in your research, please cite:

```bibtex
@software{ross2025faa,
  title={Fractal-Attention Analysis: A Mathematical Framework for LLM Interpretability},
  author={Ross, Andre and Ross, Leorah and Atias, Eyal},
  year={2025},
  url={https://github.com/ross-sec/fractal_attention_analysis}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution

- Support for additional model architectures
- New fractal transformation methods
- Enhanced visualization capabilities
- Performance optimizations
- Documentation improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Andre Ross** - *Lead Developer* - [Ross Technologies](mailto:devops.ross@gmail.com)
- **Leorah Ross** - *Co-Developer* - [Ross Technologies](mailto:leorah@ross-developers.com)
- **Eyal Atias** - *Co-Developer* - [Hooking LTD](mailto:eyal@hooking.co.il)

## 🙏 Acknowledgments

- HuggingFace team for the Transformers library
- The open-source AI research community
- Fractal geometry pioneers: Benoit Mandelbrot, Michael Barnsley

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/ross-sec/fractal_attention_analysis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ross-sec/fractal_attention_analysis/discussions)
- **Email**: devops.ross@gmail.com

## 🗺️ Roadmap

- [ ] Support for multi-head parallel analysis
- [ ] CUDA-optimized fractal computations
- [ ] Real-time streaming analysis
- [ ] Interactive web dashboard
- [ ] Integration with popular interpretability tools (SHAP, LIME)
- [ ] Extended model zoo with pre-computed benchmarks

---

**Made with ❤️ by Ross Technologies & Hooking LTD**





## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=fractal_attention_analysis/fractal_attention_analysis&type=timeline&legend=top-left)](https://www.star-history.com/#fractal_attention_analysis/fractal_attention_analysis&type=timeline&legend=top-left)
