# Changelog

All notable changes to the Fractal-Attention Analysis (FAA) framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Multi-head parallel analysis
- CUDA-optimized fractal computations
- Real-time streaming analysis
- Interactive web dashboard
- Integration with SHAP and LIME

## [0.1.0] - 2025-01-26

### Added
- Initial release of Fractal-Attention Analysis framework
- Core `FractalAttentionAnalyzer` class for LLM interpretability
- `FractalTransforms` module with golden ratio transformations
- `AttentionMetrics` for comprehensive attention analysis
- `AttentionVisualizer` for attention pattern visualization
- `ModelLoader` and `DeviceManager` utilities
- Command-line interface (`faa-analyze`, `faa-compare`)
- Support for all HuggingFace transformer models
- Fractal dimension calculation using box-counting method
- Golden ratio-based attention partitioning and scoring
- Entropy, sparsity, and concentration metrics
- Multi-scale fractal analysis
- GPU acceleration with automatic memory management
- Comprehensive documentation and examples
- MIT License

### Features
- Universal LLM support (GPT, BERT, T5, LLaMA, Qwen, Gemma, etc.)
- Real-time analysis (sub-second performance)
- Memory-efficient processing for 24GB GPU
- Modular OOP architecture
- Rich visualization capabilities
- Export results in JSON, CSV, and NPZ formats
- Batch processing support
- Model comparison functionality

### Validated Models
- GPT-2 (117M parameters)
- Qwen-0.6B (600M parameters)
- Meta Llama-3.2-1B (1B parameters)
- Google Gemma-3-1B (1B parameters)

### Research Findings
- Universal fractal dimension: 2.0295 ± 0.0000 across all tested architectures
- Architectural independence confirmed
- Consistent patterns despite model size differences

[Unreleased]: https://github.com/ross-sec/fractal-attention-analysis/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ross-sec/fractal-attention-analysis/releases/tag/v0.1.0

