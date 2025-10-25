"""
Advanced usage example for Fractal-Attention Analysis.

This script demonstrates advanced features including:
- Batch processing
- Model comparison
- Visualization generation
- Multi-layer analysis
"""

from fractal_attention_analysis import FractalAttentionAnalyzer
from pathlib import Path


def batch_analysis_example():
    """Demonstrate batch processing."""
    print("\n" + "="*60)
    print("BATCH ANALYSIS EXAMPLE")
    print("="*60)
    
    analyzer = FractalAttentionAnalyzer("gpt2")
    
    texts = [
        "Artificial intelligence is transforming technology.",
        "The fractal dimension reveals hidden patterns.",
        "Attention mechanisms are crucial for transformers."
    ]
    
    print("\nAnalyzing multiple texts...")
    results = analyzer.analyze_batch(texts)
    
    print("\nResults:")
    for i, result in enumerate(results, 1):
        print(f"\nText {i}: {texts[i-1][:40]}...")
        print(f"  Fractal Dimension: {result['fractal_dimension']:.6f}")
        print(f"  Entropy: {result['metrics']['entropy']:.6f}")
        print(f"  Sparsity: {result['metrics']['sparsity']:.6f}")


def model_comparison_example():
    """Demonstrate model comparison."""
    print("\n" + "="*60)
    print("MODEL COMPARISON EXAMPLE")
    print("="*60)
    
    analyzer = FractalAttentionAnalyzer("gpt2")
    
    text = "Compare attention patterns across different models."
    
    print(f"\nComparing GPT-2 vs DistilGPT-2...")
    print(f"Text: {text}")
    
    comparison = analyzer.compare_models("distilgpt2", text)
    
    print("\nComparison Results:")
    print(f"  Model 1 ({comparison['model_1']['name']}):")
    print(f"    Fractal Dimension: {comparison['model_1']['fractal_dimension']:.6f}")
    
    print(f"\n  Model 2 ({comparison['model_2']['name']}):")
    print(f"    Fractal Dimension: {comparison['model_2']['fractal_dimension']:.6f}")
    
    print(f"\n  Dimension Difference: {comparison['dimension_difference']:.6f}")


def visualization_example():
    """Demonstrate visualization generation."""
    print("\n" + "="*60)
    print("VISUALIZATION EXAMPLE")
    print("="*60)
    
    analyzer = FractalAttentionAnalyzer("gpt2")
    
    text = "Visualize the attention patterns in this sentence."
    
    output_dir = Path("./output/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating visualizations...")
    print(f"Text: {text}")
    
    results = analyzer.analyze(
        text,
        return_visualizations=True,
        save_dir=output_dir
    )
    
    print(f"\nVisualizations saved to: {output_dir}")
    print(f"  - attention_matrix.png")
    print(f"  - fractal_comparison.png")
    print(f"  - token_importance.png")


def multi_layer_analysis_example():
    """Demonstrate multi-layer analysis."""
    print("\n" + "="*60)
    print("MULTI-LAYER ANALYSIS EXAMPLE")
    print("="*60)
    
    analyzer = FractalAttentionAnalyzer("gpt2")
    
    text = "Analyze attention across different transformer layers."
    
    print(f"\nAnalyzing multiple layers...")
    print(f"Text: {text}")
    
    # Analyze first, middle, and last layers
    layers = [0, 6, -1]  # GPT-2 has 12 layers
    
    print("\nResults by layer:")
    for layer_idx in layers:
        result = analyzer.analyze(text, layer_idx=layer_idx)
        layer_name = f"Layer {layer_idx}" if layer_idx >= 0 else "Last Layer"
        print(f"\n  {layer_name}:")
        print(f"    Fractal Dimension: {result['fractal_dimension']:.6f}")
        print(f"    Entropy: {result['metrics']['entropy']:.6f}")


def main():
    """Run all advanced examples."""
    print("="*60)
    print("FRACTAL-ATTENTION ANALYSIS - ADVANCED EXAMPLES")
    print("="*60)
    
    try:
        batch_analysis_example()
        model_comparison_example()
        visualization_example()
        multi_layer_analysis_example()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Note: Some examples may require additional models to be downloaded.")


if __name__ == "__main__":
    main()

