"""
Basic usage example for Fractal-Attention Analysis.

This script demonstrates the fundamental usage of the FAA framework.
"""

from fractal_attention_analysis import FractalAttentionAnalyzer
from pathlib import Path


def main():
    """Run basic FAA analysis example."""
    
    print("="*60)
    print("FRACTAL-ATTENTION ANALYSIS - BASIC EXAMPLE")
    print("="*60)
    
    # Initialize analyzer with GPT-2
    print("\n1. Initializing analyzer...")
    analyzer = FractalAttentionAnalyzer("gpt2")
    print(f"   Model loaded: {analyzer.model_name}")
    
    # Analyze a simple sentence
    print("\n2. Analyzing text...")
    text = "The golden ratio appears in nature and mathematics."
    results = analyzer.analyze(text)
    
    # Display results
    print("\n3. Results:")
    print(f"   Fractal Dimension: {results['fractal_dimension']:.6f}")
    print(f"   Analysis Time: {results['analysis_time']:.4f}s")
    print(f"   Number of Tokens: {len(results['tokens'])}")
    
    print("\n4. Metrics:")
    for metric_name, value in results['metrics'].items():
        print(f"   {metric_name}: {value:.6f}")
    
    # Export results
    print("\n5. Exporting results...")
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    analyzer.export_results(
        results,
        output_dir / "basic_results.json",
        format='json'
    )
    print(f"   Results saved to: {output_dir / 'basic_results.json'}")
    
    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()

