"""
Command-line interface for Fractal-Attention Analysis.

This module provides a CLI for running FAA analysis from the command line.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .core import FractalAttentionAnalyzer
from .utils import DeviceManager


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fractal-Attention Analysis (FAA) for LLM Interpretability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze text with GPT-2
  faa-analyze --model gpt2 --text "Hello world"
  
  # Analyze with custom model and save visualizations
  faa-analyze --model meta-llama/Llama-3.2-1B --text "AI is transforming the world" --save-viz ./output
  
  # Analyze from file
  faa-analyze --model gpt2 --input-file input.txt --output results.json
  
  # Compare two models
  faa-compare --model1 gpt2 --model2 distilgpt2 --text "Test sentence"
        """
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model identifier (e.g., 'gpt2', 'meta-llama/Llama-3.2-1B')"
    )
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text",
        type=str,
        help="Input text to analyze"
    )
    input_group.add_argument(
        "--input-file",
        type=Path,
        help="Path to input text file"
    )
    
    # Analysis options
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Transformer layer index to analyze (default: -1, last layer)"
    )
    parser.add_argument(
        "--head",
        type=int,
        default=0,
        help="Attention head index to analyze (default: 0)"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save analysis results (JSON format)"
    )
    parser.add_argument(
        "--save-viz",
        type=Path,
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--format",
        choices=['json', 'csv', 'npz'],
        default='json',
        help="Output format (default: json)"
    )
    
    # Device options
    parser.add_argument(
        "--device",
        choices=['auto', 'cuda', 'cpu'],
        default='auto',
        help="Device to use for inference (default: auto)"
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=int,
        default=20,
        help="Maximum GPU memory in GB (default: 20)"
    )
    
    # Display options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all output except errors"
    )
    
    args = parser.parse_args()
    
    try:
        run_analysis(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def run_analysis(args):
    """Run FAA analysis with given arguments."""
    # Read input text
    if args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
    else:
        text = args.text
    
    if not text:
        raise ValueError("Input text is empty")
    
    # Setup device manager
    prefer_gpu = args.device != 'cpu'
    device_manager = DeviceManager(
        prefer_gpu=prefer_gpu,
        max_gpu_memory_gb=args.max_gpu_memory
    )
    
    # Initialize analyzer
    if not args.quiet:
        print(f"Initializing Fractal-Attention Analyzer for model: {args.model}")
    
    analyzer = FractalAttentionAnalyzer(
        model_name=args.model,
        device_manager=device_manager
    )
    
    # Run analysis
    if not args.quiet:
        print(f"Analyzing text: {text[:50]}{'...' if len(text) > 50 else ''}")
    
    results = analyzer.analyze(
        text=text,
        layer_idx=args.layer,
        head_idx=args.head,
        return_visualizations=args.save_viz is not None,
        save_dir=args.save_viz
    )
    
    # Display results
    if not args.quiet:
        print("\n" + "="*60)
        print("FRACTAL-ATTENTION ANALYSIS RESULTS")
        print("="*60)
        print(f"Model: {results['model_name']}")
        print(f"Fractal Dimension: {results['fractal_dimension']:.6f}")
        print(f"Analysis Time: {results['analysis_time']:.4f}s")
        print(f"\nMetrics:")
        for metric_name, value in results['metrics'].items():
            print(f"  {metric_name}: {value:.6f}")
        print("="*60)
    
    # Save results
    if args.output:
        if not args.quiet:
            print(f"\nSaving results to: {args.output}")
        analyzer.export_results(results, args.output, format=args.format)
    
    # Save visualizations
    if args.save_viz:
        if not args.quiet:
            print(f"Visualizations saved to: {args.save_viz}")
    
    return results


def compare_models():
    """CLI entry point for model comparison."""
    parser = argparse.ArgumentParser(
        description="Compare Fractal-Attention patterns between two models"
    )
    
    parser.add_argument("--model1", type=str, required=True, help="First model")
    parser.add_argument("--model2", type=str, required=True, help="Second model")
    parser.add_argument("--text", type=str, required=True, help="Input text")
    parser.add_argument("--output", type=Path, help="Output file for comparison results")
    
    args = parser.parse_args()
    
    try:
        # Initialize first analyzer
        print(f"Loading model 1: {args.model1}")
        analyzer1 = FractalAttentionAnalyzer(args.model1)
        
        # Compare
        print(f"Comparing with model 2: {args.model2}")
        comparison = analyzer1.compare_models(args.model2, args.text)
        
        # Display results
        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        print(f"Model 1: {comparison['model_1']['name']}")
        print(f"  Fractal Dimension: {comparison['model_1']['fractal_dimension']:.6f}")
        print(f"\nModel 2: {comparison['model_2']['name']}")
        print(f"  Fractal Dimension: {comparison['model_2']['fractal_dimension']:.6f}")
        print(f"\nDimension Difference: {comparison['dimension_difference']:.6f}")
        print("="*60)
        
        # Save if requested
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(comparison, f, indent=2, default=str)
            print(f"\nResults saved to: {args.output}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

