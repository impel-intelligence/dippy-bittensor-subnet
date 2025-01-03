import numpy as np
from safetensors.torch import load_file
import torch
from pathlib import Path
import json

def analyze_model_differences(model1_path: str, model2_path: str, threshold: float = 1e-6):
    """
    Analyze differences between two safetensors models.
    
    Args:
        model1_path: Path to first model file
        model2_path: Path to second model file
        threshold: Minimum difference to consider significant (default: 1e-6)
        
    Returns:
        dict: Analysis results containing difference statistics
    """
    # Load both models
    model1 = load_file(model1_path)
    model2 = load_file(model2_path)
    
    # Compare keys
    keys1 = set(model1.keys())
    keys2 = set(model2.keys())
    
    shared_keys = keys1.intersection(keys2)
    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    
    differences = {}
    max_diff_key = None
    max_diff_value = 0
    total_params = 0
    different_params = 0
    
    # Analyze differences in shared keys
    for key in shared_keys:
        tensor1 = model1[key]
        tensor2 = model2[key]
        
        if tensor1.shape != tensor2.shape:
            differences[key] = {
                "type": "shape_mismatch",
                "shape1": tensor1.shape,
                "shape2": tensor2.shape
            }
            continue
            
        # Compare tensors
        diff = torch.abs(tensor1 - tensor2)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        # Count different parameters
        total_params += tensor1.numel()
        different_params += torch.sum(diff > threshold).item()
        
        if max_diff > threshold:
            differences[key] = {
                "type": "value_diff",
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "diff_params": torch.sum(diff > threshold).item(),
                "total_params": tensor1.numel()
            }
            
            if max_diff > max_diff_value:
                max_diff_value = max_diff
                max_diff_key = key
    
    return {
        "summary": {
            "total_keys": len(shared_keys),
            "keys_only_in_model1": list(only_in_1),
            "keys_only_in_model2": list(only_in_2),
            "total_parameters": total_params,
            "different_parameters": different_params,
            "different_parameter_percentage": (different_params / total_params * 100) if total_params > 0 else 0,
            "max_difference": max_diff_value,
            "max_difference_key": max_diff_key
        },
        "detailed_differences": differences
    }

def save_analysis(analysis_results: dict, output_path: str):
    """Save analysis results to a JSON file"""
    with open(output_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze differences between two safetensors models")
    parser.add_argument("model1", help="Path to first model file")
    parser.add_argument("model2", help="Path to second model file")
    parser.add_argument("--threshold", type=float, default=1e-6, help="Minimum difference threshold")
    parser.add_argument("--output", default="model_differences.json", help="Output JSON file path")
    
    args = parser.parse_args()
    
    results = analyze_model_differences(args.model1, args.model2, args.threshold)
    save_analysis(results, args.output)
    
    # Print summary
    summary = results["summary"]
    print(f"\nAnalysis Summary:")
    print(f"Total shared keys: {summary['total_keys']}")
    print(f"Keys only in model 1: {len(summary['keys_only_in_model1'])}")
    print(f"Keys only in model 2: {len(summary['keys_only_in_model2'])}")
    print(f"Total parameters: {summary['total_parameters']:,}")
    print(f"Different parameters: {summary['different_parameters']:,}")
    print(f"Percentage different: {summary['different_parameter_percentage']:.4f}%")
    print(f"Maximum difference: {summary['max_difference']}")
    print(f"Key with max difference: {summary['max_difference_key']}")
