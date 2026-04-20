"""
Model Comparison Tool

Compare inference metrics across different SAC model variants.
Analyzes results JSON files and generates comparison statistics.

Usage:
    python compare_models.py              # Compare all recent results
    python compare_models.py --models 5cnn 2stt  # Compare specific models
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


class ModelComparator:
    """Compare trained model performance"""
    
    def __init__(self, results_dir: str = "."):
        self.results_dir = results_dir
        self.results = {}  # Dict[model_name] = list of episode results
    
    def load_results(self, model_names: List[str] = None):
        """Load result JSON files for models"""
        if model_names is None:
            # Auto-discover all result files
            result_files = Path(self.results_dir).glob("results_*.json")
            model_names = []
            for rf in result_files:
                # Extract model name from filename: results_5cnn_3eps.json
                parts = rf.stem.split("_")
                if len(parts) >= 2:
                    model_name = parts[1]
                    if model_name not in model_names:
                        model_names.append(model_name)
        
        for model_name in model_names:
            # Try to find most recent results file for this model
            result_files = sorted(
                Path(self.results_dir).glob(f"results_{model_name}_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            if result_files:
                result_file = result_files[0]
                print(f"[INFO] Loading {model_name} from: {result_file}")
                try:
                    with open(result_file, 'r') as f:
                        self.results[model_name] = json.load(f)
                except Exception as e:
                    print(f"[ERROR] Failed to load {result_file}: {e}")
    
    def compute_stats(self) -> Dict[str, Dict]:
        """Compute statistics for each model"""
        stats = {}
        
        for model_name, episodes in self.results.items():
            if not episodes:
                continue
            
            rewards = [ep['total_reward'] for ep in episodes]
            lengths = [ep['episode_length'] for ep in episodes]
            collisions = [ep.get('collisions', 0) for ep in episodes]
            waypoints = [ep.get('waypoints_crossed', 0) for ep in episodes]
            cbf_corr = [ep.get('cbf_corrections', 0) for ep in episodes]
            
            stats[model_name] = {
                'num_episodes': len(episodes),
                'reward': {
                    'mean': np.mean(rewards),
                    'std': np.std(rewards),
                    'min': np.min(rewards),
                    'max': np.max(rewards),
                    'total': np.sum(rewards),
                },
                'length': {
                    'mean': np.mean(lengths),
                    'std': np.std(lengths),
                    'min': np.min(lengths),
                    'max': np.max(lengths),
                    'total': np.sum(lengths),
                },
                'collisions': {
                    'total': np.sum(collisions),
                    'mean': np.mean(collisions),
                    'max': np.max(collisions),
                },
                'waypoints': {
                    'total': np.sum(waypoints),
                    'mean': np.mean(waypoints),
                    'max': np.max(waypoints),
                },
                'cbf_corrections': {
                    'total': np.sum(cbf_corr),
                    'mean': np.mean(cbf_corr),
                    'max': np.max(cbf_corr),
                } if any(cbf_corr) else None,
            }
        
        return stats
    
    def print_comparison_table(self, stats: Dict[str, Dict]):
        """Print comparison table"""
        if not stats:
            print("[ERROR] No statistics to compare")
            return
        
        models = list(stats.keys())
        
        print("\n" + "="*120)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*120)
        
        # Reward comparison
        print("\nREWARD METRICS (higher is better)")
        print("-" * 120)
        print(f"{'Model':<10} | {'Mean':<12} | {'Std':<12} | {'Min':<12} | {'Max':<12} | {'Total':<12}")
        print("-" * 120)
        for model in models:
            r = stats[model]['reward']
            print(f"{model:<10} | {r['mean']:<12.2f} | {r['std']:<12.2f} | {r['min']:<12.2f} | {r['max']:<12.2f} | {r['total']:<12.2f}")
        
        # Length comparison
        print("\nEPISODE LENGTH (steps)")
        print("-" * 120)
        print(f"{'Model':<10} | {'Mean':<12} | {'Std':<12} | {'Min':<12} | {'Max':<12}")
        print("-" * 120)
        for model in models:
            l = stats[model]['length']
            print(f"{model:<10} | {l['mean']:<12.2f} | {l['std']:<12.2f} | {l['min']:<12.0f} | {l['max']:<12.0f}")
        
        # Safety comparison
        print("\nSAFETY METRICS (lower is better)")
        print("-" * 120)
        print(f"{'Model':<10} | {'Total Collisions':<16} | {'Mean Collisions':<16} | {'CBF Corrections':<16}")
        print("-" * 120)
        for model in models:
            c = stats[model]['collisions']
            cbf = stats[model]['cbf_corrections']
            cbf_str = f"{cbf['mean']:.2f}" if cbf else "N/A"
            print(f"{model:<10} | {c['total']:<16.0f} | {c['mean']:<16.2f} | {cbf_str:<16}")
        
        # Navigation comparison
        print("\nNAVIGATION METRICS")
        print("-" * 120)
        print(f"{'Model':<10} | {'Total Waypoints':<16} | {'Mean Waypoints':<16} | {'Episodes':<10}")
        print("-" * 120)
        for model in models:
            w = stats[model]['waypoints']
            n = stats[model]['num_episodes']
            print(f"{model:<10} | {w['total']:<16.0f} | {w['mean']:<16.2f} | {n:<10}")
        
        print("="*120 + "\n")
    
    def print_rankings(self, stats: Dict[str, Dict]):
        """Print model rankings by key metrics"""
        if not stats:
            return
        
        models = list(stats.keys())
        
        print("\n" + "="*80)
        print("MODEL RANKINGS")
        print("="*80)
        
        # Reward ranking
        print("\n1. REWARD PERFORMANCE (Mean Reward - Higher is Better)")
        print("-" * 80)
        reward_ranking = sorted(
            models,
            key=lambda m: stats[m]['reward']['mean'],
            reverse=True
        )
        for rank, model in enumerate(reward_ranking, 1):
            r = stats[model]['reward']['mean']
            print(f"  {rank}. {model:<8} | Mean Reward: {r:>10.2f}")
        
        # Safety ranking
        print("\n2. SAFETY RECORD (Total Collisions - Lower is Better)")
        print("-" * 80)
        collision_ranking = sorted(
            models,
            key=lambda m: stats[m]['collisions']['total']
        )
        for rank, model in enumerate(collision_ranking, 1):
            c = stats[model]['collisions']['total']
            print(f"  {rank}. {model:<8} | Total Collisions: {c:>5.0f}")
        
        # Efficiency ranking (reward per step)
        print("\n3. EFFICIENCY (Reward per Step - Higher is Better)")
        print("-" * 80)
        efficiency = {}
        for model in models:
            s = stats[model]
            if s['length']['mean'] > 0:
                efficiency[model] = s['reward']['mean'] / s['length']['mean']
        
        eff_ranking = sorted(efficiency.items(), key=lambda x: x[1], reverse=True)
        for rank, (model, eff) in enumerate(eff_ranking, 1):
            print(f"  {rank}. {model:<8} | Reward/Step: {eff:>8.4f}")
        
        # Navigation ranking
        print("\n4. NAVIGATION (Waypoints Crossed - Higher is Better)")
        print("-" * 80)
        waypoint_ranking = sorted(
            models,
            key=lambda m: stats[m]['waypoints']['mean'],
            reverse=True
        )
        for rank, model in enumerate(waypoint_ranking, 1):
            w = stats[model]['waypoints']['mean']
            print(f"  {rank}. {model:<8} | Mean Waypoints: {w:>8.2f}")
        
        print("="*80 + "\n")
    
    def export_summary(self, filename: str = "model_comparison_summary.json"):
        """Export comparison summary to JSON"""
        stats = self.compute_stats()
        
        summary = {
            'timestamp': str(np.datetime64('now')),
            'models_compared': list(stats.keys()),
            'detailed_stats': stats,
            'best_performers': {
                'reward': sorted(
                    stats.items(),
                    key=lambda x: x[1]['reward']['mean'],
                    reverse=True
                )[0][0] if stats else None,
                'safety': sorted(
                    stats.items(),
                    key=lambda x: x[1]['collisions']['total']
                )[0][0] if stats else None,
                'navigation': sorted(
                    stats.items(),
                    key=lambda x: x[1]['waypoints']['mean'],
                    reverse=True
                )[0][0] if stats else None,
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[OK] Comparison summary exported to: {filename}\n")
        return summary


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Compare SAC model inference results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  Compare all recent results:
    python compare_models.py

  Compare specific models:
    python compare_models.py --models 5cnn 2stt

  Compare with custom results directory:
    python compare_models.py --results-dir ./demo

  Export comparison:
    python compare_models.py --export my_comparison.json
        """
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        choices=["5cnn", "2cnn", "5stt", "2stt"],
        help="Specific models to compare (default: all available)"
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default=".",
        help="Directory containing result JSON files (default: current dir)"
    )
    
    parser.add_argument(
        "--export",
        type=str,
        help="Export summary to JSON file"
    )
    
    args = parser.parse_args()
    
    # Initialize comparator
    comparator = ModelComparator(results_dir=args.results_dir)
    
    # Load results
    print("\n" + "="*80)
    print("LOADING MODEL RESULTS")
    print("="*80 + "\n")
    
    comparator.load_results(model_names=args.models)
    
    if not comparator.results:
        print("[ERROR] No result files found")
        print(f"Looking in: {args.results_dir}")
        print("Expected format: results_{MODEL}_{N}eps.json")
        return
    
    # Compute statistics
    print("\n[INFO] Computing statistics...")
    stats = comparator.compute_stats()
    
    # Print results
    comparator.print_comparison_table(stats)
    comparator.print_rankings(stats)
    
    # Export if requested
    if args.export:
        comparator.export_summary(filename=args.export)
    else:
        comparator.export_summary(filename="model_comparison_summary.json")


if __name__ == "__main__":
    main()
