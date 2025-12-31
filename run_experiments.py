"""
Unified runner for AutoPrompt and GCG experiments

Usage:
    python run_experiments.py --task classification --method autoprompt
    python run_experiments.py --task target_generation --method gcg
    python run_experiments.py --all
"""

import os
import sys
import argparse
import subprocess

def run_script(script_path, description):
    """Run a single script"""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Script: {script_path}")
    print(f"{'='*70}\n")

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            check=True
        )
        print(f"\n✓ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run adversarial attack experiments')
    parser.add_argument('--task', type=str, choices=['classification', 'target_generation', 'all'],
                        help='Task type')
    parser.add_argument('--method', type=str, choices=['autoprompt', 'gcg', 'all'],
                        help='Method')
    parser.add_argument('--all', action='store_true',
                        help='Run all experiments')

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    experiments = {
        'classification': {
            'autoprompt': {
                'script': os.path.join(base_dir, 'classification', 'autoprompt_classification.py'),
                'description': 'AutoPrompt Classification Attack'
            },
            'gcg': {
                'script': os.path.join(base_dir, 'classification', 'gcg_classification.py'),
                'description': 'GCG Classification Attack'
            }
        },
        'target_generation': {
            'autoprompt': {
                'script': os.path.join(base_dir, 'target_generation', 'autoprompt_generate.py'),
                'description': 'AutoPrompt Target Generation (idiot)'
            },
            'gcg': {
                'script': os.path.join(base_dir, 'target_generation', 'gcg_generate.py'),
                'description': 'GCG Target Generation (idiot)'
            }
        }
    }

    to_run = []

    if args.all:
        for task in experiments.values():
            for exp in task.values():
                to_run.append(exp)
    else:
        if not args.task:
            print("Error: specify --task or use --all")
            parser.print_help()
            return

        if args.method and args.method != 'all':
            exp = experiments[args.task][args.method]
            to_run.append(exp)
        else:
            for exp in experiments[args.task].values():
                to_run.append(exp)

    print(f"\nRunning {len(to_run)} experiments")

    results = []
    for exp in to_run:
        success = run_script(exp['script'], exp['description'])
        results.append((exp['description'], success))

    print(f"\n\n{'='*70}")
    print("Summary:")
    print(f"{'='*70}")
    for desc, success in results:
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {status}: {desc}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
