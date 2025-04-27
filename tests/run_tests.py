#!/usr/bin/env python
"""
Run script for Flag Frenzy tests with proper path setup
"""
import os
import sys
import argparse

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def parse_args():
    parser = argparse.ArgumentParser(description='Run Flag Frenzy environment tests')
    parser.add_argument('--test', type=str, choices=['all', 'sim', 'gym', 'influence', 'policy'], 
                        default='all', help='Which test to run')
    parser.add_argument('--checkpoint', type=str, 
                        default="ray_results/flag_frenzy_ppo/PPO_FlagFrenzyEnv-v0_26b0e_00000_0_2025-04-19_09-13-57/checkpoint_000029",
                        help='Path to the checkpoint file for model tests')
    parser.add_argument('--output-dir', type=str, default="influence_analysis",
                        help='Directory to save influence analysis outputs')
    parser.add_argument('--show-plots', action='store_true',
                        help='Show plots interactively during influence analysis')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.test in ['all', 'sim']:
        print("Running simulation test...")
        from tests.env_tests.basic_tests import sim_test
        sim_test()
        print("Simulation test completed.")

    if args.test in ['all', 'gym']:
        print("Running gymnasium test...")
        from tests.env_tests.basic_tests import gym_env_test
        gym_env_test()
        print("Gymnasium test completed.")
    
    if args.test in ['all', 'influence']:
        print("Running input influence test...")
        from tests.influence_tests import input_influence_test
        input_influence_test(
            args.checkpoint, 
            output_dir=args.output_dir,
            show_plots=args.show_plots,
            json_output=True
        )
        print("Input influence test completed.")
    
    if args.test in ['all', 'policy']:
        print("Running policy behavior test...")
        from tests.policy_tests import initialize_ray, policy_behavior_test
        initialize_ray()
        policy_behavior_test(
            args.checkpoint,
            num_steps=3
        )
        print("Policy behavior test completed.")