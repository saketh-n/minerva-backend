# -----------------------------------------------------------
# Analyze model input influences for Flag Frenzy
#
# This script loads a trained model and analyzes how different inputs 
# affect the model's action decisions using gradient-based attribution
# -----------------------------------------------------------
import os
import sys
import ray
import argparse

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from env_module.flag_frenzy_env import FlagFrenzyEnv
from models.model import FlagFrenzyModel
from models.hybrid_action_dist import HybridActionDistribution
from register_env import env_creator
from analysis.attribution import analyze_model_influences

def load_model(checkpoint_path):
    """
    Load a trained model from a checkpoint
    """
    algorithm = Algorithm.from_checkpoint(checkpoint_path)
    model = algorithm.get_policy().model
    return model

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze model input influences')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to the checkpoint file')
    parser.add_argument('--episodes', type=int, default=2,
                        help='Number of episodes to analyze')
    
    args = parser.parse_args()
    
    # Register the custom model and action distribution
    ModelCatalog.register_custom_model("flag_frenzy_model", FlagFrenzyModel)
    ModelCatalog.register_custom_action_dist("hybrid_action_dist", HybridActionDistribution)
    register_env("FlagFrenzyEnv-v0", env_creator)
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    print(f"Analyzing model from checkpoint: {args.checkpoint}")
    
    # Load model
    model = load_model(args.checkpoint)
    print(f"Loaded model: {model}")
    
    # Create environment
    env = FlagFrenzyEnv()
    
    # Run analysis
    analyze_model_influences(model, env, args.episodes)