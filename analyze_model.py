# -----------------------------------------------------------
# Analyze model input influences for Flag Frenzy
#
# This script loads a trained model and analyzes how different inputs 
# affect the model's action decisions using gradient-based attribution
# -----------------------------------------------------------
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from models.model import FlagFrenzyModel, compute_input_influence
from env.flag_frenzy_env import FlagFrenzyEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from register_env import env_creator
from models.hybrid_action_dist import HybridActionDistribution

# Register the custom model and action distribution
ModelCatalog.register_custom_model("flag_frenzy_model", FlagFrenzyModel)
ModelCatalog.register_custom_action_dist("hybrid_action_dist", HybridActionDistribution)
register_env("FlagFrenzyEnv-v0", env_creator)

# Initialize Ray
ray.init(ignore_reinit_error=True)

def load_model(checkpoint_path):
    """
    Load a trained model from a checkpoint
    """
    algorithm = Algorithm.from_checkpoint(checkpoint_path)
    model = algorithm.get_policy().model
    return model

def visualize_attributions(attributions, title, save_path=None):
    """
    Visualize the input attributions as bar charts or heatmaps
    """
    plt.figure(figsize=(14, 8))
    
    # Entity influence
    if "entities" in attributions:
        plt.subplot(2, 2, 1)
        entity_scores = attributions["entities"].detach().cpu().numpy()
        plt.bar(range(len(entity_scores)), entity_scores)
        plt.title("Entity Influence")
        plt.xlabel("Entity Index")
        plt.ylabel("Attribution Score")
    
    # Visibility influence
    if "visibility" in attributions:
        plt.subplot(2, 2, 2)
        vis_scores = [
            attributions["visibility"]["legacy"].detach().cpu().numpy(),
            attributions["visibility"]["dynasty"].detach().cpu().numpy()
        ]
        plt.bar(["Legacy", "Dynasty"], vis_scores)
        plt.title("Visibility Influence")
        plt.ylabel("Attribution Score")
    
    # Mission influence
    if "mission" in attributions:
        plt.subplot(2, 2, 3)
        mission_scores = attributions["mission"].detach().cpu().numpy()
        plt.bar(range(len(mission_scores)), mission_scores)
        plt.title("Mission Status Influence")
        plt.xlabel("Mission Feature Index")
        plt.ylabel("Attribution Score")
    
    # Controllable entities influence
    if "controllable_entities" in attributions:
        plt.subplot(2, 2, 4)
        ctrl_scores = attributions["controllable_entities"].detach().cpu().numpy()
        plt.imshow(ctrl_scores.reshape(1, -1), aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title("Controllable Entities Influence")
        plt.xlabel("Entity Index")
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.9)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def analyze_model_influences(checkpoint_path, num_episodes=5):
    """
    Analyze a trained model's input influences across multiple episodes
    """
    # Load model
    model = load_model(checkpoint_path)
    print(f"Loaded model: {model}")
    
    # Create environment
    env = FlagFrenzyEnv()
    
    action_names = ["No-Op", "Move", "Return to Base", "Engage Target"]
    
    for episode in range(num_episodes):
        print(f"\nRunning analysis for episode {episode+1}/{num_episodes}")
        obs, info = env.reset()
        
        episode_done = False
        step = 0
        
        while not episode_done and step < 20:  # Limit to 20 steps for analysis
            # Get model's action
            obs_dict = {k: torch.FloatTensor(v) if not isinstance(v, dict) 
                       else {kk: torch.FloatTensor(vv) for kk, vv in v.items()} 
                       for k, v in obs.items()}
            
            input_dict = {"obs": obs_dict}
            action_logits, _ = model(input_dict, [], None)
            
            # Analyze each action type
            for action_idx in range(4):
                print(f"\nAnalyzing influence for action: {action_names[action_idx]} (Step {step})")
                attributions = compute_input_influence(model, obs, action_idx)
                
                # Visualize
                title = f"Input Influence on {action_names[action_idx]} (Episode {episode+1}, Step {step})"
                save_path = f"influence_ep{episode+1}_step{step}_action{action_idx}.png"
                visualize_attributions(attributions, title, save_path)
            
            # Take a random action to proceed
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_done = terminated or truncated
            step += 1
        
        print(f"Episode {episode+1} completed after {step} steps")
    
    env.close()

if __name__ == "__main__":
    # Check if a checkpoint path was provided
    import argparse
    parser = argparse.ArgumentParser(description='Analyze model input influences')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to the checkpoint file')
    parser.add_argument('--episodes', type=int, default=2,
                        help='Number of episodes to analyze')
    
    args = parser.parse_args()
    
    print(f"Analyzing model from checkpoint: {args.checkpoint}")
    analyze_model_influences(args.checkpoint, args.episodes) 