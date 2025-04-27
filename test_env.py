from env.flag_frenzy_env import FlagFrenzyEnv
from env.SimulationInterface import ControllableEntityManouver
import torch
import numpy as np
import matplotlib.pyplot as plt
import ray
import os
import json
from ray.rllib.algorithms.algorithm import Algorithm
from models.model import FlagFrenzyModel, compute_input_influence
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from register_env import env_creator
from models.hybrid_action_dist import HybridActionDistribution
from utils import visualize_attributions

def sim_test():
    env = FlagFrenzyEnv()

    try:
        while env.compute_reward() != 0:
            print("This is EXECUTED")
            env._tick()
            env._get_observations()

            # Example on how to perform an attack
            flagship = env.find_entity_by_name("Renhai")
            b1 = env.find_entity_by_name("B1")

            if (flagship.IsAlive()):
                if b1.CurrentManouver != ControllableEntityManouver.Combat:
                    env.execute_action([3, [b1.EntityId / env.max_entities, flagship.EntityId / env.max_entities, 0.0, 0.0]])
            elif b1.CurrentManouver == ControllableEntityManouver.NoManouver:
                env.execute_action([2, [b1.EntityId / env.max_entities]])
        print("Successfully ran simulation.")

    except Exception as e:
        print(f"There was error running the game simulation! {e}")

def gym_env_test():
    env = FlagFrenzyEnv()

    try:
        obs, info = env.reset()

        for i in range(5):
            action = env.action_space.sample()
            observation, reward, term, trunc, info = env.step(action)

        env.close()
        print("Successfully stepped through gymansium environment!")

    except Exception as e:
        print(f"There was an error running the game simulation! {e}")

def input_influence_test(checkpoint_path, output_dir="influence_analysis", show_plots=False, json_output=True):
    """
    Test the input_influence function with a trained model
    
    Args:
        checkpoint_path: Path to the checkpoint to load
        output_dir: Directory to save the visualizations
        show_plots: Whether to display the plots interactively
        json_output: Whether to output the influence analysis in JSON format
    """
    # Register the model and action distribution (required for loading checkpoint)
    ModelCatalog.register_custom_model("flag_frenzy_model", FlagFrenzyModel)
    ModelCatalog.register_custom_action_dist("hybrid_action_dist", HybridActionDistribution)
    register_env("FlagFrenzyEnv-v0", env_creator)
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the model from checkpoint
        algorithm = Algorithm.from_checkpoint(checkpoint_path)
        model = algorithm.get_policy().model
        print(f"Successfully loaded model from {checkpoint_path}")
        
        # Create environment and get observation
        env = FlagFrenzyEnv()
        obs, info = env.reset()
        
        # Debug information about observation structure
        print("\nObservation structure:")
        for k, v in obs.items():
            if isinstance(v, dict):
                print(f"  {k}: dict with keys {list(v.keys())}")
            elif isinstance(v, np.ndarray):
                print(f"  {k}: ndarray with shape {v.shape} and dtype {v.dtype}")
            else:
                print(f"  {k}: {type(v)}")
        
        # Convert observation to tensors with grad tracking
        print("\nComputing input influence...")
        
        # Analyze each action type
        action_names = ["No-Op", "Move", "Return to Base", "Engage Target"]
        
        print("\nInfluence analysis results will be saved to the following files:")
        
        # Create a dictionary to store all influence analysis results for JSON output
        influence_results = {}
        
        for action_idx in range(4):
            print(f"Analyzing influence for action: {action_names[action_idx]}")
            
            # Now using the integrated attribution analysis
            # This will automatically enable attribution analysis, run the forward pass,
            # and compute attributions in a single call
            attributions = compute_input_influence(model, obs, action_idx)
            
            # Visualize
            title = f"Input Influence on {action_names[action_idx]}"
            save_path = os.path.join(output_dir, f"influence_action{action_idx}.png")
            visualize_attributions(attributions, title, save_path, show_plots)
            
            # Helper function to safely convert tensor to numpy
            def to_numpy(x):
                if isinstance(x, torch.Tensor):
                    return x.detach().cpu().numpy()
                elif isinstance(x, np.ndarray):
                    return x
                else:
                    return np.array([x])
            
            # Helper function to make numpy arrays JSON serializable
            def make_json_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, (list, tuple)):
                    return [make_json_serializable(item) for item in obj]
                elif isinstance(obj, dict):
                    return {key: make_json_serializable(value) for key, value in obj.items()}
                else:
                    return obj
            
            # Extract and print attribution data for numerical analysis
            print(f"  Attribution summary for {action_names[action_idx]}:")
            
            # Initialize the results dictionary for this action
            action_result = {}
            
            if "entities" in attributions and attributions["entities"] is not None:
                entity_scores = to_numpy(attributions["entities"])
                top_entities = np.argsort(-entity_scores)[:5]  # Top 5 entities
                top_entity_scores = entity_scores[top_entities]
                print(f"  Top influential entities: {top_entities}")
                
                # Add to JSON results
                action_result["entities"] = {
                    "scores": make_json_serializable(entity_scores),
                    "top_entities": make_json_serializable(top_entities),
                    "top_entity_scores": make_json_serializable(top_entity_scores),
                    "top_3": {
                        "indices": make_json_serializable(top_entities[:3]),
                        "scores": make_json_serializable(top_entity_scores[:3])
                    }
                }
            
            if "mission" in attributions and attributions["mission"] is not None:
                mission_scores = to_numpy(attributions["mission"])
                top_mission = np.argsort(-mission_scores)[:3]  # Top 3 mission features
                top_mission_scores = mission_scores[top_mission]
                print(f"  Top influential mission features: {top_mission}")
                
                # Add to JSON results
                action_result["mission"] = {
                    "scores": make_json_serializable(mission_scores),
                    "top_features": make_json_serializable(top_mission),
                    "top_feature_scores": make_json_serializable(top_mission_scores),
                    "top_3": {
                        "indices": make_json_serializable(top_mission[:min(3, len(top_mission))]),
                        "scores": make_json_serializable(top_mission_scores[:min(3, len(top_mission))])
                    }
                }
                
            if "visibility" in attributions:
                vis = attributions["visibility"]
                if isinstance(vis, dict) and "legacy" in vis and "dynasty" in vis:
                    legacy_score = to_numpy(vis["legacy"]).sum() if hasattr(vis["legacy"], "sum") else vis["legacy"]
                    dynasty_score = to_numpy(vis["dynasty"]).sum() if hasattr(vis["dynasty"], "sum") else vis["dynasty"]
                    print(f"  Visibility influence: Legacy={legacy_score:.4f}, Dynasty={dynasty_score:.4f}")
                    
                    # Add to JSON results
                    vis_scores = [float(legacy_score) if isinstance(legacy_score, (np.number, np.ndarray)) else legacy_score,
                                float(dynasty_score) if isinstance(dynasty_score, (np.number, np.ndarray)) else dynasty_score]
                    vis_names = ["legacy", "dynasty"]
                    
                    # Sort by score to get top features
                    sorted_indices = np.argsort([-s if isinstance(s, (int, float)) else 0 for s in vis_scores])
                    top_vis_indices = sorted_indices[:min(3, len(sorted_indices))]
                    top_vis_names = [vis_names[i] for i in top_vis_indices]
                    top_vis_scores = [vis_scores[i] for i in top_vis_indices]
                    
                    action_result["visibility"] = {
                        "legacy": vis_scores[0],
                        "dynasty": vis_scores[1],
                        "top_3": {
                            "names": top_vis_names,
                            "scores": top_vis_scores
                        }
                    }
            
            # Add a summary of top 3 features across all components
            # First, collect all features with their scores
            all_features = []
            
            if "entities" in action_result and "top_entity_scores" in action_result["entities"]:
                for i, (idx, score) in enumerate(zip(action_result["entities"]["top_entities"], 
                                                  action_result["entities"]["top_entity_scores"])):
                    if i >= 3:  # Only consider top 3
                        break
                    all_features.append((f"entity_{idx}", score, "entities"))
            
            if "mission" in action_result and "top_feature_scores" in action_result["mission"]:
                for i, (idx, score) in enumerate(zip(action_result["mission"]["top_features"], 
                                                  action_result["mission"]["top_feature_scores"])):
                    if i >= 3:  # Only consider top 3
                        break
                    all_features.append((f"mission_{idx}", score, "mission"))
            
            if "visibility" in action_result and "top_3" in action_result["visibility"]:
                for i, (name, score) in enumerate(zip(action_result["visibility"]["top_3"]["names"], 
                                                   action_result["visibility"]["top_3"]["scores"])):
                    if i >= 3:  # Only consider top 3
                        break
                    all_features.append((f"visibility_{name}", score, "visibility"))
            
            # Sort by score and take top 3 overall
            all_features.sort(key=lambda x: -x[1] if isinstance(x[1], (int, float)) else 0)
            top_overall = all_features[:min(3, len(all_features))]
            
            # Add to results
            action_result["top_3_overall"] = {
                "features": [f[0] for f in top_overall],
                "scores": [f[1] for f in top_overall],
                "components": [f[2] for f in top_overall]
            }
            
            # Store the results for this action
            influence_results[action_names[action_idx]] = action_result
        
        env.close()
        print("\nSuccessfully analyzed input influences!")
        print(f"All visualizations saved to: {os.path.abspath(output_dir)}/")
        
        # Save the influence analysis results to a JSON file if requested
        if json_output:
            json_path = os.path.join(output_dir, "influence_analysis.json")
            with open(json_path, 'w') as f:
                json.dump(influence_results, f, indent=2)
            print(f"Influence analysis JSON saved to: {json_path}")
        
    except Exception as e:
        print(f"There was an error analyzing input influences: {e}")
        import traceback
        traceback.print_exc()

def policy_behavior_test(checkpoint_path, num_steps=5):
    """
    Test the model behavior with attribution analysis enabled
    
    Args:
        checkpoint_path: Path to the checkpoint to load
        num_steps: Number of steps to run
    """
    # Register the model and action distribution
    ModelCatalog.register_custom_model("flag_frenzy_model", FlagFrenzyModel)
    ModelCatalog.register_custom_action_dist("hybrid_action_dist", HybridActionDistribution)
    register_env("FlagFrenzyEnv-v0", env_creator)
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    try:
        # Load the model
        algorithm = Algorithm.from_checkpoint(checkpoint_path)
        model = algorithm.get_policy().model
        print(f"Successfully loaded model from {checkpoint_path}")
        
        # Create environment
        env = FlagFrenzyEnv()
        obs, info = env.reset()
        
        print(f"\nRunning {num_steps} steps with attribution analysis:")
        
        for step in range(num_steps):
            print(f"\nStep {step+1}/{num_steps}")
            
            # Enable attribution analysis for this forward pass
            if isinstance(model, FlagFrenzyModel):
                model.enable_attribution_analysis()
                
            # Pass observation through model
            obs_dict = {k: torch.FloatTensor(v) if not isinstance(v, dict) 
                      else {kk: torch.FloatTensor(vv) for kk, vv in v.items()} 
                      for k, v in obs.items()}
            
            input_dict = {"obs": obs_dict}
            action_output, _ = model(input_dict, [], None)
            
            # Get action type with highest probability
            logits = action_output[:4]
            action_probs = torch.softmax(logits, dim=0)
            action_type = torch.argmax(action_probs).item()
            
            print(f"Selected action: {['No-Op', 'Move', 'Return to Base', 'Engage Target'][action_type]}")
            print(f"Action probabilities: {action_probs.detach().numpy()}")
            
            # Get attributions that were computed during the forward pass
            if isinstance(model, FlagFrenzyModel):
                attributions = model.get_attributions()
                if attributions:
                    # Process and print attribution information
                    if "gradient_sums" in attributions:
                        print("Input component influence (gradient sums):")
                        for k, v in attributions["gradient_sums"].items():
                            component = k.replace("_grad_sum", "")
                            print(f"  {component}: {v:.4f}")
            
            # Take a random action to step the environment
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print("Episode ended early")
                break
        
        env.close()
        print("\nSuccessfully completed policy behavior test")
        
    except Exception as e:
        print(f"Error in policy behavior test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Running simulation test...")
    sim_test()
    print("Simulation test completed.")

    print("Running gymnasium test...")
    gym_env_test()
    print("Gymansium test completed.")
    
    # Run input influence test
    print("Running input influence test...")
    input_influence_test(
        "ray_results/flag_frenzy_ppo/PPO_FlagFrenzyEnv-v0_26b0e_00000_0_2025-04-19_09-13-57/checkpoint_000029", 
        output_dir="influence_analysis",
        show_plots=False,  # Set to True if you want interactive plots
        json_output=True   # Output the influence analysis in JSON format

    )
    print("Input influence test completed.")
    
    # # Run policy behavior test to demonstrate attribution during regular forward passes
    # print("Running policy behavior test...")
    # policy_behavior_test(
    #     "ray_results/flag_frenzy_ppo/PPO_FlagFrenzyEnv-v0_26b0e_00000_0_2025-04-19_09-13-57/checkpoint_000029",
    #     num_steps=3
    # )
    # print("Policy behavior test completed.")
