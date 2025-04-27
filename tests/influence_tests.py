import os
import ray
import json
import numpy as np
import torch
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

import sys
import os
sys.path.append(os.getcwd())
from env_module.flag_frenzy_env import FlagFrenzyEnv
from models.model import compute_input_influence, FlagFrenzyModel
from config.constants import ACTION_NAMES
from analysis.visualization import generate_influence_report, save_influence_analysis
from utils import initialize_ray, load_model, ensure_directory, tensor_to_numpy

def input_influence_test(checkpoint_path, output_dir="influence_analysis", show_plots=False, json_output=True, num_steps=1000):
    """
    Test the input_influence function with a trained model over multiple environment steps
    
    Args:
        checkpoint_path: Path to the checkpoint to load
        output_dir: Directory to save the visualizations
        show_plots: Whether to display the plots interactively
        json_output: Whether to output the influence analysis in JSON format
        num_steps: Number of steps to run the environment for
    """
    # Initialize Ray and register components
    initialize_ray()
    
    try:
        # Create output directory
        ensure_directory(output_dir)
        
        # Load the model from checkpoint
        model = load_model(checkpoint_path)
        print(f"Successfully loaded model from {checkpoint_path}")
        
        # Create environment and get initial observation
        env = FlagFrenzyEnv()
        obs, info = env.reset()
        
        # Debug information about observation structure
        print("\nObservation structure:")
        for k, v in obs.items():
            if isinstance(v, dict):
                print(f"  {k}: dict with keys {list(v.keys())}")
            elif hasattr(v, 'shape'):
                print(f"  {k}: ndarray with shape {v.shape} and dtype {v.dtype}")
            else:
                print(f"  {k}: {type(v)}")
        
        # Create a dictionary to store all influence analysis results for all steps
        all_steps_influence = {}
        
        # Run for specified number of steps
        for step in range(num_steps):
            print(f"\n--- Step {step+1}/{num_steps} ---")
            step_dir = os.path.join(output_dir, f"step_{step+1}")
            ensure_directory(step_dir)
            
            # Create a dictionary to store influence results for this step
            step_influence_results = {}
            
            # Compute influence for each action
            for action_idx in range(4):
                print(f"Analyzing influence for action: {ACTION_NAMES[action_idx]}")
                
                # Using the integrated attribution analysis
                attributions = compute_input_influence(model, obs, action_idx)
                
                # Generate report and visualization for this action
                action_result = generate_influence_report(
                    attributions, 
                    action_idx,
                    step_dir
                )
                
                # Store the results for this action
                step_influence_results[ACTION_NAMES[action_idx]] = action_result
            
            # Store step results
            all_steps_influence[f"step_{step+1}"] = step_influence_results
            
            # Save the influence analysis results for this step to a JSON file
            if json_output:
                step_json_path = save_influence_analysis(step_influence_results, step_dir)
                print(f"Step {step+1} influence analysis saved to: {step_json_path}")
            
            # Use the model's forward pass to select the next action
            # Convert observation to tensor format with proper batch dimensions
            obs_dict = {}
            for k, v in obs.items():
                if isinstance(v, dict):
                    obs_dict[k] = {}
                    for kk, vv in v.items():
                        tensor_v = torch.FloatTensor(vv)
                        if tensor_v.dim() == 1:
                            tensor_v = tensor_v.unsqueeze(0)  # Add batch dim
                        obs_dict[k][kk] = tensor_v
                else:
                    tensor_v = torch.FloatTensor(v)
                    # Special handling for entities which needs 3D shape [batch, entities, features]
                    if k == "entities" and tensor_v.dim() == 2:
                        tensor_v = tensor_v.unsqueeze(0)  # [entities, features] -> [1, entities, features]
                    elif tensor_v.dim() == 1:
                        tensor_v = tensor_v.unsqueeze(0)  # Add batch dim
                    obs_dict[k] = tensor_v
            
            # Pass through model
            input_dict = {"obs": obs_dict}
            
            try:
                # Debug entity shape
                if 'entities' in obs_dict:
                    print(f"Entities shape: {obs_dict['entities'].shape}")
                
                action_output, _ = model(input_dict, [], None)
                
                # The first 4 values correspond to the action type logits
                action_logits = action_output[0, :4]  # Add batch dimension index
                
                # Apply softmax to get proper probabilities
                action_probs = torch.softmax(action_logits, dim=0)
                action_type = torch.argmax(action_probs).item()
                
                print(f"Selecting action: {ACTION_NAMES[action_type]}")
                
                # Get the 10 action parameters directly (no mean/std split)
                action_params = action_output[0, 4:14].detach().cpu().numpy()
                
                # Print debug info about environment state before action
                b1 = env.find_entity_by_name("B1")
                flagship = env.find_entity_by_name("Renhai")
                
                # Create action dict
                action = {
                    "action_type": action_type,
                    "params": action_params.astype(np.float32)
                }
                
                # Execute the action
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
                
            except Exception as e:
                print(f"Error executing action: {e}")
                # Try to get a new observation to continue
                obs, info = env.reset()
                terminated, truncated = False, False
            
            # Break if episode is done
            if terminated or truncated:
                print(f"Episode ended at step {step+1}")
                break
        
        # Save overall influence analysis to a JSON file if requested
        if json_output:
            overall_json_path = save_influence_analysis(all_steps_influence, output_dir)
            print(f"\nComplete influence analysis for all steps saved to: {overall_json_path}")
        
        env.close()
        print("\nSuccessfully analyzed input influences across multiple steps!")
        print(f"All visualizations and data saved to: {os.path.abspath(output_dir)}/")
        
    except Exception as e:
        print(f"There was an error analyzing input influences: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run input influence test
    print("Running input influence test...")
    input_influence_test(
        "ray_results/rewardshape_flag_frenzy_ppo/PPO_FlagFrenzyEnv-v0_e73c2_00000_0_2025-04-19_22-48-13/checkpoint_000029", 
        output_dir="influence_analysis",
        show_plots=False,  # Set to True if you want interactive plots
        json_output=True,  # Output the influence analysis in JSON format
        num_steps=1000     # Number of steps to run
    )
    print("Input influence test completed.")