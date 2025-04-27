import torch
import numpy as np

import sys
import os
sys.path.append(os.getcwd())
from env_module.flag_frenzy_env import FlagFrenzyEnv
from models.model import FlagFrenzyModel
from config.constants import ACTION_NAMES
from utils import initialize_ray, load_model, tensor_to_numpy

def policy_behavior_test(checkpoint_path, num_steps=5):
    """
    Test the model behavior with attribution analysis enabled
    
    Args:
        checkpoint_path: Path to the checkpoint to load
        num_steps: Number of steps to run
    """
    try:
        # Load the model
        model = load_model(checkpoint_path)
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
            
            print(f"Selected action: {ACTION_NAMES[action_type]}")
            print(f"Action probabilities: {tensor_to_numpy(action_probs)}")
            
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
            
            # Take a random action to step the environment - this is already in the correct format
            action = env.action_space.sample()  # This already returns a dict with action_type and params
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
    initialize_ray()
    
    # Run policy behavior test to demonstrate attribution during regular forward passes
    print("Running policy behavior test...")
    policy_behavior_test(
        "ray_results/flag_frenzy_ppo/PPO_FlagFrenzyEnv-v0_26b0e_00000_0_2025-04-19_09-13-57/checkpoint_000029",
        num_steps=3
    )
    print("Policy behavior test completed.")