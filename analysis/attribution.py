import os
import torch
import numpy as np
from typing import Dict, Any, Optional, Union, List

from config.constants import ACTION_NAMES
from models.model import FlagFrenzyModel, compute_input_influence

def analyze_model_influences(model: FlagFrenzyModel, env, num_episodes: int = 5) -> None:
    """
    Analyze a trained model's input influences across multiple episodes
    
    Args:
        model: The trained FlagFrenzyModel
        env: The environment to run episodes in
        num_episodes: Number of episodes to analyze
    """
    for episode in range(num_episodes):
        print(f"\nRunning analysis for episode {episode+1}/{num_episodes}")
        obs, info = env.reset()
        
        episode_done = False
        step = 0
        
        while not episode_done and step < 20:  # Limit to 20 steps for analysis
            # Analyze each action type
            for action_idx in range(4):
                print(f"\nAnalyzing influence for action: {ACTION_NAMES[action_idx]} (Step {step})")
                attributions = compute_input_influence(model, obs, action_idx)
                
                # Visualize
                title = f"Input Influence on {ACTION_NAMES[action_idx]} (Episode {episode+1}, Step {step})"
                save_path = f"influence_ep{episode+1}_step{step}_action{action_idx}.png"
                visualize_attributions(attributions, title, save_path)
            
            # Take a random action to proceed
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_done = terminated or truncated
            step += 1
        
        print(f"Episode {episode+1} completed after {step} steps")
    
    env.close()

def to_scalar_or_numpy(x: Any) -> Union[float, np.ndarray]:
    """
    Safely convert tensor/array to numpy, keeping scalars as scalars
    
    Args:
        x: The value to convert
        
    Returns:
        Converted scalar or numpy array
    """
    if isinstance(x, torch.Tensor):
        # If tensor has only one element, return scalar, else return numpy array
        np_array = x.detach().cpu().numpy()
        return np_array.item() if np.size(np_array) == 1 else np_array
    elif isinstance(x, np.ndarray):
        # If numpy array has only one element, return scalar, else return array
        return x.item() if np.size(x) == 1 else x
    elif isinstance(x, (int, float, np.number)): # Check if it's already a scalar type
        return x
    else:
        # Try converting other types, return as numpy array if conversion works
        try:
            np_array = np.array(x)
            return np_array.item() if np.size(np_array) == 1 else np_array
        except:
            return x # Return original object if conversion fails

def visualize_attributions(attributions: Dict[str, Any], title: str, save_path: Optional[str] = None, show_plot: bool = False) -> None:
    """
    Visualize the input attributions as bar charts or heatmaps
    
    Args:
        attributions: Dictionary of attribution scores
        title: Title for the plot
        save_path: Path to save the figure
        show_plot: Whether to display the plot interactively
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(14, 8))
    
    # Debug information
    print(f"\nDebug - Attribution keys: {list(attributions.keys())}")
    has_data = False
    
    # Entity influence
    if "entities" in attributions and attributions["entities"] is not None:
        entity_scores = to_scalar_or_numpy(attributions["entities"])
        print(f"  Entity scores shape/type: {getattr(entity_scores, 'shape', type(entity_scores))}")
        
        if isinstance(entity_scores, np.ndarray) and hasattr(entity_scores, 'min') and hasattr(entity_scores, 'max'):
            print(f"  Entity scores min: {entity_scores.min():.6f}, max: {entity_scores.max():.6f}")
            
        has_data = True
        
        # Only plot if there are non-zero values and it's an array
        if isinstance(entity_scores, np.ndarray) and hasattr(entity_scores, 'max') and entity_scores.max() > 1e-10:
            plt.subplot(2, 2, 1)
            plt.bar(range(len(entity_scores)), entity_scores)
            plt.title(f"Entity Influence (max: {entity_scores.max():.6f})")
            plt.xlabel("Entity Index")
            plt.ylabel("Attribution Score")
        else:
            plt.subplot(2, 2, 1)
            plt.text(0.5, 0.5, "All values near zero", ha='center', va='center', transform=plt.gca().transAxes)
            plt.title("Entity Influence (all zero)")
    else:
        plt.subplot(2, 2, 1)
        plt.text(0.5, 0.5, "No entity data available", ha='center', va='center', transform=plt.gca().transAxes)
        plt.title("Entity Influence (missing data)")
    
    # Visibility influence
    if "visibility" in attributions:
        vis = attributions["visibility"]
        if isinstance(vis, dict) and "legacy" in vis and "dynasty" in vis:
            legacy_score = to_scalar_or_numpy(vis["legacy"])
            dynasty_score = to_scalar_or_numpy(vis["dynasty"])
            print(f"  Visibility scores - Legacy: {legacy_score}, Dynasty: {dynasty_score}")
            has_data = True
            
            plt.subplot(2, 2, 2)
            vis_scores = [legacy_score, dynasty_score]
            plt.bar(["Legacy", "Dynasty"], vis_scores)
            plt.title("Visibility Influence")
            plt.ylabel("Attribution Score")
        else:
            plt.subplot(2, 2, 2)
            plt.text(0.5, 0.5, "Visibility data not in expected format", ha='center', va='center', transform=plt.gca().transAxes)
            plt.title("Visibility Influence (wrong format)")
    else:
        plt.subplot(2, 2, 2)
        plt.text(0.5, 0.5, "No visibility data available", ha='center', va='center', transform=plt.gca().transAxes)
        plt.title("Visibility Influence (missing data)")
    
    # Mission influence
    if "mission" in attributions and attributions["mission"] is not None:
        mission_scores = to_scalar_or_numpy(attributions["mission"])
        print(f"  Mission scores shape/type: {getattr(mission_scores, 'shape', type(mission_scores))}")
        
        if isinstance(mission_scores, np.ndarray) and hasattr(mission_scores, 'min') and hasattr(mission_scores, 'max'):
            print(f"  Mission scores min: {mission_scores.min():.6f}, max: {mission_scores.max():.6f}")
            
        has_data = True
        
        # Only plot if there are non-zero values and it's an array
        if isinstance(mission_scores, np.ndarray) and hasattr(mission_scores, 'max') and mission_scores.max() > 1e-10:
            plt.subplot(2, 2, 3)
            plt.bar(range(len(mission_scores)), mission_scores)
            plt.title(f"Mission Status Influence (max: {mission_scores.max():.6f})")
            plt.xlabel("Mission Feature Index")
            plt.ylabel("Attribution Score")
        else:
            plt.subplot(2, 2, 3)
            plt.text(0.5, 0.5, "All values near zero", ha='center', va='center', transform=plt.gca().transAxes)
            plt.title("Mission Status Influence (all zero)")
    else:
        plt.subplot(2, 2, 3)
        plt.text(0.5, 0.5, "No mission data available", ha='center', va='center', transform=plt.gca().transAxes)
        plt.title("Mission Status Influence (missing data)")
    
    # Controllable entities influence
    if "controllable_entities" in attributions and attributions["controllable_entities"] is not None:
        ctrl_scores = to_scalar_or_numpy(attributions["controllable_entities"])
        print(f"  Controllable entity scores shape/type: {getattr(ctrl_scores, 'shape', type(ctrl_scores))}")
        
        if isinstance(ctrl_scores, np.ndarray) and hasattr(ctrl_scores, 'min') and hasattr(ctrl_scores, 'max'):
            print(f"  Controllable entity scores min: {ctrl_scores.min():.6f}, max: {ctrl_scores.max():.6f}")
            
        has_data = True
        
        # Only plot if there are non-zero values and it's an array
        if isinstance(ctrl_scores, np.ndarray) and hasattr(ctrl_scores, 'max') and ctrl_scores.max() > 1e-10:
            plt.subplot(2, 2, 4)
            plt.imshow(ctrl_scores.reshape(1, -1), aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.title(f"Controllable Entities (max: {ctrl_scores.max():.6f})")
            plt.xlabel("Entity Index")
        else:
            plt.subplot(2, 2, 4)
            plt.text(0.5, 0.5, "All values near zero", ha='center', va='center', transform=plt.gca().transAxes)
            plt.title("Controllable Entities (all zero)")
    else:
        plt.subplot(2, 2, 4)
        plt.text(0.5, 0.5, "No controllable entity data available", ha='center', va='center', transform=plt.gca().transAxes)
        plt.title("Controllable Entities (missing data)")
    
    # Display gradient sums if available
    if "gradient_sums" in attributions:
        print("\nGradient sums:")
        for k, v in attributions["gradient_sums"].items():
            print(f"  {k}: {v}")
    
    plt.tight_layout()
    plt.suptitle(f"{title} - Has Data: {has_data}", fontsize=16)
    plt.subplots_adjust(top=0.9)
    
    if not has_data:
        print(f"WARNING: No attribution data found for {title}")
    
    # Always save the figure
    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path)
        print(f"Figure saved to: {os.path.abspath(save_path)}")
    
    # Optionally show the plot
    if show_plot:
        plt.show()
    
    plt.close()

def make_json_serializable(obj: Any) -> Any:
    """
    Convert numpy arrays and other types to JSON-serializable objects
    
    Args:
        obj: The object to convert
        
    Returns:
        JSON-serializable version of the object
    """
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