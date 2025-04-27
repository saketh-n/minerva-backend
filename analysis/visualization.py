import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Tuple

from analysis.attribution import to_scalar_or_numpy, make_json_serializable
from config.constants import ACTION_NAMES

def generate_influence_report(attributions: Dict[str, Any], action_idx: int, 
                             output_dir: str = "influence_analysis") -> Dict[str, Any]:
    """
    Generate a comprehensive influence analysis report for a specific action
    
    Args:
        attributions: The attribution scores dictionary
        action_idx: Index of the action to analyze
        output_dir: Directory to save the visualization
        
    Returns:
        Dictionary containing the summarized analysis
    """
    action_result = {}
    
    # Analyze entity influence
    if "entities" in attributions and attributions["entities"] is not None:
        entity_scores = to_scalar_or_numpy(attributions["entities"])
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
    
    # Analyze mission influence
    if "mission" in attributions and attributions["mission"] is not None:
        mission_scores = to_scalar_or_numpy(attributions["mission"])
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
        
    # Analyze visibility influence
    if "visibility" in attributions:
        vis = attributions["visibility"]
        if isinstance(vis, dict) and "legacy" in vis and "dynasty" in vis:
            legacy_score = to_scalar_or_numpy(vis["legacy"]).sum() if hasattr(vis["legacy"], "sum") else vis["legacy"]
            dynasty_score = to_scalar_or_numpy(vis["dynasty"]).sum() if hasattr(vis["dynasty"], "sum") else vis["dynasty"]
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
    
    # Identify top 3 features across all components
    all_features = []
    
    # Collect entities features
    if "entities" in action_result and "top_entity_scores" in action_result["entities"]:
        for i, (idx, score) in enumerate(zip(action_result["entities"]["top_entities"], 
                                          action_result["entities"]["top_entity_scores"])):
            if i >= 3:  # Only consider top 3
                break
            all_features.append((f"entity_{idx}", score, "entities"))
    
    # Collect mission features
    if "mission" in action_result and "top_feature_scores" in action_result["mission"]:
        for i, (idx, score) in enumerate(zip(action_result["mission"]["top_features"], 
                                          action_result["mission"]["top_feature_scores"])):
            if i >= 3:  # Only consider top 3
                break
            all_features.append((f"mission_{idx}", score, "mission"))
    
    # Collect visibility features
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
    
    # Create visualization
    title = f"Input Influence on {ACTION_NAMES[action_idx]}"
    save_path = os.path.join(output_dir, f"influence_action{action_idx}.png")
    
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize attribution
    from analysis.attribution import visualize_attributions
    visualize_attributions(attributions, title, save_path, show_plot=False)
    
    return action_result

def save_influence_analysis(influence_results: Dict[str, Any], output_dir: str = "influence_analysis") -> str:
    """
    Save influence analysis results to a JSON file
    
    Args:
        influence_results: The analysis results dictionary
        output_dir: Directory to save the JSON file
        
    Returns:
        Path to the saved JSON file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the influence analysis results to a JSON file
    json_path = os.path.join(output_dir, "influence_analysis.json")
    with open(json_path, 'w') as f:
        json.dump(influence_results, f, indent=2)
    
    return json_path