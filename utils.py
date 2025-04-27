"""
Utility functions for the Flag Frenzy project
"""
import os
import numpy as np
import torch
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from models.model import FlagFrenzyModel
from models.hybrid_action_dist import HybridActionDistribution
from register_env import env_creator
from config.constants import ACTION_NAMES

def initialize_ray():
    """
    Initialize Ray and register custom components
    """
    # Register the model and action distribution
    ModelCatalog.register_custom_model("flag_frenzy_model", FlagFrenzyModel)
    ModelCatalog.register_custom_action_dist("hybrid_action_dist", HybridActionDistribution)
    register_env("FlagFrenzyEnv-v0", env_creator)
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)

def load_model(checkpoint_path):
    """
    Load a trained model from a checkpoint
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        The loaded model
    """
    algorithm = Algorithm.from_checkpoint(checkpoint_path)
    model = algorithm.get_policy().model
    return model

def tensor_to_numpy(tensor):
    """
    Convert a PyTorch tensor to NumPy array
    
    Args:
        tensor: PyTorch tensor
        
    Returns:
        NumPy array
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor

def ensure_directory(path):
    """
    Ensure that a directory exists
    
    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)

def get_action_name(action_idx):
    """
    Get the name of an action based on its index
    
    Args:
        action_idx: Action index
        
    Returns:
        Action name
    """
    if 0 <= action_idx < len(ACTION_NAMES):
        return ACTION_NAMES[action_idx]
    return f"Unknown Action ({action_idx})"