# -----------------------------------------------------------
# Custom Model for Flag Frenzy Env
#
# Author : Sanjna Ravichandar
# Created: April 2025
# -----------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch


torch, nn = try_import_torch()

class FlagFrenzyModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.last_obs = None
        
        # Add required attributes for attribution analysis
        self.enable_attribution = False
        self.attribution_target_idx = None
        self.attribution_results = None
        self.gradients = {}
        self._features = None

        # TODO: Don't hardcode!
        self.max_entities = 100
        self.entity_feat_dim = 26
        self.mission_dim = 7

        self.action_dim_param = 10

        # Entity encoder
        self.entity_encoder = nn.Sequential(
            nn.Flatten(),  # (max_entities * entity_feat_dim,)
            nn.Linear(self.max_entities * self.entity_feat_dim, 256),
            nn.ReLU()
        )

        # Radar visibility encoder
        self.visibility_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.max_entities * 2, 64),
            nn.ReLU()
        )

        # Mission status encoder
        self.mission_encoder = nn.Sequential(
            nn.Linear(self.mission_dim, 64),
            nn.ReLU()
        )

        # Controllable entities encoder
        self.controllable_encoder = nn.Sequential(
            nn.Linear(self.max_entities, 64),
            nn.ReLU()
        )

        # Combine and output
        self.combined_fc = nn.Sequential(
            nn.Linear(256 + 64 + 64 + 64, 256),
            nn.ReLU()
        )

        # Discrete action type head
        self.action_type_head = nn.Linear(256, 4)

        # Continuous param vector head
        self.param_head_mu = nn.Linear(256, self.action_dim_param)
        self.param_head_log_std = nn.Linear(256, self.action_dim_param)

        self._value_branch = nn.Linear(256, 1)
        self._last_value = None

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        self.last_obs = obs

        self.entity_id_list = input_dict["obs"]["entity_id_list"]

        entities = obs["entities"]
        mission = obs["mission"]
        visibility = obs["visibility"]
        controllable_entities= obs.get("controllable_entities", None)
        engage_mask = obs.get("valid_engage_mask", None)

        entity_encoded = self.entity_encoder(entities)

        legacy = visibility["legacy"].float()   # (batch, 100)
        dynasty = visibility["dynasty"].float() # (batch, 100)
        vis_concat = torch.cat([legacy, dynasty], dim=-1)   # (batch, 200)
        vis_encoded = self.visibility_encoder(vis_concat)   # (batch, 64)

        mission_encoded = self.mission_encoder(mission)

        controllable_encoded = self.controllable_encoder(controllable_entities)  # (B, 64)

        # Combine encoders
        x = torch.cat([controllable_encoded, entity_encoded, vis_encoded, mission_encoded], dim=-1)   # (batch, 256 + 64 + 64 + 64 = 384)
        features = self.combined_fc(x)  # (batch, 256)
        self._features = features

        # Compute logits + params
        logits = self.action_type_head(features)

        mu = self.param_head_mu(features)
        log_std = self.param_head_log_std(features).clamp(-5, 2)
        std = torch.exp(log_std)

        params = torch.cat([mu, std], dim=1)  # (batch, 20)

        # Mask out "engage target" (action_type == 3) if no valid targets are available
        if engage_mask is not None:
            if engage_mask.dim() == 1:
                engage_mask = engage_mask.unsqueeze(0)  # Add batch dim

            engage_mask = engage_mask.view(engage_mask.shape[0], -1)
            has_valid_targets = (engage_mask > 0.5).any(dim=1)

            mask_value = torch.tensor(-1e9, dtype=logits.dtype, device=logits.device)
            logits[:, 3] = torch.where(has_valid_targets, logits[:, 3], mask_value)


        # Concatenate: 4 logits, 10 params
        return torch.cat([logits, params], dim=1), state

    @override(ModelV2)
    def value_function(self):
        return self._value_branch(self._features).squeeze(1)

    def _process_attributions(self):
        """
        Process the gradients into attribution scores
        """
        # Create attribution scores dictionary
        attribution_scores = {}
        
        # Process gradients for different components
        for name, gradient in self.gradients.items():
            if gradient is None:
                continue
                
            # Skip empty tensors
            if gradient.numel() == 0:
                continue
                
            # Process different observation components
            if name == "entities":
                # For entities, compute attributions per entity
                # Sum over the feature dimension to get per-entity scores
                if gradient.dim() > 2:  # batch, entities, features
                    entity_scores = torch.sum(torch.abs(gradient), dim=2).squeeze(0)
                    attribution_scores["entities"] = entity_scores.detach().cpu().numpy()
                else:
                    # Handle case where entities tensor might have different dims
                    entity_scores = torch.abs(gradient)
                    if entity_scores.dim() == 2:
                        entity_scores = entity_scores.sum(dim=1)
                    attribution_scores["entities"] = entity_scores.detach().cpu().numpy()
                
            elif name == "mission":
                # For mission status, use raw gradients
                mission_scores = torch.sum(torch.abs(gradient), dim=0)
                # Ensure 1D array
                if mission_scores.dim() > 1:
                    mission_scores = mission_scores.sum(dim=0)
                attribution_scores["mission"] = mission_scores.detach().cpu().numpy()
                
            elif name == "controllable_entities":
                # For controllable entities mask
                ctrl_scores = torch.abs(gradient).squeeze(0)
                # Ensure 1D array
                if ctrl_scores.dim() > 1:
                    ctrl_scores = ctrl_scores.sum(dim=0)
                attribution_scores["controllable_entities"] = ctrl_scores.detach().cpu().numpy()
                
            elif name == "visibility.legacy":
                # Create visibility dict if it doesn't exist
                if "visibility" not in attribution_scores:
                    attribution_scores["visibility"] = {}
                # Add legacy visibility scores
                attribution_scores["visibility"]["legacy"] = torch.abs(gradient).squeeze(0).detach().cpu().numpy()
                
            elif name == "visibility.dynasty":
                # Create visibility dict if it doesn't exist
                if "visibility" not in attribution_scores:
                    attribution_scores["visibility"] = {}
                # Add dynasty visibility scores
                attribution_scores["visibility"]["dynasty"] = torch.abs(gradient).squeeze(0).detach().cpu().numpy()
        
        # Store attributions in self.attribution_results
        self.attribution_results = attribution_scores
        
        print(f"Attribution processing complete. Found {len(attribution_scores)} components.")

def compute_input_influence(model, observation, action_type_of_interest=None):
    """
    Calculate the influence of each input component on the model's output
    using integrated gradients attribution method.
    
    Args:
        model: The trained FlagFrenzyModel
        observation: A single observation dict
        action_type_of_interest: Optional index of action type to analyze (0-3)
    
    Returns:
        Dictionary of attribution scores for each input component
    """
    # Reset any existing gradients
    model.zero_grad()
    
    # Convert observation to tensors with gradient tracking
    obs_tensors = {}
    for k, v in observation.items():
        if isinstance(v, dict):
            # For nested dicts like visibility
            obs_tensors[k] = {}
            for kk, vv in v.items():
                if isinstance(vv, (np.ndarray, list, float, int)):
                    tensor_v = torch.tensor(vv, dtype=torch.float32, requires_grad=True)
                    if tensor_v.dim() == 1:
                        tensor_v = tensor_v.unsqueeze(0)  # Add batch dim
                    obs_tensors[k][kk] = tensor_v
                else:
                    obs_tensors[k][kk] = vv
        elif isinstance(v, (np.ndarray, list, float, int)):
            tensor_v = torch.tensor(v, dtype=torch.float32, requires_grad=True)
            # Special handling for entities which needs 3D shape [batch, entities, features]
            if k == "entities" and tensor_v.dim() == 2:
                tensor_v = tensor_v.unsqueeze(0)  # [entities, features] -> [1, entities, features]
            elif tensor_v.dim() == 1:
                tensor_v = tensor_v.unsqueeze(0)  # Add batch dim
            obs_tensors[k] = tensor_v
    
    # Enable attribution analysis in the model
    model.enable_attribution = True
    model.attribution_target_idx = action_type_of_interest
    
    # Setup gradients dict to track results
    gradients = {}
    
    # Register hooks for all input tensors
    hooks = []
    
    def hook_factory(name):
        def hook(grad):
            gradients[name] = grad.detach().clone()
        return hook
    
    # Register hooks for all input tensors
    for k, v in obs_tensors.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, torch.Tensor) and vv.requires_grad:
                    hook = vv.register_hook(hook_factory(f"{k}.{kk}"))
                    hooks.append(hook)
        elif isinstance(v, torch.Tensor) and v.requires_grad:
            hook = v.register_hook(hook_factory(k))
            hooks.append(hook)
    
    try:
        # Add debugging for tensor shapes
        print(f"Entities shape: {obs_tensors['entities'].shape if 'entities' in obs_tensors else 'missing'}")
        
        # Forward pass through the model
        input_dict = {"obs": obs_tensors}
        output, _ = model(input_dict, [], None)
        
        # Select the action logits for the action of interest
        if action_type_of_interest is not None and action_type_of_interest < 4:
            logits = output[0, action_type_of_interest]
        else:
            # Use all action logits if not specified
            logits = torch.sum(output[0, :4])
        
        # Compute gradients
        logits.backward()
        
        # Process gradients into attribution scores
        attribution_scores = {}
        
        # Process entities gradients
        if "entities" in gradients:
            entity_grad = gradients["entities"]
            # For entities, compute attribution per entity by summing across features
            if entity_grad.dim() > 2:  # batch, entities, features
                entity_scores = torch.sum(torch.abs(entity_grad), dim=2).squeeze(0)
                attribution_scores["entities"] = entity_scores.detach().cpu().numpy()
            else:
                # Handle case where entities tensor might have different dims
                entity_scores = torch.abs(entity_grad)
                if entity_scores.dim() == 2:
                    entity_scores = entity_scores.sum(dim=1)
                attribution_scores["entities"] = entity_scores.detach().cpu().numpy()
        
        # Process mission gradients
        if "mission" in gradients:
            mission_grad = gradients["mission"]
            # For mission, use absolute gradients directly
            mission_scores = torch.abs(mission_grad).squeeze(0)
            # Ensure 1D array
            if mission_scores.dim() > 1:
                mission_scores = mission_scores.sum(dim=0)
            attribution_scores["mission"] = mission_scores.detach().cpu().numpy()
        
        # Process controllable entities gradients
        if "controllable_entities" in gradients:
            ctrl_grad = gradients["controllable_entities"]
            # For controllable entities, use absolute gradients
            ctrl_scores = torch.abs(ctrl_grad).squeeze(0)
            # Ensure 1D array
            if ctrl_scores.dim() > 1:
                ctrl_scores = ctrl_scores.sum(dim=0)
            attribution_scores["controllable_entities"] = ctrl_scores.detach().cpu().numpy()
        
        # Process visibility gradients
        if "visibility.legacy" in gradients or "visibility.dynasty" in gradients:
            attribution_scores["visibility"] = {}
            
            if "visibility.legacy" in gradients:
                legacy_grad = gradients["visibility.legacy"]
                # Sum to get a scalar value
                attribution_scores["visibility"]["legacy"] = torch.abs(legacy_grad).sum().item()
            
            if "visibility.dynasty" in gradients:
                dynasty_grad = gradients["visibility.dynasty"]
                # Sum to get a scalar value
                attribution_scores["visibility"]["dynasty"] = torch.abs(dynasty_grad).sum().item()
    except Exception as e:
        print(f"Error during attribution analysis: {e}")
        # Return empty results on error
        attribution_scores = {"error": str(e)}
    finally:
        # Clean up hooks to prevent memory leaks
        for hook in hooks:
            hook.remove()
        
        # Disable attribution in model
        model.enable_attribution = False
        model.attribution_target_idx = None
    
    return attribution_scores