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
import collections

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch


torch, nn = try_import_torch()

class FlagFrenzyModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.num_outputs = num_outputs
        self._features = None
        self._last_value = None
        self.enable_attribution = False
        self.attribution_target_idx = None
        self.attribution_results = None
        self.keep_attribution_enabled = False

        self.last_obs = None

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
        
        # Attribution analysis variables
        self.hooks = None
        
        # Add attribution analysis tracking
        self.gradients = {}  # Initialize gradients dictionary
        self._last_action_type_of_interest = None

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        raw_obs_dict = input_dict["obs"]
        self.last_obs = raw_obs_dict # Store the original observation
        
        if self.enable_attribution:
            # Setup hooks and get the obs dict with gradient-enabled tensors
            obs_dict = self._setup_attribution_hooks(raw_obs_dict)
        else:
            # Use the raw observation dictionary if attribution is disabled
            obs_dict = raw_obs_dict
            
        # Process entities
        if "entities" in obs_dict:
            entities = obs_dict["entities"]
            # Ensure entities tensor has the expected shape [batch, num_entities, features]
            if entities.dim() == 2: # Add batch dimension if missing
                 entities = entities.unsqueeze(0)
            entity_features = self.entity_encoder(entities)
        else:
            # Fallback if entities is missing (Use a default device and batch size)
            device = next(self.parameters()).device
            batch_size = 1
            entity_features = torch.zeros(batch_size, 256, device=device) # Encoder output size

        # Process visibility - handle both tensor and dict cases
        visibility_features = None # Initialize
        if "visibility" in obs_dict:
            visibility = obs_dict["visibility"]
            if isinstance(visibility, dict) or isinstance(visibility, collections.OrderedDict):
                # If visibility is a dictionary, extract legacy and dynasty components
                if "legacy" in visibility and "dynasty" in visibility and \
                   isinstance(visibility["legacy"], torch.Tensor) and isinstance(visibility["dynasty"], torch.Tensor):
                    
                    legacy = visibility["legacy"]
                    dynasty = visibility["dynasty"]
                    
                    # Ensure correct dimensions [batch, entities]
                    if legacy.dim() == 1: legacy = legacy.unsqueeze(0)
                    if dynasty.dim() == 1: dynasty = dynasty.unsqueeze(0)
                        
                    # Combine into a single tensor with shape [batch, entities, 2]
                    batch_size = legacy.shape[0]
                    combined_visibility = torch.cat([legacy.unsqueeze(-1), dynasty.unsqueeze(-1)], dim=-1)
                    
                    # Flatten to [batch, entities * 2]
                    flat_visibility = combined_visibility.reshape(batch_size, -1)
                    visibility_features = self.visibility_encoder(flat_visibility)
            elif isinstance(visibility, torch.Tensor):
                 # If already a tensor, use as is
                 if visibility.dim() == 2: # Add batch dim if missing
                      visibility = visibility.unsqueeze(0)
                 visibility_features = self.visibility_encoder(visibility)

        # Fallback for visibility features if processing failed or missing
        if visibility_features is None:
            device = entity_features.device # Use device from previous step
            batch_size = entity_features.shape[0]
            visibility_features = torch.zeros(batch_size, 64, device=device) # Encoder output size

        # Handle either "mission" or "mission_status" key
        mission_features = None # Initialize
        mission_key = "mission" if "mission" in obs_dict else "mission_status" if "mission_status" in obs_dict else None
        if mission_key and isinstance(obs_dict[mission_key], torch.Tensor):
            mission = obs_dict[mission_key]
            if mission.dim() == 1: mission = mission.unsqueeze(0) # Add batch dim
            mission_features = self.mission_encoder(mission)
            
        # Fallback for mission features
        if mission_features is None:
            device = entity_features.device
            batch_size = entity_features.shape[0]
            mission_features = torch.zeros(batch_size, 64, device=device) # Encoder output size

        # Process controllable entities
        controllable_features = None # Initialize
        if "controllable_entities" in obs_dict and isinstance(obs_dict["controllable_entities"], torch.Tensor):
            controllable = obs_dict["controllable_entities"]
            if controllable.dim() == 1: controllable = controllable.unsqueeze(0) # Add batch dim
            controllable_features = self.controllable_encoder(controllable)
            
        # Fallback for controllable features
        if controllable_features is None:
            device = entity_features.device
            batch_size = entity_features.shape[0]
            controllable_features = torch.zeros(batch_size, 64, device=device) # Encoder output size
            
        # Combine features
        # Ensure all feature tensors have a batch dimension before concatenating
        combined = torch.cat([
            entity_features, 
            visibility_features, 
            mission_features, 
            controllable_features
        ], dim=1)
        
        # Get combined features through the combined network
        features = self.combined_fc(combined)
        self._features = features # Store for value function

        # Get action type logits and params
        action_type_logits = self.action_type_head(features)
        param_mu = self.param_head_mu(features)
        param_log_std = self.param_head_log_std(features)
        
        # Combine outputs for the hybrid action distribution
        output = torch.cat([action_type_logits, param_mu, param_log_std], dim=1)
        
        # Verify dimensions match the hybrid action distribution's expectations
        expected_size = 4 + self.action_dim_param * 2  # 4 (discrete) + 10*2 (continuous mean & std)
        assert output.shape[1] == expected_size, f"Output dimension mismatch: got {output.shape[1]}, expected {expected_size}"

        # --- Attribution Calculation --- 
        if self.enable_attribution:
            # Select the output corresponding to the target action for gradient calculation
            if self.attribution_target_idx is not None and self.attribution_target_idx < action_type_logits.shape[1]:
                target_output = action_type_logits[:, self.attribution_target_idx]
            else:
                # Default: Use sum of all action logits if index invalid or None
                target_output = action_type_logits.sum(dim=1)

            # Zero out previous gradients stored in the model ONLY (hooks handle tensor grads)
            self.zero_grad() 
            
            # Calculate gradients w.r.t the target output
            # Gradients will flow back to the input tensors via the hooks in grad_obs_dict
            target_output.sum().backward(retain_graph=True) # Retain graph might be needed if value_function is called later
            
            # Process captured gradients into attribution scores
            self._process_attributions()
            
            # Auto-disable attribution unless keep_attribution_enabled is True
            if not self.keep_attribution_enabled:
                self.disable_attribution_analysis()
                
        return output, state

    @override(ModelV2)
    def value_function(self):
        """Return the value function estimate for the most recent forward pass."""
        assert self._features is not None, "must call forward() first"
        return self._value_branch(self._features).squeeze(-1)

    def _setup_attribution_hooks(self, raw_obs_dict):
        """
        Clones input tensors, enables gradients, registers hooks, and returns a 
        new obs dict with gradient-enabled tensors.
        
        Args:
            raw_obs_dict: The original observation dictionary.
            
        Returns:
            A new dictionary with relevant tensors cloned and requires_grad=True.
        """
        self.gradients = {}  # Reset gradients dictionary
        self.hooks = [] # Reset hooks list
        grad_obs_dict = {} # Create a new dict for gradient-enabled tensors

        # Function to save gradients
        def save_grad(name):
            def hook(grad):
                # Detach grad to prevent accumulation in subsequent passes if graph retained
                self.gradients[name] = grad.detach().clone() if grad is not None else None
            return hook
        
        # Process and clone tensors, enabling gradients and registering hooks
        for key, value in raw_obs_dict.items():
            if key == "visibility" and (isinstance(value, dict) or isinstance(value, collections.OrderedDict)):
                # Handle visibility dictionary separately
                grad_obs_dict[key] = {}
                if "legacy" in value and isinstance(value["legacy"], torch.Tensor):
                    legacy = value["legacy"].clone().detach().requires_grad_(True)
                    legacy.register_hook(save_grad("visibility_legacy"))
                    grad_obs_dict[key]["legacy"] = legacy
                    self.hooks.append(legacy)
                else: # Keep non-tensor value as is
                     grad_obs_dict[key]["legacy"] = value.get("legacy", None)
                     
                if "dynasty" in value and isinstance(value["dynasty"], torch.Tensor):
                    dynasty = value["dynasty"].clone().detach().requires_grad_(True)
                    dynasty.register_hook(save_grad("visibility_dynasty"))
                    grad_obs_dict[key]["dynasty"] = dynasty
                    self.hooks.append(dynasty)
                else: # Keep non-tensor value as is
                     grad_obs_dict[key]["dynasty"] = value.get("dynasty", None)
                     
            elif isinstance(value, torch.Tensor):
                # Clone tensor, enable grad, register hook
                tensor = value.clone().detach().requires_grad_(True)
                tensor.register_hook(save_grad(key)) # Use original key name for gradient storage
                grad_obs_dict[key] = tensor
                self.hooks.append(tensor)
            else:
                # Keep non-tensor values as they are (e.g., entity_id_list)
                grad_obs_dict[key] = value
                
        return grad_obs_dict

    def _compute_attribution(self, action_dict, obs_dict):
        """
        Compute gradients with respect to the model's output.
        
        Args:
            action_dict: Dictionary of actions
            obs_dict: Dictionary of observations
            
        Returns:
            None, but stores attribution results in self.attribution_results
        """
        if not self.enable_attribution or self.hooks is None:
            return
            
        # Clear previous gradients
        self.zero_grad()
        
        # Get the target action index
        target_idx = self.attribution_target_idx if self.attribution_target_idx is not None else 0
        
        # Get the logits for the target action
        logits = action_dict["logits"]
        if target_idx < logits.size(1):
            target = logits[0, target_idx]
            
            # Compute gradients
            target.backward(retain_graph=True)
            
            # Collect attribution results
            self.attribution_results = {}
            for name, module in self.named_modules():
                if hasattr(module, "weight") and hasattr(module.weight, "grad") and module.weight.grad is not None:
                    self.attribution_results[f"{name}.weight"] = module.weight.grad.detach().cpu().numpy()
                if hasattr(module, "bias") and hasattr(module.bias, "grad") and module.bias.grad is not None:
                    self.attribution_results[f"{name}.bias"] = module.bias.grad.detach().cpu().numpy()
                    
            # Store gradients for inputs
            for input_name, input_tensor in obs_dict.items():
                if input_tensor.grad is not None:
                    self.attribution_results[f"input.{input_name}"] = input_tensor.grad.detach().cpu().numpy()
        
        # Disable attribution if not meant to be kept enabled
        if not self.keep_attribution_enabled:
            self.disable_attribution_analysis()

    def enable_attribution_analysis(self, target_idx=None, keep_enabled=False):
        """Enable attribution analysis for the specified action index.
        
        Args:
            target_idx: Index of the action to compute attribution for
            keep_enabled: Whether to keep attribution enabled after the next forward pass
        """
        self.enable_attribution = True
        self.attribution_target_idx = target_idx
        self.keep_attribution_enabled = keep_enabled
        
    def disable_attribution_analysis(self):
        """Disable attribution analysis and clean up."""
        self.enable_attribution = False
        self.attribution_results = None
        
    def get_attributions(self):
        """
        Get the most recent attribution results.
        
        Returns:
            dict: Attribution results, or None if no attributions have been computed
        """
        return self.attribution_results

    def _clean_attribution_hooks(self):
        """
        Remove gradient hooks to prevent memory leaks.
        """
        if hasattr(self, 'hooks') and self.hooks:
            # Simply remove references to the tensors with hooks
            self.hooks = []
            
    def _process_attributions(self):
        """
        Process gradients to compute attribution scores.
        """
        if not hasattr(self, 'gradients') or not self.gradients:
            print("No gradients found for attribution")
            self.attribution_results = {} # Ensure it's initialized even if no gradients
            return
            
        # Now we process the gradients to get attributions
        attribution_scores = {}
        
        # Process entity gradients
        if "entities" in self.gradients and self.gradients["entities"] is not None:
            entity_grads = self.gradients["entities"]
            # Sum across feature dimensions to get per-entity importance
            if entity_grads.dim() > 2:  # If shape is [batch, num_entities, features]
                entity_attribution = entity_grads.abs().sum(dim=2)
                attribution_scores["entities"] = entity_attribution.squeeze(0) # Remove batch dim if present
            elif entity_grads.dim() == 2: # Shape [num_entities, features]
                 entity_attribution = entity_grads.abs().sum(dim=1)
                 attribution_scores["entities"] = entity_attribution

        # Process visibility gradients
        vis_scores = {}
        if "visibility_legacy" in self.gradients and self.gradients["visibility_legacy"] is not None:
             legacy_grads = self.gradients["visibility_legacy"]
             vis_scores["legacy"] = legacy_grads.abs().sum().item() # Sum scalar value
        if "visibility_dynasty" in self.gradients and self.gradients["visibility_dynasty"] is not None:
             dynasty_grads = self.gradients["visibility_dynasty"]
             vis_scores["dynasty"] = dynasty_grads.abs().sum().item() # Sum scalar value
        if vis_scores:
             attribution_scores["visibility"] = vis_scores
        elif "visibility" in self.gradients and self.gradients["visibility"] is not None: # Fallback for tensor visibility
             vis_grads = self.gradients["visibility"]
             if vis_grads.dim() > 1:
                 vis_attribution = vis_grads.abs().sum(dim=-1) # Sum features
                 attribution_scores["visibility"] = vis_attribution.squeeze(0) # Remove batch dim
             else:
                 attribution_scores["visibility"] = vis_grads.abs()


        # Process mission gradients
        mission_key = None
        if "mission" in self.gradients and self.gradients["mission"] is not None:
            mission_key = "mission"
        elif "mission_status" in self.gradients and self.gradients["mission_status"] is not None:
             mission_key = "mission_status"

        if mission_key:
            mission_grads = self.gradients[mission_key]
            # Sum across feature dimensions if necessary
            if mission_grads.dim() > 1: # [batch, features] or just [features]
                mission_attribution = mission_grads.abs() # Keep features separate? Or sum? Let's keep separate for now
                attribution_scores["mission"] = mission_attribution.squeeze(0) # Remove batch dim
            else:
                 attribution_scores["mission"] = mission_grads.abs()

        # Process controllable entities gradients
        if "controllable_entities" in self.gradients and self.gradients["controllable_entities"] is not None:
            ctrl_grads = self.gradients["controllable_entities"]
            # Sum across feature dimensions if necessary
            if ctrl_grads.dim() > 1: # [batch, features] or [features]
                ctrl_attribution = ctrl_grads.abs() # Keep features separate
                attribution_scores["controllable_entities"] = ctrl_attribution.squeeze(0) # Remove batch dim
            else:
                 attribution_scores["controllable_entities"] = ctrl_grads.abs()
        
        # Store attributions in self.attribution_results
        self.attribution_results = attribution_scores
        
        print(f"Attribution processing complete. Found {len(attribution_scores)} components.")
        # Debug print
        # for k, v in self.attribution_results.items():
        #     if isinstance(v, dict):
        #         print(f"  {k}: { {kk: vv.shape if hasattr(vv, 'shape') else vv for kk, vv in v.items()} }")
        #     elif hasattr(v, 'shape'):
        #         print(f"  {k}: shape {v.shape}")
        #     else:
        #          print(f"  {k}: {v}")

def compute_input_influence(model, observation, action_type_of_interest=None):
    """
    Calculate the influence of each input component on the model's output
    using gradients.
    
    Args:
        model: The trained FlagFrenzyModel
        observation: A single observation dict
        action_type_of_interest: Optional index of action type to analyze (0-3)
    
    Returns:
        Dictionary of attribution scores for each input component, or None if error.
    """
    # Reset any existing gradients
    model.zero_grad()
    
    # Convert obs to tensor if needed and ensure requires_grad and add batch dimension
    obs_tensors = {}
    for k, v in observation.items():
        if isinstance(v, dict):
            # For nested dicts like visibility
            obs_tensors[k] = {}
            for kk, vv in v.items():
                if isinstance(vv, (np.ndarray, list, float, int)):
                    # Add batch dimension if not already present
                    tensor_v = torch.tensor(vv, dtype=torch.float32, requires_grad=True)
                    if tensor_v.dim() == 1:
                        tensor_v = tensor_v.unsqueeze(0)  # Add batch dim
                    elif tensor_v.dim() == 0: # Handle scalar values
                         tensor_v = tensor_v.unsqueeze(0)
                    obs_tensors[k][kk] = tensor_v
                else:
                    obs_tensors[k][kk] = vv
        else:
            if isinstance(v, (np.ndarray, list, float, int)):
                # Add batch dimension if not already present
                tensor_v = torch.tensor(v, dtype=torch.float32, requires_grad=True)
                if tensor_v.dim() == 2:  # For 2D arrays like entities [100, 26]
                    tensor_v = tensor_v.unsqueeze(0)  # Add batch dim -> [1, 100, 26]
                elif tensor_v.dim() == 1:  # For 1D arrays
                    tensor_v = tensor_v.unsqueeze(0)  # Add batch dim
                elif tensor_v.dim() == 0: # Handle scalar values
                     tensor_v = tensor_v.unsqueeze(0)
                obs_tensors[k] = tensor_v
            else:
                # Skip non-numeric fields that can't have gradients
                obs_tensors[k] = v
    
    # Special handling for entity_id_list which is needed by the model
    if "entity_id_list" not in obs_tensors and "entity_id_list" in observation:
        obs_tensors["entity_id_list"] = observation["entity_id_list"]
    
    # Create input dict in the format model expects
    input_dict = {"obs": obs_tensors}
    
    # Enable attribution analysis on the model
    if isinstance(model, FlagFrenzyModel):
        # Use the consolidated method with target_idx and keep_enabled
        model.enable_attribution_analysis(target_idx=action_type_of_interest, keep_enabled=True)
        
        # Forward pass through the model
        try:
            with torch.enable_grad():
                # Forward pass through the model - this will also compute attributions
                # because keep_enabled is True and forward calls _setup & _process
                output, _ = model.forward(input_dict, [], None)
                
                # Get the processed attributions using the consolidated method
                attributions = model.get_attributions() # Should return self.attribution_results
                                
                # Explicitly disable attribution after the pass if needed (keep_enabled=True keeps it on)
                # model.disable_attribution_analysis() 

                return attributions if attributions is not None else {}
                
        except Exception as e:
            print(f"Error during gradient computation: {e}")
            import traceback
            traceback.print_exc()
            # Ensure attribution is disabled in case of error
            model.disable_attribution_analysis()
            return {}  # Return empty dict if there's an error
    else:
        print("Model is not an instance of FlagFrenzyModel, cannot compute influence.")
        return {}
