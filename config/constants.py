from typing import Dict, Any

ENV_CONFIG: Dict[str, Any] = {
    "max_entities": 100,
    "entity_feat_dim": 26,
    "mission_dim": 7,
    "action_dim_param": 10,
    "horizon": 900
}

TRAINING_CONFIG: Dict[str, Any] = {
    "gamma": 0.995,
    "lambda_": 0.95,
    "clip_param": 0.2,
    "entropy_coeff": 0.01,
    "vf_clip_param": 10.0,
    "grad_clip": 0.5,
    "lr": 3e-5,
    "train_batch_size": 43200,
    "sgd_minibatch_size": 2048,
    "num_sgd_iter": 20,
    "num_rollout_workers": 1,
    "rollout_fragment_length": 900,
}

ACTION_NAMES = ["No-Op", "Move", "Return to Base", "Engage Target"]