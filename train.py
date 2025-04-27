# -----------------------------------------------------------
# Rllib PPO training script for FlagFrenzy
#
# Author : Sanjna Ravichandar
# Created: April 2025
# -----------------------------------------------------------
import os
import sys
import ray
from ray import air, tune

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from register_env import env_creator
from models.model import FlagFrenzyModel
from models.hybrid_action_dist import HybridActionDistribution
from config.constants import TRAINING_CONFIG, ENV_CONFIG

# Register model
ModelCatalog.register_custom_model("flag_frenzy_model", FlagFrenzyModel)
ModelCatalog.register_custom_action_dist("hybrid_action_dist", HybridActionDistribution)

register_env("FlagFrenzyEnv-v0", env_creator)
ray.init(ignore_reinit_error=True)


from ray.rllib.algorithms.callbacks import DefaultCallbacks


class FlagFrenzyCallbacks(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result, **kwargs):
        it = result["training_iteration"]
        if it <= 200:  # linear decay 0.01 -> 0.003
            new_coeff = 0.01 - 0.007 * (it / 200)
            algorithm.config["entropy_coeff"] = new_coeff

# Config for PPO
config = (
    PPOConfig()
    .environment(env="FlagFrenzyEnv-v0", env_config={})
    .framework("torch")
    .rollouts(
        num_rollout_workers=TRAINING_CONFIG["num_rollout_workers"],
        rollout_fragment_length=TRAINING_CONFIG["rollout_fragment_length"],
    )
    .training(
        model={
            "custom_model": "flag_frenzy_model",
            "custom_action_dist": "hybrid_action_dist",
        },
        gamma=TRAINING_CONFIG["gamma"],
        lambda_=TRAINING_CONFIG["lambda_"],
        clip_param=TRAINING_CONFIG["clip_param"],
        entropy_coeff=TRAINING_CONFIG["entropy_coeff"],
        vf_clip_param=TRAINING_CONFIG["vf_clip_param"],
        grad_clip=TRAINING_CONFIG["grad_clip"],
        lr=TRAINING_CONFIG["lr"],
        train_batch_size=TRAINING_CONFIG["train_batch_size"],
        sgd_minibatch_size=TRAINING_CONFIG["sgd_minibatch_size"],
        num_sgd_iter=TRAINING_CONFIG["num_sgd_iter"],
    )
    .resources(num_gpus=0)
    .callbacks(FlagFrenzyCallbacks)
    .to_dict()
)

# Use horizon from the centralized config
config["horizon"] = ENV_CONFIG["horizon"]

results = tune.Tuner(
    "PPO",
    run_config=air.RunConfig(
        stop={"training_iteration": 450},
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=1,
            num_to_keep=5),
        name="flag_frenzy_ppo",
        log_to_file=True),
    param_space=config,
).fit()