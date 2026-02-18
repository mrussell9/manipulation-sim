import os
import jax
import torch
import dataclasses
import numpy as np
from pathlib import Path
from openpi.policies import policy_config
from openpi.training import config as train_config
from openpi.models.pi0 import pi0_config

# Configure JAX to use less GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"  # Use only 30% of GPU memory
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# Configure PyTorch to use less GPU memory
torch.cuda.set_per_process_memory_fraction(0.3)  # Use only 30% of GPU memory


class Pi0Agent:
    def __init__(self, checkpoint_path):
        print("Restoring π₀ parameters (CPU staging)...")
        
        # Patch the download function to use local tokenizer
        tokenizer_path = Path(os.path.join(os.path.dirname(__file__), "tokenizers", "paligemma_tokenizer.model"))
        
        # Monkey-patch the download.maybe_download function
        from openpi.shared import download
        original_maybe_download = download.maybe_download
        
        def patched_maybe_download(url, **kwargs):
            if "paligemma_tokenizer.model" in url:
                return tokenizer_path
            return original_maybe_download(url, **kwargs)
        
        download.maybe_download = patched_maybe_download
        
        with jax.default_device(jax.devices("cpu")[0]):
            base_train_config = train_config.get_config("pi0_droid")
            optimized_model = pi0_config.Pi0Config(
                action_dim=32, action_horizon=4, max_token_len=32,
                dtype='bfloat16', paligemma_variant='gemma_2b',
                action_expert_variant='gemma_300m', pi05=False
            )
            model_config = dataclasses.replace(base_train_config, model=optimized_model)
            self.policy = policy_config.create_trained_policy(model_config, checkpoint_path)
        
        # Restore original function
        download.maybe_download = original_maybe_download
        
        gpu = jax.devices("gpu")[0]
        self.policy._model = jax.tree.map(
            lambda x: jax.device_put(x, gpu) if hasattr(x, "dtype") else x, 
            self.policy._model
        )
        print(f"π₀ Model initialized on {gpu}") 

    def get_observation(self, exterior_image_1_left, wrist_image_left, robot, prompt):
            # Get joint positions and convert to numpy array
            dof_positions = np.array(robot.get_dof_positions())
            
            # First 7 are arm joints, last 2 are gripper joints
            arm_joints = dof_positions[0, :7]
            gripper_position = dof_positions[7:9] if len(dof_positions) >= 9 else np.array([0.0, 0.0])
            # OpenPI expects observations in DROID format
            return {
                "observation/exterior_image_1_left": exterior_image_1_left,
                "observation/wrist_image_left": wrist_image_left,
                "observation/joint_position": arm_joints,
                "observation/gripper_position": gripper_position,
                "prompt": prompt
            }

    def compute_step(self, obs_packet):
        if obs_packet is None:
            return None
        with torch.no_grad():
            result = self.policy.infer(obs_packet)
        return result["actions"]
