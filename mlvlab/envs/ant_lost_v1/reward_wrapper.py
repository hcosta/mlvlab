"""
Reward Wrapper for AntLost-v1 Environment

This wrapper modifies the default rewards of the AntLost-v1 environment:
- -1.0 per step (unchanged)
- -100.0 for hitting an obstacle (unchanged) 
- -1000.0 for dying (exhausting the maximum number of steps)

Usage:
    from mlvlab.envs.ant_lost_v1.reward_wrapper import AntLostRewardWrapper
    import gymnasium as gym
    import mlvlab
    
    env = gym.make("mlv/AntLost-v1")
    env = AntLostRewardWrapper(env)
"""

import gymnasium as gym
from gymnasium import Wrapper
import numpy as np


class AntLostRewardWrapper(Wrapper):
    """
    Wrapper that modifies rewards for the AntLost-v1 environment.

    Reward structure:
    - -1.0 per step (movement cost)
    - -100.0 for hitting an obstacle
    - -1000.0 for dying (episode truncation due to step limit)
    """

    def __init__(self, env: gym.Env):
        """
        Initialize the reward wrapper.

        Args:
            env: The AntLost-v1 environment to wrap
        """
        super().__init__(env)
        self.step_count = 0
        self.max_steps = env.spec.max_episode_steps if env.spec else 20

    def reset(self, **kwargs):
        """Reset the environment and step counter."""
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0
        return obs, info

    def step(self, action):
        """
        Take a step in the environment and modify the reward.

        Args:
            action: The action to take

        Returns:
            observation, reward, terminated, truncated, info
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        # Modify reward based on the conditions
        if truncated:
            # Episode ended due to step limit (death)
            reward = -1000.0
        elif info.get('collided', False):
            # Hit an obstacle
            reward = -100.0
        else:
            # Normal step
            reward = -1.0

        return obs, reward, terminated, truncated, info


# Convenience function for easy usage
def wrap_antlost_env(env: gym.Env) -> AntLostRewardWrapper:
    """
    Convenience function to wrap an AntLost-v1 environment with custom rewards.

    Args:
        env: The AntLost-v1 environment

    Returns:
        Wrapped environment with modified rewards
    """
    return AntLostRewardWrapper(env)


# Example usage and testing
if __name__ == "__main__":
    import mlvlab

    # Create environment with custom rewards
    env = gym.make("mlv/AntLost-v1", render_mode="human")
    env = AntLostRewardWrapper(env)

    print("Testing AntLost-v1 with custom rewards:")
    print("- -1.0 per step")
    print("- -100.0 for hitting obstacle")
    print("- -1000.0 for dying (step limit)")
    print()

    obs, info = env.reset(seed=42)
    total_reward = 0
    step_count = 0

    while True:
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        total_reward += reward

        print(
            f"Step {step_count}: Position {obs}, Reward {reward}, Total {total_reward}")

        if terminated or truncated:
            print(f"\nEpisode finished after {step_count} steps")
            print(f"Final total reward: {total_reward}")
            break

    env.close()
