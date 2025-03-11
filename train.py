import os
import math
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# Internal Module Imports
from classes import *

DEBUG = True
REWARD_DEBUG = False
POSITIONAL_DEBUG = True
TARGET_RANDOM = False

class PPOEnv(gym.Env):
    """
    Custom Environment for training the ML model using our simulation.
    
    - Action Space: [Pl, Pr] with values in [-127, 127].
    - Observation Space: 
        [current_x, current_y, current_theta,
         linear_velocity, angular_velocity,
         linear_acceleration, angular_acceleration,
         target_x, target_y, target_theta]
    """
    def __init__(self):
        super(PPOEnv, self).__init__()
        
        # Instantiate the simulation with dt based on your sensor's update rate (e.g., 1/18.9 seconds)
        self.simulation = TankDriveSimulation(dt=1/18.9, domain_randomization=True)
        
        # Action space: two continuous values for left and right power inputs.
        self.action_space = spaces.Box(low=-127, high=127, shape=(2,), dtype=np.float32)
        
        # Observation space: 10-dimensional
        # [x, y, theta, linear_velocity, angular_velocity, linear_acceleration, angular_acceleration,
        #  target_x, target_y, target_theta]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        
        # Target will be set during reset
        self.target = None

        # Step Tracking
        self.current_step = 0
        self.max_steps = 190 # Total Episode Length which translates via (max_steps * dt) seconds

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)

        # Reset step counts
        self.current_step = 0
        
        # Randomize the robot's starting position and orientation within the field. (DISABLED CURRENTLY)
        #self.simulation.robot.x = np.random.uniform(0, FIELD_SIZE_INCHES)
        #self.simulation.robot.y = np.random.uniform(0, FIELD_SIZE_INCHES)
        #self.simulation.robot.theta = np.random.uniform(-math.pi, math.pi)
        
        if TARGET_RANDOM:
            # Randomize target position and orientation within the field.
            target_x = np.random.uniform(0, FIELD_SIZE_INCHES)
            target_y = np.random.uniform(0, FIELD_SIZE_INCHES)
            target_theta = np.random.uniform(-math.pi, math.pi)
        
        else:
            # Randomize target position and orientation within the field.
            target_x = FIELD_SIZE_INCHES - 20
            target_y = FIELD_SIZE_INCHES - 20
            target_theta = math.pi

        self.target = np.array([target_x, target_y, target_theta], dtype=np.float32)
        
        # Get initial state from simulation using zero power inputs.
        state = self.simulation.simulate_step(0, 0)
        # Combine simulation state with target information.
        obs = np.concatenate((np.array(state, dtype=np.float32), self.target))
        return obs, {}

    def step(self, action):
        self.current_step += 1
        action = np.array(action, dtype=np.float32).flatten()
        if action.shape[0] != 2:
            raise ValueError("Action must be a 2-dimensional vector for [Pl, Pr].")
        
        # Apply action for one time step.
        state = self.simulation.simulate_step(action[0], action[1])
        # Combine with target to form observation.
        obs = np.concatenate((np.array(state, dtype=np.float32), self.target))
        
        # Define reward function (customize as needed).
        reward = self._calculate_reward(obs, action)
        
        # You can set done if, for example, here it is set for a certain time step count
        done = False
        truncated = False
        
        if self.current_step == self.max_steps:
            done = truncated = True
        
        return obs, reward, done, truncated, {}

    def _angle_difference(self, angle1, angle2):
        """
        Computes the minimal difference between two angles (in radians).
        """
        diff = angle1 - angle2
        return (diff + np.pi) % (2 * np.pi) - np.pi

    def _calculate_reward(self, obs, action):
        # Extract robot's current position and heading from observation
        robot_x, robot_y, robot_theta = obs[0], obs[1], obs[2]
        # Extract target position (ignore target theta for now)
        target_x, target_y = self.target[0], self.target[1]
        
        # Compute Euclidean distance to target
        distance = np.linalg.norm(np.array([robot_x, robot_y]) - np.array([target_x, target_y]))
        
        # Compute desired heading: angle from robot to target
        desired_heading = np.arctan2(target_y - robot_y, target_x - robot_x)
        # Compute heading error (minimal angle difference)
        heading_error = abs(self._angle_difference(desired_heading, robot_theta))
        
        # Base reward: negative distance and penalty for misalignment.
        # You can tune the weight for the heading error penalty (here, 0.5)
        reward = -distance - 0.5 * heading_error
        
        # Bonus for being very close to the target (threshold of 5 inches)
        if distance < 5.0:
            reward += 100.0
        
        # Optional: penalize high magnitude actions to encourage smoother control.
        action_penalty = 0.001 * np.sum(np.square(action))
        reward -= action_penalty
        
        if REWARD_DEBUG:
            print(f"[Reward Debug] Distance: {distance:.2f}, Heading Error: {heading_error:.2f}, "
                  f"Action Penalty: {action_penalty:.4f}, Reward: {reward:.2f}")
        
        if POSITIONAL_DEBUG:
            with open("positional_debug.log", "a") as f:
                f.write(f"Robot: ({robot_x:.2f}, {robot_y:.2f}), Target: ({target_x:.2f}, {target_y:.2f}), Distance: {distance:.2f}\n")
                # tail -f positional_debug.log
        return reward


if __name__ == "__main__":

    def linear_schedule(initial_value: float):
        """
        Linear schedule function for learning rate.
        Decreases linearly from `initial_value` to 0.
        """
        def schedule(progress_remaining: float):
            return progress_remaining * initial_value
        return schedule
    

    env = PPOEnv() 
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    policy_kwargs = dict(
        features_extractor_class=Transformer,
        features_extractor_kwargs=dict(embed_dim=64, num_heads=8, ff_hidden=64*4, num_layers=4, seq_len=5),
        net_arch = dict(pi = [128,128,64],vf=[128,256,256,64]) #use keyword (pi) for policy network architecture -> additional ffn for decoding output, (vf) for reward func
    )

    model = PPO(
        policy="MlpPolicy",
        policy_kwargs=policy_kwargs,
        env=vec_env,
        verbose=1,
        normalize_advantage = True,
        use_sde = False, # Essentially reducing delta of actions when rewards are very positive (breaks it while initially learning)
        #sde_sample_freq = 3,
        learning_rate=linear_schedule(initial_value= 0.0003),
        n_steps=95, # Steps per learning update
        batch_size=19,
        gamma=0.85,
        gae_lambda= 0.8,
        vf_coef = 0.85, # Lower reliance on v(s) to compute advantage which is then used to compute Loss -> Gradient
        clip_range=0.4, # Clips larger updates to remain within +- 60%
        #ent_coef=0.05,
        #tensorboard_log="./Patrol&Proetect_PPO/ppo_patrol_tensorboard/"
    )

    model.learn(total_timesteps=190_000)

    # Save the model
    model.save("./PPO_V2/Trained_Model")
    vec_env.save("./PPO_V2/Trained_VecNormalize.pkl")
    print("Model Saved Succesfully!")
    #os.system("pmset displaysleepnow")