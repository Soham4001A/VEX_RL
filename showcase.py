import gymnasium as gym
import numpy as np
import pygame
from stable_baselines3 import PPO
from train import PPOEnv  # Ensure this imports your custom environment (PPOEnv)
from classes import FIELD_SIZE_INCHES

# Initialize the environment
env = PPOEnv()
obs, info = env.reset()

# Explicitly set the starting position and target position within the showcase:
# For example, set the start position at (50, 50) inches with heading 0
env.simulation.robot.x = 50.0
env.simulation.robot.y = 50.0
env.simulation.robot.theta = 0.0

# And set the target to (100, 100) inches with 0 radians heading
env.target = np.array([80.0, 80.0, 0.0], dtype=np.float32)

# Update the observation to include these explicit values
obs = np.concatenate((env.simulation.simulate_step(0, 0), env.target))

# Load the trained model. Adjust the path if necessary.
model = PPO.load("./PPO_V2/Trained_Model", env=env)

# Main episode loop
done = False
truncated = False

print("Running trained model. Close the pygame window to exit.")
while not done and not truncated:
    # Use the model to predict the next action.
    action, _ = model.predict(obs, deterministic=True)
    
    # Take a step in the environment.
    obs, reward, done, truncated, info = env.step(action)
    
    # Process internal events and force a render update.
    pygame.event.pump()
    env.simulation.render()

# Keep the window open until the user closes it.
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
print("Episode ended and window closed.")