from classes import *
import pygame
import numpy as np

SERIES_DEMO = False
STEP_DEMO = True

if __name__ == "__main__":
    sim = TankDriveSimulation(dt=1)
    time_steps = 100  # Number of time steps per trajectory

    if SERIES_DEMO:
        # Define three different power input series.
        # Adjust the power values to be moderate given the robot's max velocity is 2 inches/step.
        P_L_1 = np.linspace(1, 3, time_steps)   # Gradually increasing left power
        P_R_1 = np.linspace(3, 1, time_steps)   # Decreasing right power; produces a curved path

        P_L_2 = np.full(time_steps, 2)          # Constant power for left wheel
        P_R_2 = np.full(time_steps, 2)          # Constant power for right wheel; should go straight

        P_L_3 = np.linspace(1, 4, time_steps)     # Sharper increase in left power
        P_R_3 = np.full(time_steps, 2)           # Constant right power; produces a sharp turn

        def run_trajectory(label, P_L, P_R):
            print(f"Running Trajectory: {label}")
            traj = sim.run_simulation(P_L, P_R, time_steps)
            # Pause for 2 seconds before next trajectory
            pygame.time.wait(2000)
            sim.reset()
            return traj

        traj1 = run_trajectory("Curved", P_L_1, P_R_1)
        traj2 = run_trajectory("Straight", P_L_2, P_R_2)
        traj3 = run_trajectory("Sharp Turn", P_L_3, P_R_3)

        print("Simulation complete. Close the window to exit.")
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
    
        pygame.quit()
    
    if STEP_DEMO:
        # --- Sample run using simulate_step ---
        num_steps = 19  # Number of simulation steps for the sample run
        print("Sample run using simulate_step:")
        for i in range(num_steps):
            # Here, we use constant power inputs for both wheels (e.g., 2 and 2)
            state = sim.simulate_step(90, 100)
            #print(f"Step {i+1}: x = {state[0]:.2f}, y = {state[1]:.2f}, theta = {state[2]:.2f}")
            print(f"({state[0]:.2f},{state[1]:.2f})")

