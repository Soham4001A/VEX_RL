import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import pygame

# World constants (in inches)
FIELD_SIZE_INCHES = 144  # 12 feet x 12 inches

# Display window size (in pixels)
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800

# Camera scale: how many pixels per inch
CAMERA_SCALE = 5  # The field will be 144*5 = 720 pixels wide
# Offsets to center the field in the window
OFFSET_X = (WINDOW_WIDTH - FIELD_SIZE_INCHES * CAMERA_SCALE) // 2  # (800 - 720)/2 = 40
OFFSET_Y = (WINDOW_HEIGHT - FIELD_SIZE_INCHES * CAMERA_SCALE) // 2  # 40

# Colors and framerate
BG_COLOR = (30, 30, 30)
FIELD_COLOR = (255, 255, 255)
ROBOT_COLOR = (0, 255, 0)
FPS = 30

# Robot dimensions in inches
ROBOT_WIDTH = 12
ROBOT_LENGTH = 16

# Global maximum power value (assumed range: -127 to 127)
P_max = 127.0

class TankDriveRobot:
    def __init__(self, x=FIELD_SIZE_INCHES/2, y=FIELD_SIZE_INCHES/2, theta=0, wheelbase=ROBOT_WIDTH, max_velocity=20):
        """
        Initialize the robot at the center of the field.
        x, y: Position in inches.
        theta: Orientation in radians.
        wheelbase: Distance between wheels (inches).
        max_velocity: Maximum velocity in inches per second.
        """
        self.x = float(x)
        self.y = float(y)
        self.theta = float(theta)
        self.wheelbase = float(wheelbase)
        self.max_velocity = float(max_velocity)  # V_max in inches per second

        # Wheel velocities (in inches per second)
        self.V_L = 0.0
        self.V_R = 0.0

        self.width = float(ROBOT_WIDTH)
        self.length = float(ROBOT_LENGTH)

    def apply_power(self, P_L, P_R, dt, A_max, V_max):
        """
        Update the robot's state based on power inputs.
        P_L, P_R: Power inputs for left/right wheels.
        dt: Time step in seconds.
        A_max: Maximum acceleration (in inches/s²) at full power.
        V_max: Maximum velocity (inches/s).
        """
        # Compute wheel acceleration:
        # When P == P_max and V == 0, a = A_max.
        # When V == V_max, a = 0.
        damping = A_max / V_max  # This replaces k_v.
        a_L = A_max * (P_L / P_max) - damping * self.V_L
        a_R = A_max * (P_R / P_max) - damping * self.V_R

        # Update wheel velocities (in inches per second)
        self.V_L = np.clip(self.V_L + a_L * dt, -V_max, V_max)
        self.V_R = np.clip(self.V_R + a_R * dt, -V_max, V_max)

        # Compute overall forward (linear) velocity and angular velocity.
        # Linear velocity V is the average of the two wheel velocities.
        # Angular velocity ω is given by (V_R - V_L)/wheelbase.
        V = (self.V_L + self.V_R) / 2.0
        omega = (self.V_R - self.V_L) / self.wheelbase

        # Update position and orientation (in inches and radians)
        self.x += V * np.cos(self.theta) * dt
        self.y += V * np.sin(self.theta) * dt
        self.theta += omega * dt

class TankDriveSimulation:
    def __init__(self, dt=1/18.9, domain_randomization=True):
        """
        dt: Time step in seconds (e.g., 1/19 s)
        """
        pygame.init()
        self.dt = dt
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Tank Drive Simulation")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # NOTE: You should know V_max & A_max of your robot for fine-tuning this model to your robot's parameters!

        # Domain randomization: randomize V_max and A_max
        if domain_randomization:
            # Randomize max velocity between 20 and 30 inches per second.
            V_max = np.random.uniform(30, 40)
            # Randomize maximum acceleration between 2 and 5 inches/s².
            A_max = np.random.uniform(0.5, 2)
        else:
            V_max = 20
            A_max = 5

        self.robot = TankDriveRobot(max_velocity=V_max)
        self.V_max = V_max
        self.A_max = A_max

    def reset(self):
        """Reinitialize the robot to the center of the field (preserving its V_max)."""
        self.robot = TankDriveRobot(max_velocity=self.robot.max_velocity)

    def simulate_step(self, P_L, P_R):
        """
        Perform a single simulation time step with the given power inputs.
        Returns a tuple:
          (x, y, theta, linear_velocity, angular_velocity, linear_acceleration, angular_acceleration)
        All values are in float precision.
        No rendering is done in this method.
        """
        # Record previous velocities for acceleration computation
        prev_v = (self.robot.V_L + self.robot.V_R) / 2.0
        prev_omega = (self.robot.V_R - self.robot.V_L) / self.robot.wheelbase
        
        # Apply the power inputs
        self.robot.apply_power(P_L, P_R, self.dt, self.A_max, self.V_max)
        
        # Compute new velocities
        new_v = (self.robot.V_L + self.robot.V_R) / 2.0
        new_omega = (self.robot.V_R - self.robot.V_L) / self.robot.wheelbase
        
        # Compute acceleration components
        linear_acc = (new_v - prev_v) / self.dt
        angular_acc = (new_omega - prev_omega) / self.dt
        
        return (self.robot.x, self.robot.y, self.robot.theta, new_v, new_omega, linear_acc, angular_acc)




    def render(self):
        """Draw the field and the robot."""
        self.screen.fill(BG_COLOR)
        # Draw the field boundaries (world grid from (0,0) to (144,144) inches)
        field_pixel_size = FIELD_SIZE_INCHES * CAMERA_SCALE
        field_rect = pygame.Rect(OFFSET_X, OFFSET_Y, field_pixel_size, field_pixel_size)
        pygame.draw.rect(self.screen, FIELD_COLOR, field_rect, 2)

        # Draw the robot as a rotated rectangle.
        # Convert robot's world coordinates (inches) to screen coordinates (pixels)
        cx = int(self.robot.x * CAMERA_SCALE + OFFSET_X)
        cy = int(self.robot.y * CAMERA_SCALE + OFFSET_Y)

        # Calculate half-dimensions in pixels
        half_length = (self.robot.length / 2.0) * CAMERA_SCALE
        half_width = (self.robot.width / 2.0) * CAMERA_SCALE

        # Define corners in the robot's local frame (assuming forward is positive x)
        corners = np.array([
            [ half_length,  half_width],
            [ half_length, -half_width],
            [-half_length, -half_width],
            [-half_length,  half_width]
        ])

        # Rotation matrix based on the robot's orientation
        cos_theta = np.cos(self.robot.theta)
        sin_theta = np.sin(self.robot.theta)
        R = np.array([[cos_theta, -sin_theta],
                      [sin_theta, cos_theta]])
        rotated = np.dot(corners, R.T)
        translated = rotated + np.array([cx, cy])
        pts = [(int(x), int(y)) for x, y in translated]

        pygame.draw.polygon(self.screen, ROBOT_COLOR, pts)
        pygame.display.flip()


class Transformer(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        embed_dim=64,
        num_heads=3,
        ff_hidden=128,
        num_layers=4,
        seq_len=6,
        dropout=0.3
    ):
        """
        :param observation_space: Gym observation space.
        :param embed_dim: Size of the embedding (d_model) in the Transformer.
        :param num_heads: Number of attention heads in the multi-head attention layers.
        :param ff_hidden: Dimension of the feedforward network in the Transformer.
        :param num_layers: Number of layers in the Transformer encoder.
        :param seq_len: Number of time steps to unroll in the Transformer.
        :param dropout: Dropout probability to use throughout the model.
        """
        # Features dimension after Transformer = embed_dim * seq_len
        feature_dim = embed_dim * seq_len
        super(Transformer, self).__init__(observation_space, features_dim=feature_dim)

        self.embed_dim = embed_dim
        self.input_dim = observation_space.shape[0]
        self.seq_len = seq_len  
        self.dropout_p = dropout

        # Validate that seq_len divides input_dim evenly
        if self.input_dim % seq_len != 0:
            raise ValueError("Input dimension must be divisible by seq_len.")

        # Linear projection for input -> embedding
        self.input_embedding = nn.Linear(self.input_dim // seq_len, embed_dim)

        # Dropout layer for embeddings
        self.embedding_dropout = nn.Dropout(p=self.dropout_p)

        # Transformer Encoder
        #   - The 'dropout' parameter here applies to:
        #       1) The self-attention mechanism outputs
        #       2) The feed-forward sub-layer outputs
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_hidden,
            dropout=self.dropout_p,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Flatten final output to feed into the policy & value networks
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Reshape input into (batch_size, seq_len, features_per_seq)
        batch_size = x.shape[0]
        features_per_seq = self.input_dim // self.seq_len
        x = x.view(batch_size, self.seq_len, features_per_seq)

        # Linear projection
        x = self.input_embedding(x)

        # Add positional encoding
        batch_size, seq_len, embed_dim = x.shape
        x = x + self._positional_encoding(seq_len, embed_dim).to(x.device)

        # Drop out some embeddings for regularization
        x = self.embedding_dropout(x)

        # Pass sequence through the Transformer encoder
        x = self.transformer(x)

        # Flatten for final feature vector
        return self.flatten(x)

    def _positional_encoding(self, seq_len, embed_dim):
        """Sine-cosine positional encoding, shape: (seq_len, embed_dim)."""
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # The standard “div_term” for sine/cosine in attention
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float()
            * (-np.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe