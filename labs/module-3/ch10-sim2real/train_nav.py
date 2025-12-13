#!/usr/bin/env python3
"""
Reinforcement Learning Navigation Training Script

This script trains a navigation policy using RL in Isaac Sim,
with domain randomization for sim-to-real transfer.

Lab 10: Navigation and Sim2Real Transfer
"""

from omni.isaac.kit import SimulationApp

CONFIG = {
    "headless": True,  # Set to False for visualization during development
    "width": 800,
    "height": 600,
}

simulation_app = SimulationApp(CONFIG)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
import random
from typing import Tuple, List, Dict, Optional
import time
import os
import json

# Isaac Sim imports
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects import DynamicSphere, VisualSphere
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.sensor import IMUSensor
from pxr import Gf


class NavigationEnvironment:
    """
    RL Environment for robot navigation with domain randomization.

    Observations:
    - Robot position and orientation
    - Goal position relative to robot
    - LIDAR-like distance measurements
    - Linear and angular velocities

    Actions:
    - Linear velocity (forward/backward)
    - Angular velocity (turn left/right)

    Domain Randomization:
    - Goal position
    - Obstacle positions
    - Floor friction
    - Robot mass variations
    """

    def __init__(self,
                 max_episode_steps: int = 500,
                 goal_threshold: float = 0.3,
                 collision_penalty: float = -10.0,
                 goal_reward: float = 100.0,
                 step_penalty: float = -0.1):
        """
        Initialize navigation environment.

        Args:
            max_episode_steps: Maximum steps per episode
            goal_threshold: Distance to goal for success (meters)
            collision_penalty: Penalty for collision
            goal_reward: Reward for reaching goal
            step_penalty: Per-step penalty to encourage efficiency
        """
        self.max_episode_steps = max_episode_steps
        self.goal_threshold = goal_threshold
        self.collision_penalty = collision_penalty
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty

        # Create world
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()

        # Environment bounds
        self.arena_size = 5.0

        # Robot and goal
        self.robot = None
        self.goal_marker = None
        self.goal_position = np.array([0, 0, 0])
        self.obstacles = []

        # State
        self.current_step = 0
        self.episode_count = 0

        # Domain randomization settings
        self.dr_config = {
            "randomize_goal": True,
            "randomize_obstacles": True,
            "randomize_friction": True,
            "randomize_mass": True,
            "num_obstacles": (3, 8),  # min, max
            "friction_range": (0.3, 1.0),
            "mass_scale_range": (0.8, 1.2),
        }

        # Observation and action spaces
        self.obs_dim = 12  # [pos(2), ori(1), goal_rel(2), vel(2), dist_sensors(5)]
        self.act_dim = 2   # [linear_vel, angular_vel]

        self._setup_scene()

    def _setup_scene(self):
        """Set up the simulation scene."""
        # Add robot (using a simple differential drive model)
        try:
            assets_root = get_assets_root_path()
            robot_usd = f"{assets_root}/Isaac/Robots/Jetbot/jetbot.usd"
            add_reference_to_stage(robot_usd, "/World/Robot")

            self.robot = self.world.scene.add(
                Articulation(
                    prim_path="/World/Robot",
                    name="robot",
                    position=np.array([0, 0, 0.1])
                )
            )
        except Exception as e:
            print(f"Could not load robot asset: {e}")
            print("Using simple sphere as placeholder")

            from omni.isaac.core.objects import DynamicSphere
            self.robot = self.world.scene.add(
                DynamicSphere(
                    prim_path="/World/Robot",
                    name="robot",
                    position=np.array([0, 0, 0.1]),
                    radius=0.15,
                    color=np.array([0.2, 0.2, 0.8])
                )
            )

        # Add goal marker (visual only)
        self.goal_marker = self.world.scene.add(
            VisualSphere(
                prim_path="/World/Goal",
                name="goal",
                position=np.array([2, 0, 0.1]),
                radius=0.15,
                color=np.array([0.0, 1.0, 0.0])  # Green
            )
        )

        self.world.reset()
        print("Navigation environment initialized")

    def _randomize_goal(self):
        """Randomize goal position."""
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(2.0, self.arena_size - 0.5)
        self.goal_position = np.array([
            distance * np.cos(angle),
            distance * np.sin(angle),
            0.1
        ])
        self.goal_marker.set_world_pose(position=self.goal_position)

    def _randomize_obstacles(self):
        """Add random obstacles to the scene."""
        # Remove existing obstacles
        for obs in self.obstacles:
            self.world.scene.remove_object(obs.name)
        self.obstacles = []

        if not self.dr_config["randomize_obstacles"]:
            return

        num_obs = np.random.randint(*self.dr_config["num_obstacles"])

        for i in range(num_obs):
            # Random position (avoiding robot start and goal)
            while True:
                pos = np.array([
                    np.random.uniform(-self.arena_size + 0.5, self.arena_size - 0.5),
                    np.random.uniform(-self.arena_size + 0.5, self.arena_size - 0.5),
                    0.25
                ])
                # Check distance from origin (robot start) and goal
                if np.linalg.norm(pos[:2]) > 1.0 and np.linalg.norm(pos[:2] - self.goal_position[:2]) > 1.0:
                    break

            # Random size
            size = np.random.uniform(0.2, 0.5)

            from omni.isaac.core.objects import DynamicCuboid
            obs = self.world.scene.add(
                DynamicCuboid(
                    prim_path=f"/World/Obstacle_{i}",
                    name=f"obstacle_{i}",
                    position=pos,
                    scale=np.array([size, size, 0.5]),
                    color=np.array([0.5, 0.5, 0.5])
                )
            )
            self.obstacles.append(obs)

    def _get_distance_sensors(self) -> np.ndarray:
        """
        Simulate LIDAR-like distance sensors.

        Returns 5 distance readings at angles: -90, -45, 0, 45, 90 degrees
        """
        # Get robot position
        robot_pos, robot_ori = self.robot.get_world_pose()

        # Simplified: return distances to nearest obstacles
        # In production, use ray casting for accurate readings
        distances = np.full(5, self.arena_size)

        angles = [-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2]

        for i, angle in enumerate(angles):
            for obs in self.obstacles:
                obs_pos, _ = obs.get_world_pose()
                rel_pos = obs_pos[:2] - robot_pos[:2]
                dist = np.linalg.norm(rel_pos)
                if dist < distances[i]:
                    distances[i] = dist

            # Arena walls
            # ... simplified wall distance calculation

        return distances

    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation."""
        self.current_step = 0
        self.episode_count += 1

        # Reset robot position
        self.robot.set_world_pose(
            position=np.array([0, 0, 0.1]),
            orientation=np.array([1, 0, 0, 0])  # w, x, y, z quaternion
        )

        # Apply domain randomization
        if self.dr_config["randomize_goal"]:
            self._randomize_goal()

        if self.dr_config["randomize_obstacles"]:
            self._randomize_obstacles()

        self.world.reset()

        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Robot state
        robot_pos, robot_ori = self.robot.get_world_pose()

        # Convert quaternion to yaw angle
        # Simplified: just use z rotation
        yaw = 2 * np.arctan2(robot_ori[3], robot_ori[0])

        # Goal relative to robot
        goal_rel = self.goal_position[:2] - robot_pos[:2]

        # Velocities (simplified - would get from physics in real implementation)
        linear_vel = np.array([0.0, 0.0])
        angular_vel = 0.0

        # Distance sensors
        distances = self._get_distance_sensors()

        obs = np.concatenate([
            robot_pos[:2],      # 2
            [yaw],              # 1
            goal_rel,           # 2
            linear_vel,         # 2
            distances           # 5
        ])

        return obs.astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take action and return (observation, reward, done, info).

        Args:
            action: [linear_velocity, angular_velocity]
        """
        self.current_step += 1

        # Apply action (simplified - would control actual joints)
        linear_vel = float(np.clip(action[0], -1, 1))
        angular_vel = float(np.clip(action[1], -1, 1))

        # Simple kinematic update
        robot_pos, robot_ori = self.robot.get_world_pose()
        yaw = 2 * np.arctan2(robot_ori[3], robot_ori[0])

        dt = 0.02  # 50 Hz
        new_yaw = yaw + angular_vel * dt
        new_x = robot_pos[0] + linear_vel * np.cos(new_yaw) * dt
        new_y = robot_pos[1] + linear_vel * np.sin(new_yaw) * dt

        # Update robot position
        new_ori = np.array([np.cos(new_yaw/2), 0, 0, np.sin(new_yaw/2)])
        self.robot.set_world_pose(
            position=np.array([new_x, new_y, robot_pos[2]]),
            orientation=new_ori
        )

        # Step physics
        self.world.step(render=False)

        # Get observation
        obs = self._get_observation()

        # Calculate reward
        reward, done, info = self._calculate_reward(obs)

        # Check termination
        if self.current_step >= self.max_episode_steps:
            done = True
            info["timeout"] = True

        return obs, reward, done, info

    def _calculate_reward(self, obs: np.ndarray) -> Tuple[float, bool, dict]:
        """Calculate reward and check termination."""
        info = {"success": False, "collision": False}

        robot_pos = obs[:2]
        goal_rel = obs[3:5]
        distances = obs[7:12]

        # Distance to goal
        dist_to_goal = np.linalg.norm(goal_rel)

        # Progress reward (moving toward goal)
        reward = self.step_penalty

        # Goal reached
        if dist_to_goal < self.goal_threshold:
            reward = self.goal_reward
            info["success"] = True
            return reward, True, info

        # Collision check (simplified)
        if np.min(distances) < 0.2:
            reward = self.collision_penalty
            info["collision"] = True
            return reward, True, info

        # Out of bounds
        if np.linalg.norm(robot_pos) > self.arena_size:
            reward = self.collision_penalty
            return reward, True, info

        # Shaping reward: closer to goal is better
        reward += -dist_to_goal * 0.1

        return reward, False, info


class PolicyNetwork(nn.Module):
    """
    Actor-Critic policy network for navigation.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor (policy)
        self.actor_mean = nn.Linear(hidden_dim, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))

        # Critic (value function)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        features = self.features(obs)
        return features

    def get_action(self, obs, deterministic=False):
        features = self.forward(obs)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std)

        if deterministic:
            return action_mean

        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)

        return action, log_prob

    def get_value(self, obs):
        features = self.forward(obs)
        return self.critic(features)


class PPOTrainer:
    """
    Proximal Policy Optimization trainer for navigation.
    """

    def __init__(self,
                 env: NavigationEnvironment,
                 hidden_dim: int = 256,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 n_epochs: int = 10,
                 batch_size: int = 64):

        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on: {self.device}")

        # Policy network
        self.policy = PolicyNetwork(env.obs_dim, env.act_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Logging
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate = deque(maxlen=100)

    def collect_rollout(self, n_steps: int = 2048) -> dict:
        """Collect experience from environment."""
        obs_list = []
        actions_list = []
        rewards_list = []
        dones_list = []
        log_probs_list = []
        values_list = []

        obs = self.env.reset()

        for _ in range(n_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, log_prob = self.policy.get_action(obs_tensor)
                value = self.policy.get_value(obs_tensor)

            action_np = action.cpu().numpy().squeeze()
            next_obs, reward, done, info = self.env.step(action_np)

            obs_list.append(obs)
            actions_list.append(action_np)
            rewards_list.append(reward)
            dones_list.append(done)
            log_probs_list.append(log_prob.item())
            values_list.append(value.item())

            obs = next_obs

            if done:
                self.episode_rewards.append(sum(rewards_list[-self.env.current_step:]))
                self.episode_lengths.append(self.env.current_step)
                self.success_rate.append(1.0 if info.get("success", False) else 0.0)
                obs = self.env.reset()

        return {
            "obs": np.array(obs_list),
            "actions": np.array(actions_list),
            "rewards": np.array(rewards_list),
            "dones": np.array(dones_list),
            "log_probs": np.array(log_probs_list),
            "values": np.array(values_list),
        }

    def compute_gae(self, rewards, values, dones) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation."""
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)

        last_advantage = 0
        last_value = values[-1]

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]

        returns = advantages + values
        return advantages, returns

    def update(self, rollout: dict):
        """Update policy using PPO."""
        obs = torch.FloatTensor(rollout["obs"]).to(self.device)
        actions = torch.FloatTensor(rollout["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(rollout["log_probs"]).to(self.device)

        advantages, returns = self.compute_gae(
            rollout["rewards"],
            rollout["values"],
            rollout["dones"]
        )

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update epochs
        for _ in range(self.n_epochs):
            # Shuffle data
            indices = np.random.permutation(len(obs))

            for start in range(0, len(obs), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Get new log probs and values
                features = self.policy(batch_obs)
                action_mean = self.policy.actor_mean(features)
                action_std = torch.exp(self.policy.actor_log_std)
                dist = Normal(action_mean, action_std)
                new_log_probs = dist.log_prob(batch_actions).sum(-1)
                values = self.policy.critic(features).squeeze()

                # Policy loss (clipped)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(values, batch_returns)

                # Entropy bonus
                entropy = dist.entropy().mean()

                # Total loss
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

    def train(self, total_timesteps: int = 1000000, log_interval: int = 10):
        """Main training loop."""
        print(f"\nStarting PPO training for {total_timesteps} timesteps")
        print("=" * 60)

        timesteps = 0
        iteration = 0

        while timesteps < total_timesteps:
            # Collect rollout
            rollout = self.collect_rollout(n_steps=2048)
            timesteps += len(rollout["rewards"])

            # Update policy
            self.update(rollout)

            iteration += 1

            # Logging
            if iteration % log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
                avg_length = np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0
                success = np.mean(self.success_rate) if self.success_rate else 0

                print(f"Iter {iteration} | Steps: {timesteps} | "
                      f"Reward: {avg_reward:.2f} | Length: {avg_length:.1f} | "
                      f"Success: {success:.1%}")

        return self.policy

    def save(self, path: str):
        """Save trained policy."""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        print(f"Policy saved to {path}")

    def load(self, path: str):
        """Load trained policy."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Policy loaded from {path}")


def main():
    """Main training script."""
    print("=" * 60)
    print("Lab 10: RL Navigation Training with Domain Randomization")
    print("=" * 60)

    # Create environment
    env = NavigationEnvironment(
        max_episode_steps=500,
        goal_threshold=0.3,
    )

    # Enable domain randomization
    env.dr_config.update({
        "randomize_goal": True,
        "randomize_obstacles": True,
        "randomize_friction": True,
        "num_obstacles": (3, 6),
    })

    # Create trainer
    trainer = PPOTrainer(
        env=env,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
    )

    # Train
    try:
        policy = trainer.train(
            total_timesteps=100000,  # Increase for better results
            log_interval=10
        )

        # Save policy
        os.makedirs("models", exist_ok=True)
        trainer.save("models/nav_policy.pt")

        # Save training stats
        stats = {
            "episode_rewards": trainer.episode_rewards,
            "episode_lengths": trainer.episode_lengths,
            "final_success_rate": float(np.mean(trainer.success_rate)),
        }
        with open("models/training_stats.json", "w") as f:
            json.dump(stats, f)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save("models/nav_policy_interrupted.pt")

    finally:
        simulation_app.close()

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
