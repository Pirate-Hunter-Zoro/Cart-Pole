import math
import torch
import gymnasium as gym
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer, Transition
from dqn import DQN
import random
import os
import datetime

discount_factor = 0.95
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 10000
batch_size = 64
learning_rate = 5e-4
replay_buffer_size = 10000
target_update_frequency = 5
tau = 0.005 # Soft update parameter for target network

steps_done = 0

epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * steps_done / epsilon_decay)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, -0.01, 0.01)
        nn.init.constant_(m.bias, 0)

state_dim = 4  # Cart position, cart velocity, pole angle, pole angular velocity
action_dim = 2  # Move left or right
policy_dqn = DQN(state_dim, action_dim)
target_dqn = DQN(state_dim, action_dim)
policy_dqn.apply(init_weights)
target_dqn.load_state_dict(policy_dqn.state_dict())

replay_buffer = ReplayBuffer(replay_buffer_size)
optimizer = optim.Adam(policy_dqn.parameters(), lr=learning_rate)

num_episodes = 20000
env = gym.make('CartPole-v1')

def normalize_state(state):
    return np.clip(state, -10, 10) / 10.0

from collections import deque

reward_window = deque(maxlen=100)

episode_rewards = []
episode_losses = []

for episode in range(num_episodes):
    """
    Deep Q-Learning loop for CartPole-v1.
    - Uses epsilon-greedy exploration, experience replay, and a target network.
    - Optimizes with Huber loss (smooth L1).
    - Syncs the target network every N episodes.
    """
    state, _ = env.reset()
    done = False

    episode_reward = 0
    episode_loss = 0

    while not done:
        steps_done += 1
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * steps_done / epsilon_decay)

        if random.random() > epsilon:
            with torch.no_grad():
                # Select action according to policy
                action = policy_dqn(torch.tensor(state, dtype=torch.float32)).max(0)[1].item()
        else: # Select random action
            action = env.action_space.sample()

        next_state, reward, done, _, _ = env.step(action)
        state = normalize_state(state)
        next_state = normalize_state(next_state)
        episode_reward += reward
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state

        if len(replay_buffer) > batch_size:
            transitions = replay_buffer.sample(batch_size)
            batch = Transition(*zip(*transitions))

            # Identify which transitions have non-terminal next states
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
            
            # Extract next states for the non-terminal transitions
            non_final_next_states = torch.stack([torch.tensor(normalize_state(s), dtype=torch.float32) for s in batch.next_state if s is not None])
            state_batch = torch.stack([torch.tensor(normalize_state(s), dtype=torch.float32) for s in batch.state])
            action_batch = torch.stack([torch.tensor(a, dtype=torch.long) for a in batch.action])
            reward_batch = torch.stack([torch.tensor(r, dtype=torch.float32) for r in batch.reward])

            # Compute Q values - note that we need to add the batch dimension to the action_batch
            # Compute Q-values for all actions in each state in the batch
            all_action_values = policy_dqn(state_batch)
            # Extract the Q-values corresponding to the actions actually taken
            state_action_values = all_action_values.gather(1, action_batch.unsqueeze(1)).squeeze()

            next_state_values = torch.zeros(batch_size)
            # Only compute next Q-values for non-terminal states

            # The following three steps are double dqn - let the POLICY network pick the next actions, and the target evaluate the next states based on those actions

            # Action selection with policy DQN
            next_actions = policy_dqn(non_final_next_states).argmax(1)

            # Action evaluation with target DQN
            target_qs = target_dqn(non_final_next_states).gather(1, next_actions.unsqueeze(1)).squeeze().detach()

            # Apply to non-terminal indices
            next_state_values[non_final_mask] = target_qs

            # Terminal state Q-values are already zero-initialized, 
            # so we only overwrite values for non-terminal transitions
            expected_state_action_values = reward_batch + discount_factor * next_state_values
            expected_state_action_values = torch.clamp(expected_state_action_values, -100.0, 100.0)
            
            if episode % 100 == 0 and steps_done % 1000 == 0:
                with torch.no_grad():
                    max_q = policy_dqn(state_batch).max().item()
                    print(f"Episode {episode} | Max Q: {max_q:.2f} | Loss: {episode_loss:.2f} | Epsilon: {epsilon:.2f}")
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
            episode_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), 1.0)
            optimizer.step()
    
    episode_rewards.append(episode_reward)
    episode_losses.append(episode_loss)

    if episode % 500 == 0:
        avg_reward = np.mean(reward_window) if len(reward_window) > 0 else 0
        print(f"Episode {episode}, Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {epsilon:.2f}")
        
    # Track episode reward
    reward_window.append(episode_reward)

    # Early stopping check
    if len(reward_window) == 100 and np.mean(reward_window) >= 475:
        print(f"Solved at episode {episode}! 🎉")
        torch.save({
            'policy_dqn_state_dict': policy_dqn.state_dict(),
            'target_dqn_state_dict': target_dqn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, 'cartpole_dqn_checkpoint.pth')
        break

    # Update target network
    if episode % target_update_frequency == 0:
        with torch.no_grad():
            for target_param, param in zip(target_dqn.parameters(), policy_dqn.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

# Save even if early stopping wasn't triggered
torch.save({
    'policy_dqn_state_dict': policy_dqn.state_dict(),
    'target_dqn_state_dict': target_dqn.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, 'cartpole_dqn_checkpoint.pth')

# Plot the episode rewards
import matplotlib.pyplot as plt
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.title('Episode Rewards over Time')
plt.savefig('episode_rewards.png')

# Plot the episode losses
plt.plot(episode_losses)
plt.xlabel('Episode')
plt.ylabel('Episode Loss')
plt.title('Episode Losses over Time')
plt.savefig('episode_losses.png')

video_dir = f"videos/run_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
os.makedirs(video_dir, exist_ok=True)

eval_env = gym.make("CartPole-v1", render_mode="rgb_array")
eval_env = gym.wrappers.RecordVideo(
    eval_env,
    video_folder=video_dir,
    episode_trigger=lambda _: True,
    name_prefix="cartpole_eval",
    disable_logger=True,
)

# Load the saved model
checkpoint = torch.load('cartpole_dqn_checkpoint.pth')
policy_dqn.load_state_dict(checkpoint['policy_dqn_state_dict'])
num_eval_episodes = 10
eval_rewards = []

for episode in range(num_eval_episodes):
    state, _ = eval_env.reset()
    done = False
    total_reward = 0

    while not done:
        with torch.no_grad():
            action = policy_dqn(torch.tensor(state, dtype=torch.float32)).argmax().item()

        state, reward, done, _, _ = eval_env.step(action)
        total_reward += reward

    eval_rewards.append(total_reward)

avg_eval_reward = np.mean(eval_rewards)
print(f"\nAverage Evaluation Reward over {num_eval_episodes} episodes: {avg_eval_reward:.2f}")

# Plot the evalutation rewards
plt.plot(eval_rewards)
plt.xlabel('Evaluation Episode')
plt.ylabel('Episode Reward')
plt.title('Evaluation Episode Rewards')
plt.savefig('evaluation_rewards.png')

env.close()
eval_env.close()