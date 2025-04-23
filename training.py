import math
import torch
import gymnasium as gym
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer, Transition
from dqn import DQN
import random

discount_factor = 0.99
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 500
batch_size = 64
learning_rate = 1e-3
replay_buffer_size = 10000
target_update_frequency = 10

steps_done = 0

epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * steps_done / epsilon_decay)

state_dim = 4  # Cart position, cart velocity, pole angle, pole angular velocity
action_dim = 2  # Move left or right
policy_dqn = DQN(state_dim, action_dim)
target_dqn = DQN(state_dim, action_dim)
target_dqn.load_state_dict(policy_dqn.state_dict())

replay_buffer = ReplayBuffer(replay_buffer_size)
optimizer = optim.Adam(policy_dqn.parameters(), lr=learning_rate)

num_episodes = 1000
env = gym.make('CartPole-v1')

from collections import deque

reward_window = deque(maxlen=100)

episode_rewards = []

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
        episode_reward += reward
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state

        if len(replay_buffer) > batch_size:
            transitions = replay_buffer.sample(batch_size)
            batch = Transition(*zip(*transitions))

            # Identify which transitions have non-terminal next states
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
            
            # Extract next states for the non-terminal transitions
            non_final_next_states = torch.cat([torch.tensor(s, dtype=torch.float32) for s in batch.next_state if s is not None])
            state_batch = torch.stack([torch.tensor(s, dtype=torch.float32) for s in batch.state])
            action_batch = torch.stack([torch.tensor(a, dtype=torch.long) for a in batch.action])
            reward_batch = torch.stack([torch.tensor(r, dtype=torch.float32) for r in batch.reward])

            # Compute Q values - note that we need to add the batch dimension to the action_batch
            # Compute Q-values for all actions in each state in the batch
            all_action_values = policy_dqn(state_batch)
            # Extract the Q-values corresponding to the actions actually taken
            state_action_values = all_action_values.gather(1, action_batch.unsqueeze(1)).squeeze()

            next_state_values = torch.zeros(batch_size)
            # Only compute next Q-values for non-terminal states
            # Use .detach() to avoid tracking gradients through the target network
            next_state_values[non_final_mask] = target_dqn(non_final_next_states).max(1)[0].detach()

            # Terminal state Q-values are already zero-initialized, 
            # so we only overwrite values for non-terminal transitions
            expected_state_action_values = (next_state_values * discount_factor) + reward_batch

            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    episode_rewards.append(episode_reward)

    if episode % 10 == 0:
        avg_reward = np.mean(reward_window) if reward_window else 0
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

    if episode % target_update_frequency == 0:
        target_dqn.load_state_dict(policy_dqn.state_dict())

# Plot the episode rewards
import matplotlib.pyplot as plt
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.title('Episode Rewards over Time')
plt.savefig('episode_rewards.png')

# Evaluation
eval_env = gym.make("CartPole-v1", render_mode="rgb_array")
eval_env = gym.wrappers.RecordVideo(
    eval_env,
    video_folder="videos/",
    episode_trigger=lambda _: True,
    name_prefix="cartpole_eval"
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
    print(f"Evaluation Episode {episode + 1}: Reward = {total_reward}")

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