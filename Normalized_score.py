import gym
import d4rl
import numpy as np

# 1. Create the environment
env = gym.make('halfcheetah-medium-v2')  # You can change to medium-replay-v2, medium-expert-v2, etc.

# 2. Run some episodes (this example just uses a random policy)
def evaluate_random_policy(env, episodes=10):
    total_returns = []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            total_reward += reward
        total_returns.append(total_reward)
    return np.mean(total_returns)

# 3. Get raw return (average total reward per episode)
raw_score = evaluate_random_policy(env)
print("Raw return:", raw_score)

# 4. Convert to D4RL normalized score
normalized_score = env.get_normalized_score(raw_score)
print("Normalized score:", normalized_score)
