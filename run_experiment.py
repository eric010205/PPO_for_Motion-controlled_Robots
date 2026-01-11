import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

ENV_ID = "Pendulum-v1"
TOTAL_TIMESTEPS = 200_000

env = gym.make(ENV_ID)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="results/tensorboard/"
)

model.learn(total_timesteps=TOTAL_TIMESTEPS)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"Evaluation reward: {mean_reward:.2f} ± {std_reward:.2f}")

model.save("results/ppo_pendulum")
env.close()

import numpy as np
import matplotlib.pyplot as plt

obs, _ = env.reset()
rewards = []

for _ in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    rewards.append(reward)
    if terminated or truncated:
        break

plt.plot(np.cumsum(rewards))
plt.title("Cumulative reward during evaluation rollout")
plt.xlabel("Timestep")
plt.ylabel("Cumulative reward")
plt.savefig("results/rollout_behavior.png", dpi=150)
plt.close()
