import gymnasium
from vizdoom import gymnasium_wrapper
import numpy as np
import matplotlib.pyplot as plt
import torch
from helper import plot_image_live
from agents import Basic
import tqdm

env = gymnasium.make("VizdoomBasic-v0")
# observation, info = env.reset()
# image = np.array(observation["screen"])
# print(image.shape) # (240, 320, 3)

learning_rate = 0.01
n_episodes = 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
end_epsilon = 0.1

agent = Basic(
    env=env,
    learning_rate=learning_rate,
    start_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    end_epsilon=end_epsilon
)

env = gymnasium.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    while not done:
        action = agent.get_action(env, obs["screen"])
        next_obs, reward, terminated, truncated, info = env.step(action)

        agent.update(obs["screen"], action, reward, terminated, next_obs["screen"])

        image = np.array(obs["screen"])
        plot_image_live(image)

        done = terminated or truncated
        obs = next_obs["screen"]

    agent.decay_epsilon()