import gymnasium
from vizdoom import gymnasium_wrapper
import numpy as np
import matplotlib.pyplot as plt
import torch
from helper import plot_image_live
from agents import Basic
import tqdm
from models import CNN

# torch.manual_seed(42)
# np.random.seed(42)
env = gymnasium.make("VizdoomBasic-v0")
# observation, info = env.reset()
# image = np.array(observation["screen"])
# print(image.shape) # (240, 320, 3)

learning_rate = 0.01
n_episodes = 200
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
end_epsilon = 0.1
action_space = int(env.action_space.n)
print(type(action_space))
device = "cuda" if torch.cuda.is_available else "cpu"
device = "cpu"
print(device)

agent = Basic(
    env=env,
    learning_rate=learning_rate,
    start_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    end_epsilon=end_epsilon,
    model=CNN(3, 16, action_space).to(device)
)

env = gymnasium.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
for episode in range(n_episodes):
    obs, info = env.reset()
    obs["screen"] = obs["screen"].reshape(3,240,320)
    done = False
    # image = np.array(obs["screen"])
    # plot_image_live(image)

    while not done:
        # obs = np.array(obs["screen"])
        action = agent.get_action(env, obs["screen"])
        for _ in range(2):
            next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs["screen"] = next_obs["screen"].reshape(3,240,320)
        # print(next_obs["screen"])
        # agent.update(obs["screen"], action, reward, terminated, next_obs["screen"])

        image = np.array(obs["screen"]).reshape(240, 320, 3)
        plot_image_live(image, episode)

        agent.train(obs["screen"], action, reward, next_obs["screen"])
        agent.remember(obs["screen"], action, reward, next_obs["screen"])
        
        obs = next_obs

        done = terminated or truncated

    if episode % 10 == 0:
        agent.update_target_model()

    agent.decay_epsilon()

# agent.train_long(agent.memory)