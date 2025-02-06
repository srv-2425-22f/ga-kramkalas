import gymnasium
from vizdoom import gymnasium_wrapper
import numpy as np
import matplotlib.pyplot as plt
import torch
from helper import plot_image_live, plot_loss_live, plot
from agents import Basic
import tqdm
from models import CNN
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

env = gymnasium.make("VizdoomBasic-v0")
# observation, info = env.reset()
# image = np.array(observation["screen"])
# print(image.shape) # (240, 320, 3)

learning_rate = 0.01
n_episodes = 200
start_epsilon = 1.0
BATCH_SIZE = 512
epsilon_decay = 0.95
end_epsilon = 0.1
action_space = int(env.action_space.n)
device = "cuda" if torch.cuda.is_available else "cpu"
print(device)

agent = Basic(
    env=env,
    learning_rate=learning_rate,
    start_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    end_epsilon=end_epsilon,
    model=CNN(3, 32, action_space).to(device),
    target_model=CNN(3, 32, action_space).to(device),
    device=device
)

env = gymnasium.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

loss = []


for episode in range(n_episodes):
    print(f"Epsiode: {episode}")
    obs, info = env.reset()
    obs["screen"] = obs["screen"].reshape(3,240,320)
    done = False     
    while not done:
        # obs = np.array(obs["screen"])
        action = agent.get_action(env, obs["screen"])
        try:
            next_obs, reward, terminated, truncated, info = env.step(action)
        except:
            env.reset()
            break
        next_obs["screen"] = next_obs["screen"].reshape(3,240,320)
        # print(next_obs["screen"])
        # agent.update(obs["screen"], action, reward, terminated, next_obs["screen"]       
        done = terminated or truncated

        agent.train(obs["screen"], action, reward, next_obs["screen"], done)
        agent.remember(obs["screen"], action, reward, next_obs["screen"], done)

        image = np.array(obs["screen"]).reshape(240, 320, 3)
        if episode > 150:
            plot_image_live(image, episode)
        
        obs = next_obs


    agent.train_long(agent.memory, BATCH_SIZE)
    loss.append(agent.loss)

    if episode % 20 == 0:
        print("Update target model")
        agent.update_target_model()

    agent.decay_epsilon()

plot(loss, n_episodes)