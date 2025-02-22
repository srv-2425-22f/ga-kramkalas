import gymnasium
from vizdoom import gymnasium_wrapper
import numpy as np
import matplotlib.pyplot as plt
import torch
from helper import plot_image_live, plot_loss_live, plot
from agents import Basic

# import tqdm
from models import CNN
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

env = gymnasium.make("VizdoomDeathmatch-v0")

learning_rate = 0.001
n_episodes = 2000
when_show = 1950
start_epsilon = 1
BATCH_SIZE = 32
epsilon_decay = 0.9975
end_epsilon = 0.1
update_frequency = 100

discrete_action_space = int(env.action_space["binary"].n)
continuous_action_space = env.action_space["continuous"].shape[0]

print(f"Disc: {discrete_action_space} | Cont: {continuous_action_space}")
# device = "cuda" if torch.cuda.is_available else "cpu"
device = "cpu"
print(device)

agent = Basic(
    env=env,
    learning_rate=learning_rate,
    start_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    end_epsilon=end_epsilon,
    model=CNN(3, 32, discrete_action_space, continuous_action_space).to(device),
    target_model=CNN(3, 32, discrete_action_space, continuous_action_space).to(device),
    device=device,
)

env = gymnasium.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

loss = []

for episode in range(n_episodes):
    obs, info = env.reset()
    obs["screen"] = obs["screen"].reshape(3, 240, 320)
    done = False
    num_steps = 0
    while not done:
        action = agent.get_action(env, obs["screen"])
        try:
            next_obs, reward, terminated, truncated, info = env.step(action)
        except:
            env.reset()
            break
        next_obs["screen"] = next_obs["screen"].reshape(3, 240, 320)

        done = terminated or truncated

        if num_steps % 4 == 0:
            agent.train(obs["screen"], action, reward, next_obs["screen"], done)

        agent.remember(
            obs["screen"], action, reward, next_obs["screen"], done, agent.memory
        )
        agent.remember(
            obs["screen"],
            action,
            reward,
            next_obs["screen"],
            done,
            agent.episode_memory,
        )

        image = np.array(obs["screen"]).reshape(240, 320, 3)
        if episode > when_show:
            plot_image_live(image, episode)

        obs = next_obs
        num_steps += 1
    agent.train_episode()

    if loss and agent.loss < np.min(loss):
        torch.save(agent.model.state_dict(), "saved_models/best.pth")
    loss.append(agent.loss)

    print(f"Epsiode: {episode}")
    if episode % update_frequency == 0:
        agent.train_long(agent.memory, BATCH_SIZE)
        agent.update_target_model()
        torch.save(agent.model.state_dict(), "saved_models/latest.pth")

    if episode > 200:
        agent.decay_epsilon()

plot(loss, n_episodes)
