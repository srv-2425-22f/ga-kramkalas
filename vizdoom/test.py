import gymnasium
import vizdoom
from vizdoom import gymnasium_wrapper
import numpy as np
import matplotlib.pyplot as plt
import torch
from helper import plot_image_live, plot_loss_live, plot
from agents import Basic
import time

from models import CNN, DQN, ViT
import random

# Set random seeds for reproducibility
torch.cuda.manual_seed(42)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.empty_cache()

# HYPERPARAMETERS (VARIABLES THAT CAN BE MODIFIED)
learning_rate = 0               # Learning rate for the optimizer
n_episodes = 5                  # Total number of training episodes
when_show = 0                   # Episode number when to start showing game visuals
when_decay = 4000               # Episode number when to start epsilon decay
BATCH_SIZE = 32                 # Batch size for training
start_epsilon = 0               # Initial exploration rate
epsilon_decay = 0.995           # Rate at which exploration decreases
end_epsilon = 0.05                 # Minimum exploration rate

# Initialize the VizDoom environment
env = gymnasium.make("VizdoomDeathmatch-v0")
doom_game = env.unwrapped.game
env = gymnasium.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

# Get game variables shape and add one for enemy_on_screen feature and define number of actions possible
game_variables = env.observation_space["gamevariables"].shape
game_variables = game_variables[0] + 1 # +1 för enemy_on_screen
action_space = int(env.action_space.n)

# Determine device for testing
device = "cuda" if torch.cuda.is_available else "cpu"
device = "cpu"
print(device)

# Initialize models
# DQN MODELLEN
# model = DQN(3,32,action_space,game_variables)
# target_model = DQN(3,32,action_space,game_variables)
# ViT MODELLEN
model = ViT((120, 160), 3, action_space, game_variables)
target_model = ViT((120, 160), 3, action_space, game_variables)
# model.load_state_dict(torch.load("D:\GA-kalas\GA\ga-kramkalas\saved_models\\vit_20k_extended_1800_GOOD.pth"))
# target_model.load_state_dict(torch.load("D:\GA-kalas\GA\ga-kramkalas\saved_models\\vit_20k_extended_1800_GOOD.pth"))

# Initialize the RL agent
agent = Basic(
    env=env,
    learning_rate=learning_rate,
    start_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    end_epsilon=end_epsilon,
    model=model.to(device),
    target_model=target_model.to(device),
    device=device,
)

# Synchronize target model with main model
agent.update_target_model()

# Initialize lists for tracking metrics
frag_count = []    # Tracks frags per episode
death_count = []    # Tracks deaths per episode
loss = []          # Tracks loss per episode
rewards = []       # Tracks total reward per episode

# Testing loop
start_time = time.time()
print(f"Started training at: {time.localtime(start_time)}")
for episode in range(n_episodes):
    obs, info = env.reset()
    obs["screen"] = obs["screen"].reshape(3, 120, 160)
    enemy_on_screen = agent.enemies_on_screen()
    obs["gamevariables"] = np.append(obs["gamevariables"], enemy_on_screen)

    done = False
    num_steps = 0
    total_loss = 0
    total_reward = 0
    num_train_steps = 0
    frags = 0
    deaths = 0

    # Episode loop
    while not done:
        action = agent.get_action(obs)
        try:
            next_obs, reward, terminated, truncated, info = env.step(action)          
        except:
            env.reset()
            break    
        
        frag = obs["gamevariables"][0]
        frags += (frag)
        
        deaths += obs["gamevariables"][1]+1    

        enemy_on_screen = agent.enemies_on_screen()
        if enemy_on_screen and action == 5:
            reward += 50
        else:
            reward = -1

        total_reward += reward

        # Process next observation
        next_obs["gamevariables"] = np.append(next_obs["gamevariables"], enemy_on_screen)             
        next_obs["screen"] = next_obs["screen"].reshape(3, 120, 160)
        done = terminated or truncated

        if num_steps % 4 == 0:
            agent.train(obs, action, reward, next_obs, done)
            total_loss += agent.loss
            num_train_steps += 1

        # Store experience in replay memory
        agent.remember(obs, action, reward, next_obs, done)

        image = np.array(obs["screen"]).reshape(120, 160, 3)
        if episode > when_show:
            plot_image_live(image, episode)

        obs = next_obs
        num_steps += 1   

        if(obs["gamevariables"][0]):
            print("Killed an enemy")
           
    frag_count.append(frags)
    death_count.append(deaths)
    rewards.append(total_reward)

    if loss and agent.loss < np.min(loss):
        pass
    if num_train_steps > 0:
        avg_loss = total_loss / num_train_steps
    else:
        avg_loss = 0
    loss.append(avg_loss)       

    if episode > when_decay:
        agent.decay_epsilon()

    print(f"Episode: {episode}")

# Testing complete
end_time = time.time()
print(f"Started testing at: {time.localtime(start_time)}")
print(f"Stopped testing at: {time.localtime(end_time)}")
print(f"Total time testing: {(end_time - start_time)/60} min")
print(frag_count)
plot(n_episodes, loss, rewards, frag_count)
