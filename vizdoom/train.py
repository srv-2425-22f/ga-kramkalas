import gymnasium
from vizdoom import gymnasium_wrapper
import numpy as np
import matplotlib.pyplot as plt
import torch
from helper import plot_image_live, plot_loss_live, plot
from agents import Basic
import time

# import tqdm
from models import CNN, DQN, ViT
import random
# from torch.profiler import profile, record_function, ProfilerActivity # import profiler

torch.cuda.manual_seed(42)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

env = gymnasium.make("VizdoomDeathmatch-v0")
doom_game = env.unwrapped.game
# observation, info = env.reset()
# image = np.array(observation["screen"])
# print(image.shape) # (120, 160, 3)

learning_rate = 0.005
n_episodes = 150_000
when_show = 150_000
when_decay = 50_000
start_epsilon = 1
BATCH_SIZE = 32
epsilon_decay = 0.9999
end_epsilon = 0.05
update_frequency = 2_000
action_space = int(env.action_space.n)

device = "cuda" if torch.cuda.is_available else "cpu"
# device = "cpu"
print(device)
game_variables = env.observation_space["gamevariables"].shape
game_variables = game_variables[0] + 1 # +1 för enemy_on_screen

env = gymnasium.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

model = DQN(3,32,action_space,game_variables)
# model = ViT((120, 160), 3, action_space, game_variables)
# model.load_state_dict(torch.load("saved_models/skjut_60.pth"))
target_model = DQN(3,32,action_space,game_variables)
# target_model = ViT((120, 160), 3, action_space, game_variables)
# target_model.load_state_dict(torch.load("saved_models/skjut_60.pth"))


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

agent.update_target_model()

frag_count = []
death_count = []
loss = []
rewards = []

start_time = time.time()
print(f"Started training at: {time.localtime(start_time)}")
# with profile(
#     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#     record_shapes=True,
#     with_stack=True,
#     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2)
# ) as prof:
for episode in range(n_episodes):
    # print(f"\nEpisode: {episode}")7
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

    # agent.model.initialize_hidden(batch_size=32)
    # agent.target_model.initialize_hidden(batch_size=32)

    while not done:
        action = agent.get_action(obs)
        try:
            next_obs, reward, terminated, truncated, info = env.step(action)
        except:
            env.reset()
            break        

        frag = obs["gamevariables"][0]
        frags += (frag)

        enemy_on_screen = agent.enemies_on_screen()
        if enemy_on_screen and action == 5:
            reward += 50
        else:
            reward = -1
        
        total_reward += reward

        # print(f"\n\nEnemy on screen during loop? {enemy_on_screen}")
        next_obs["gamevariables"] = np.append(next_obs["gamevariables"], enemy_on_screen)     
        
        next_obs["screen"] = next_obs["screen"].reshape(3, 120, 160)
        done = terminated or truncated

        if num_steps % 4 == 0:
            agent.train(obs, action, reward, next_obs, done)
            total_loss += agent.loss
            num_train_steps += 1
            # print(f"Loss: {agent.loss}")

        agent.remember(
            obs, action, reward, next_obs, done
        )

        image = np.array(obs["screen"]).reshape(120, 160, 3)
        if episode > when_show:
            # print(f"Enemy on screen before sending to plot? {enemy_on_screen}")
            plot_image_live(image, episode)

        obs = next_obs
        num_steps += 1

        frags += obs["gamevariables"][0]
        deaths += obs["gamevariables"][1]+1

    frag_count.append(frags)
    death_count.append(deaths)
    rewards.append(total_reward)
    # print(frag_count, death_count)

    if loss and agent.loss < np.min(loss):
        torch.save(agent.model.state_dict(), "saved_models/nu_snela_skjut_best.pth")
        pass
    if num_train_steps > 0:
        avg_loss = total_loss / num_train_steps
    else:
        avg_loss = 0
    loss.append(avg_loss)       

    if episode % update_frequency == 0 and episode > 0: 
        print(f"Training @ episode: {episode}")
        agent.train_long(BATCH_SIZE)
        agent.update_target_model()
        torch.save(agent.model.state_dict(), f"saved_models/nu_snela_skjut_{episode}.pth")

    if episode > when_decay:
        agent.decay_epsilon()

    # prof.step()

end_time = time.time()
print(f"Started training at: {time.localtime(start_time)}")
print(f"Stopped training at: {time.localtime(end_time)}")
print(f"Total time trained: {(end_time - start_time)/60} min")

plot(n_episodes, loss, rewards, frag_count)  

# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))