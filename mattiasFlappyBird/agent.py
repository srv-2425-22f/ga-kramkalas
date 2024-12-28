# IMPORTS
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from collections import deque
import random

from game_flappy import FlappyBirdAI
from model import CNN_QNet, QTrainer

# SETUP
device = "cuda" if torch.cuda.is_available else "cpu"
print(f"Using device: {device}")

# HYPERVARIABLES

EPSILON = 0
GAMMA = 0.9
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
torch.manual_seed(42)

# AGENT CLASS
class Agent:
    def __init__(self, device):
        self.n_games = 0
        self.epsilon = EPSILON
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = CNN_QNet(input_size=3, hidden_size=32, output_size=1, device=device).to(device)
        self.trainer = QTrainer(self.model, lr=LR, gamma=GAMMA, device=device)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(
            (state, action, reward, next_state, done)
        )
    
    def train_long_mem(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_mem(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        self.epsilon = 80-self.n_games
        final_move = 0
        if random.randint(0, 200) < self.epsilon:
            final_move = random.randint(-3, 1)
            final_move = 0 if final_move < 1 else 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(device)
            if len(state0.shape) == 3:
                state0 = torch.unsqueeze(state0, dim=0)
            pred = self.model(state0)
            move = torch.argmax(pred).item()
            final_move = move
        return final_move
    
import time
    
def train():
    total_score = 0
    record = 0
    agent = Agent(device)
    game = FlappyBirdAI()
    i = 0

    while True:
        state_old = game.get_state_rgb()

        final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)
        # print(i)
        # print(reward, done, score)
        # time.sleep(0.15)
        state_new = game.get_state_rgb()

        agent.train_short_mem(
            state=state_old,
            action=final_move,
            reward=reward,
            next_state=state_new,
            done=done
        )

        agent.remember(
            state=state_old,
            action=final_move,
            reward=reward,
            next_state=state_new,
            done=done
        )

        if done:
            # print("In done")
            game.reset()
            agent.n_games += 1
            agent.train_long_mem()

            if score > record:
                record = score
                # agent.model.save()

            print(f"Game: {agent.n_games} | Score: {score}, Record: {record}")
        i += 1

if __name__ == "__main__":
    train()
    # model = CNN_QNet(input_size=3, hidden_size=256, output_size=1)
    # dummy_input = torch.randn(1, 3, 288, 512)
    # cnn_output = model(dummy_input)
    # print(cnn_output.shape)
    # game = FlappyBirdAI()
    # state = game.get_state_rgb()
    # state = torch.tensor(state, dtype=torch.float32)
    # state_un = state.unsqueeze(dim=0)
    # print(len(state.shape))
    # print(len(state_un.shape))
    
    # pred = model(state_un)
    # print(pred)