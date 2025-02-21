import torch
from torch import nn
import numpy as np
from collections import defaultdict, deque
from models import QTrainer
import random

class Basic:
    def __init__(
        self,
        env,
        learning_rate: float,
        start_epsilon: float,
        epsilon_decay: float,
        end_epsilon: float,
        model: nn.Module,
        target_model: nn.Module,
        device: str,
        gamma: float = 0.99,
    ):
        """
        Initialize a Reinforcement Learning agent with an empty dictionary of
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            start_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            end_epsilon: The final epsilon value
            gamma: The discount factor for computing the Q-value
        """

        self.memory = deque(maxlen=1_000_000)
        self.episode_memory = deque()
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.end_epsilon = end_epsilon
        self.training_error = []
        self.model = model.to(device)
        self.target_model = target_model.to(device)
        self.trainer = QTrainer(self.model, self.lr)
        self.loss = 0
        self.device = device

    def get_action(self, env, obs: np.ndarray) -> int:
        """
        Returns the best action, according to the agent, with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # print(obs)
        if np.random.random() < self.epsilon:
            # print("\nRandom action")
            # print(env.action_space["binary"].sample().dtype)
            return (
                env.action_space.sample() # FÖR DEATHMATCH MÅSTE MAN SKRIVA "binary" HÄR
            )  # Returns a random action from the action_space

        else:
            # print("\nPredicted action")
            obs = torch.tensor(obs, dtype=torch.float).to(self.device)
            preds = self.model(obs)  # Returns list of probabilities with size of action_space
            preds = torch.softmax(preds, dim=0)
            prediction = torch.argmax(preds).item()  # Returns the most probable action, represented by the index of the action space
            return prediction

    def remember(self, obs, action, reward, next_obs, done, memory: deque):
        memory.append((obs, action, reward, next_obs, done))

    # def get_q_values(self, next_observation, reward):
    #     # pred = torch.argmax(self.model(observation)).float()
    #     target = (reward + self.gamma * torch.max(self.model(next_observation))) # Bellman equation to calculate the target q-value
    #     return target

    def get_q_values(self, observation, model: nn.Module):
        q_value = model(observation)
        return q_value

    def train_long(self, memory: deque, batch_size):
        # print("Train long!")

        if len(memory) > batch_size:
            random_samples = random.sample(memory, batch_size)
        else:
            random_samples = list(memory)

        for observation, action, reward, next_observation, done in random_samples:
            # sample = [sample]
            # print(f"sample: {type(sample)}")
            # observation, action, reward, next_observation = zip(*sample)

            observation = np.array(observation)
            next_observation = np.array(next_observation)
            # print(f"observation: {type(observation)}\naction: {type(action)}\nreward: {type(reward)}\nnext_observation: {type(next_observation)}\n")

            self.train(observation, action, reward, next_observation, done)

    def train_episode(self):
        sample = list(self.episode_memory)
        for observation, action, reward, next_observation, done in sample:
            # sample = [sample]
            # print(f"sample: {type(sample)}")
            # observation, action, reward, next_observation = zip(*sample)

            observation = np.array(observation)
            next_observation = np.array(next_observation)
            # print(f"observation: {type(observation)}\naction: {type(action)}\nreward: {type(reward)}\nnext_observation: {type(next_observation)}\n")

            self.train(observation, action, reward, next_observation, done)
        self.episode_memory.clear()


    # def train_short(self, pred, next_observation, reward):
    #     # observation = torch.from_numpy(observation).float()
    #     next_observation = torch.from_numpy(next_observation).float()
    #     # print(observation.type(), observation.shape)s
    #     target = self.get_q_values(next_observation, reward)
    #     self.trainer.optimize_model(pred, target)
    #     # print(f"Pred: {pred} | Target: {target}")

    def train(self, observation, action, reward, next_observation, done):
        # print("Train!")
        observation = torch.tensor(observation, dtype=torch.float).to(self.device)
        next_observation = torch.tensor(next_observation, dtype=torch.float).to(self.device)

        q_values = self.get_q_values(observation, self.model)
        q_value = q_values[action]
        # print(q_value)

        target_q_values = self.get_q_values(next_observation, self.target_model)

        target_q_value = torch.tensor(reward, dtype=torch.float).to(self.device)
        if not done:
            target_q_value += self.gamma * torch.max(target_q_values)

        self.loss = self.trainer.optimize_model(q_value, target_q_value)

    def update_target_model(self):
        state_dict = self.model.state_dict()
        self.target_model.load_state_dict(state_dict)       

    def decay_epsilon(self):
        self.epsilon = max(self.end_epsilon, self.epsilon * self.epsilon_decay)