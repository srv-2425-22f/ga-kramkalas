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
        gamma: float = 0.95,
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

        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.memory = deque(maxlen=100_000)
        self.memory_episode = deque()
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
            return (
                env.action_space.sample()
            )  # Returns a random action from the action_space

        else:
            # print("\nPredicted action")
            obs = torch.tensor(obs, dtype=torch.float).to(self.device)
            preds = self.model(obs)  # Returns list of probabilities with size of action_space
            prediction = torch.argmax(preds).item()  # Returns the most probable action, represented by the index of the action space
            return prediction

    def remember(self, obs, action, reward, next_obs, memory: deque):
        memory.append((obs, action, reward, next_obs))

    # def get_q_values(self, next_observation, reward):
    #     # pred = torch.argmax(self.model(observation)).float()
    #     target = (reward + self.gamma * torch.max(self.model(next_observation))) # Bellman equation to calculate the target q-value
    #     return target

    def get_q_values(self, observation, model: nn.Module):
        q_value = model(observation)
        return q_value.squeeze()

    def train_long(self, memory: np.ndarray, batch_size):
        if len(memory) > batch_size:
            random_samples = random.sample(memory, batch_size)
        else:
            random_samples = memory           

        for observation, action, reward, next_observation in random_samples:
            observation = np.array(observation)
            next_observation = np.array(next_observation)

            self.train(observation, action, reward, next_observation)

    def train(self, observation, action, reward, next_observation):
        # print("Train!")
        observation = torch.tensor(observation, dtype=torch.float).to(self.device)
        next_observation = torch.tensor(next_observation, dtype=torch.float).to(self.device)

        if len(observation.shape) == 3:
            observation = observation.unsqueeze(0)
        if len(next_observation.shape) == 3:
            next_observation = next_observation.unsqueeze(0)

        q_values = self.get_q_values(observation, self.model)
        # print("Shape: ", q_values.shape)
        # print("q_value:", q_values)
        q_value = q_values[action]

        target_q_values = self.get_q_values(next_observation, self.target_model)
        target_q_value = self.gamma * torch.max(target_q_values)
        target_q_value += reward

        self.loss = self.trainer.optimize_model(q_value, target_q_value)

    def train_episode(self):
        random_samples = list(self.memory_episode)

        for observation, action, reward, next_observation in random_samples:
            observation = np.array(observation)
            next_observation = np.array(next_observation)

            self.train(observation, action, reward, next_observation)

    def update_target_model(self):
        state_dict = self.model.state_dict()
        self.target_model.load_state_dict(state_dict)       

    def decay_epsilon(self):
        self.epsilon = max(self.end_epsilon, self.epsilon - self.epsilon_decay)