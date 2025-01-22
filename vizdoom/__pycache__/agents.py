import torch
from torch import nn
import numpy as np
from collections import defaultdict, deque


class Basic:
    def __init__(
        self,
        env,
        learning_rate: float,
        start_epsilon: float,
        epsilon_decay: float,
        end_epsilon: float,
        model: nn.Module,
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
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.end_epsilon = end_epsilon
        self.training_error = []
        self.model = model

    def get_action(self, env, obs: np.ndarray) -> int:
        """
        Returns the best action, according to the agent, with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # print(obs)
        if np.random.random() < self.epsilon:
            return (
                env.action_space.sample()
            )  # Returns a random action from the action_space

        else:
            obs = obs.reshape(3, 240, 320)
            obs = torch.tensor(obs, dtype=torch.float)
            preds = self.model(
                obs
            )  # Returns list of probabilities with size of action_space
            prediction = torch.argmax(preds).item()  # Returns the most probable action
            print(prediction)
            return prediction

    def remember(self, obs, action, reward, next_obs, done):
        self.memory.append((obs, action, reward, next_obs, done))

    def get_Q(self, reward, observation):
        target = (reward + self.gamma * torch.max(self.model(observation))) # Bellman equation to calculate the q-value
        return target

    def train_long(self, memory):
        print(memory)
        observations, actions, rewards, next_observations, done = zip(*memory)

    def decay_epsilon(self):
        self.epsilon = max(self.end_epsilon, self.epsilon - self.epsilon_decay)