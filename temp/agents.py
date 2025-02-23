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

        self.memory = deque(maxlen=100_000)
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
        if np.random.random() < self.epsilon:
            return {
            "binary": env.action_space["binary"].sample(),
            "continuous": env.action_space["continuous"].sample()
        }

        else:
            obs = torch.tensor(obs, dtype=torch.float).to(self.device)
            discrete_preds, continuous_preds = self.model(obs)  # Returns list of probabilities with size of action_space
            discrete_pred = torch.argmax(torch.softmax(discrete_preds, dim=0)).item() # Returns the most probable action, represented by the index of the action space
            continuous_preds = continuous_preds.detach().cpu().numpy()
            return {
            "binary": discrete_pred,
            "continuous": continuous_preds  # Adjust if needed
        }
            # return prediction

    def remember(self, obs, action, reward, next_obs, done, memory: deque):
        memory.append((obs, action, reward, next_obs, done))

    # def get_q_values(self, next_observation, reward):
    #     # pred = torch.argmax(self.model(observation)).float()
    #     target = (reward + self.gamma * torch.max(self.model(next_observation))) # Bellman equation to calculate the target q-value
    #     return target

    def get_q_values(self, observation, model: nn.Module):
        discrete_q_value, continuous_q_value = model(observation)
        # print(discrete_q_value, continuous_q_value)
        return discrete_q_value, continuous_q_value

    def train_long(self, memory: deque, batch_size):

        if len(memory) > batch_size:
            random_samples = random.sample(memory, batch_size)
        else:
            random_samples = list(memory)

        for observation, action, reward, next_observation, done in random_samples:
            observation = np.array(observation)
            next_observation = np.array(next_observation)

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

    def train_dataset(self, dataset, dataloader, batch_size=32):
        if len(dataset) >= batch_size:
            for batch in dataloader:
                obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = batch

                # Move data to GPU
                obs_batch = obs_batch.to(self.device)
                action_batch = action_batch.to(self.device)
                reward_batch = reward_batch.to(self.device)
                next_obs_batch = next_obs_batch.to(self.device)
                done_batch = done_batch.to(self.device)

                # Train the agent
                # self.train(obs_batch, action_batch, reward_batch, next_obs_batch, done_batch)
    
    def train_batch(self, observation_batch, action_batch, reward_batch, next_observation_batch, done_batch):
        observation_batch = torch.tensor(observation_batch, dtype=torch.float).to(self.device)
        next_observation_batch = torch.tensor(next_observation_batch, dtype=torch.float).to(self.device)
        action_batch = torch.tensor(action_batch["binary"], dtype=torch.long).to(self.device)  # Extract discrete actions
        reward_batch = torch.tensor(reward_batch, dtype=torch.float).to(self.device)
        done_batch = torch.tensor(done_batch, dtype=torch.float).to(self.device)

        # Get Q-values for the current observations
        discrete_q_values, continuous_q_values = self.get_q_values(observation_batch, self.model)

        # Select Q-values for the taken actions
        discrete_q_value = discrete_q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)  # Shape: [batch_size]

        # Get Q-values for the next observations
        target_discrete_q_values, target_continuous_q_values = self.get_q_values(next_observation_batch, self.target_model)

        # Compute target Q-values using the Bellman equation
        target_q_value = reward_batch + (1 - done_batch) * self.gamma * target_discrete_q_values.max(1)[0]  # Shape: [batch_size]

        # Compute the loss
        self.loss = self.trainer.optimize_model(discrete_q_value, continuous_q_values, target_q_value, target_continuous_q_values)


    # def train_short(self, pred, next_observation, reward):
    #     # observation = torch.from_numpy(observation).float()
    #     next_observation = torch.from_numpy(next_observation).float()
    #     # print(observation.type(), observation.shape)s
    #     target = self.get_q_values(next_observation, reward)
    #     self.trainer.optimize_model(pred, target)
    #     # print(f"Pred: {pred} | Target: {target}")

    def train(self, observation, action, reward, next_observation, done):
        observation = torch.tensor(observation, dtype=torch.float).to(self.device)
        next_observation = torch.tensor(next_observation, dtype=torch.float).to(self.device)

        discrete_q_values, continuous_q_values = self.get_q_values(observation, self.model)

        discrete_q_value = discrete_q_values[action["binary"]]

        target_discrete_q_values, target_continuous_q_values = self.get_q_values(next_observation, self.target_model)

        target_q_value = torch.tensor(reward, dtype=torch.float).to(self.device)
        if not done:
            target_q_value += self.gamma * torch.max(target_discrete_q_values)

        self.loss = self.trainer.optimize_model(discrete_q_value, continuous_q_values, target_q_value, target_continuous_q_values)

    def update_target_model(self):
        state_dict = self.model.state_dict()
        self.target_model.load_state_dict(state_dict)       

    def decay_epsilon(self):
        self.epsilon = max(self.end_epsilon, self.epsilon * self.epsilon_decay)