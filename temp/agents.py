import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
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


    def train_long_batch(self, batch_size=256):
        if len(self.memory) > batch_size:
            return
        
        dataset = ExperienceDataset(list(self.memory))
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=6
        )

        for batch in dataloader:
            obs_batch = batch[0].to(self.device, non_blocking=True)
            action_batch = batch[1].to(self.device, non_blocking=True)
            reward_batch = batch[2].to(self.device, non_blocking=True)
            next_obs_batch = batch[3].to(self.device, non_blocking=True)
            done_batch = batch[4].to(self.device, non_blocking=True)

            self._train_batch(obs_batch, action_batch, reward_batch, next_obs_batch, done_batch)

    def _train_batch(self, obs_batch, action_batch, reward_batch, next_obs_batch, done_batch):
        current_q_values = self.model(obs_batch)

        q_values = current_q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_model(next_obs_batch)
            max_next_q_values = next_q_values.max(1)[0]

        target_q_values = reward_batch + (1 - done_batch) * self.gamma * max_next_q_values

        self.loss = self.trainer.optimize_model(q_values, target_q_values)

    def train(self, observation, action, reward, next_observation, done):
        # print("Train!")
        # observation = torch.tensor(observation, dtype=torch.float).to(self.device) / 255.0
        # next_observation = torch.tensor(next_observation, dtype=torch.float).to(self.device) / 255.0
        observation = torch.from_numpy(observation).to(self.device).float() / 255.0
        next_observation = torch.from_numpy(next_observation).to(self.device).float() / 255.0

        q_values = self.get_q_values(observation, self.model)
        q_value = q_values[action]
        # print(q_value)

        target_q_values = self.get_q_values(next_observation, self.target_model)

        target_q_value = torch.tensor(reward, dtype=torch.float).to(self.device)
        # target_q_value = torch.from_numpy(reward).to(self.device)
        target_q_value = reward + (1 - done) * self.gamma * torch.max(target_q_values)

        self.loss = self.trainer.optimize_model(q_value, target_q_value)

    def update_target_model(self):
        state_dict = self.model.state_dict()
        self.target_model.load_state_dict(state_dict)       

    def decay_epsilon(self):
        self.epsilon = max(self.end_epsilon, self.epsilon * self.epsilon_decay)

class ExperienceDataset(Dataset):
    def __init__(self, experiences):
        # self.observations = torch.stack([torch.tensor(exp[0], dtype=torch.float32) for exp in experiences]) / 255.0
        self.actions = torch.tensor([exp[1] for exp in experiences], dtype=torch.long)        
        self.rewards = torch.tensor([exp[2] for exp in experiences], dtype=torch.float32)
        # self.next_observations = torch.stack([torch.tensor(exp[3], dtype=torch.float32) for exp in experiences]) / 255.0
        self.dones = torch.tensor([exp[4] for exp in experiences], dtype=torch.float32)

        self.observations = torch.stack([torch.from_numpy(exp[0]) for exp in experiences]).float() / 255.0
        # self.actions = torch.from_numpy([exp[1] for exp in experiences]).long()
        # self.rewards = torch.from_numpy([exp[2] for exp in experiences]).float()
        self.next_observations = torch.stack([torch.from_numpy(exp[3]) for exp in experiences]).float() / 255.0
        # self.dones = torch.from_numpy([exp[4] for exp in experiences]).float()

    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, index):
        return (
            self.observations[index],
            self.actions[index],
            self.rewards[index],
            self.next_observations[index],
            self.dones[index]
        )