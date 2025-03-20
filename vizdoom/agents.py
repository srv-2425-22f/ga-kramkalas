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
        """_summary_

        Args:
            env (_type_): _description_
            learning_rate (float): _description_
            start_epsilon (float): _description_
            epsilon_decay (float): _description_
            end_epsilon (float): _description_
            model (nn.Module): _description_
            target_model (nn.Module): _description_
            device (str): _description_
            gamma (float, optional): _description_. Defaults to 0.99.
        """      

        self.env = env
        self.memory = deque(maxlen=100_000)
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.end_epsilon = end_epsilon
        self.model = model.to(device)
        self.target_model = target_model.to(device)
        self.trainer = QTrainer(self.model, self.lr)
        self.loss = 0
        self.device = device
        self.possible_enemies = [
            "Zombieman", "ShotgunGuy", "ChaingunGuy", "Imp", "Demon", "Spectre",
            "LostSoul", "Cacodemon", "HellKnight", "BaronOfHell", "Arachnotron",
            "Mancubus", "Archvile", "Revenant", "Cyberdemon", "SpiderMastermind"
        ]

    def get_action(self, state: np.ndarray) -> int:
        """Returns a random or predicted action by the agent. Probability of getting a random action is determined by epsilon.

        Args:
            env (_type_): the vizdoom environment
            state (np.ndarray): the current state represented by image and game-values

        Returns:
            int: index for the action
        """
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()   # Returns a random action from the action_space  

        else:
            # state = torch.tensor(state, dtype=torch.float).to(self.device)
            image = torch.tensor(state["screen"], dtype=torch.float).to(self.device)
            game_values = torch.tensor(state["gamevariables"], dtype=torch.float).to(self.device)
            preds = self.model(image, game_values) # Returns list of probabilities with size of action_space
            # print(f"\nPredictions before softmax: {preds}")
            # preds = torch.softmax(preds, dim=0)
            # print(f"Predictions after softmax: {preds}")
            prediction = torch.argmax(preds).item() # Returns the most probable action, represented by the index of the action space
            # print(f"Prediction: {prediction}")
            return prediction

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Adds data to the memory for later training

        Args:
            state (np.ndarray): the current state represented by image and game-values
            action (int): action taken on state
            reward (float): reward given, result of action
            next_state (np.ndarray): the next state, result of action
            done (bool): true if game ended on action
        """
        self.memory.append((state, action, reward, next_state, done))

    def train_long(self, batch_size: int):
        """Takes a sample from memory and trains the model

        Args:
            batch_size (int): the size of the sample
        """
        if len(self.memory) > batch_size:
            random_samples = random.sample(self.memory, batch_size)
        else:
            random_samples = list(self.memory)

        for state, action, reward, next_state, done in random_samples:

            # state = np.array(state)
            # next_state = np.array(next_state)
            # print(f"\ntrain long state: {state}")
            # print(f"train long state shape: {state.shape}\n")

            self.train(state, action, reward, next_state, done)

    def train_batch(self, memory, batch_size=32):
        pass

    def train(self, state: np.ndarray, action: int, reward: float, next_state: np.array, done: bool):
        """Optimizes the model based on the q-values calculated by the sample

        Args:
            state (np.ndarray): the current state represented by image and game-values
            action (int): action taken on state
            reward (float): reward given, result of action
            next_state (np.ndarray): the next state, result of action
            done (bool): true if game ended on action
        """

        image = torch.tensor(state["screen"], dtype=torch.float).to(self.device) / 255.0
        game_values = torch.tensor(state["gamevariables"], dtype=torch.float).to(self.device)
        next_image = torch.tensor(next_state["screen"], dtype=torch.float).to(self.device) / 255.0
        next_game_values = torch.tensor(next_state["gamevariables"], dtype=torch.float).to(self.device)

        # print("Get q:")
        q_values = self.model(image, game_values)
        # print(f"q_values: {q_values}")
        q_values = q_values.squeeze()
        # print(f"q_values after squeeze: {q_values}")
        q_value = q_values[action]

        # print("Get target_q:")
        with torch.no_grad():
            target_q_values = self.target_model(next_image, next_game_values).to(self.device)

            target_q_value = torch.max(target_q_values).to(self.device).detach()
            if not done:
                target_q_value = reward + self.gamma * torch.max(target_q_values)
            else:
                target_q_value = torch.tensor(reward, dtype=torch.float).to(self.device)

        # print(f"q_value: {q_value}\ntarget_q: {target_q_value}\n\n")
        self.loss = self.trainer.optimize_model(q_value, target_q_value)

    def update_target_model(self):
        """Updates the target_models' parameters to the models' parameters 
        """
        state_dict = self.model.state_dict()
        self.target_model.load_state_dict(state_dict)  
        self.target_model.eval()     

    def decay_epsilon(self):
        """_summary_
        """
        self.epsilon = max(self.end_epsilon, self.epsilon * self.epsilon_decay)

    # def train_episode(self):
    #     sample = list(self.episode_memory)
    #     for state, action, reward, next_state, done in sample:

    #         state = np.array(state)
    #         next_state = np.array(next_state)

    #         self.train(state, action, reward, next_state, done)
    #     self.episode_memory.clear()

    def enemies_on_screen(self):
        unwrapped_state = self.env.unwrapped.game.get_state()
        # print(unwrapped_state)
        num_doom_players = 0
        
        if unwrapped_state:
            labels = unwrapped_state.labels
            # print(f"labels: {labels}")
            
            for label in labels:
                # print(f"Num doom players in label loop: {num_doom_players}")
                if label.object_name in self.possible_enemies:
                    # print(f"Return true in possible enemies")
                    return True
                if label.object_name == "DoomPlayer":
                    num_doom_players+=1

                if num_doom_players > 1:
                    # print(f"Return true in possible doom player")
                    return True
        
        # print(f"return false")
        return False
                    

class MemoryData():
    pass