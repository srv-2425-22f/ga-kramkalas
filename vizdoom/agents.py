import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict, deque
from models import QTrainer
import random

class Basic:
    """A Deep Q-Network (DQN) agent implementation for VizDoom environments.
    
    This agent implements the core DQN algorithm with experience replay and target network.
    It supports both random exploration and policy-based action selection.
    
    Args:
        env: The VizDoom environment/gymnasium wrapper
        learning_rate (float): Learning rate for the optimizer
        start_epsilon (float): Initial exploration rate
        epsilon_decay (float): Rate at which exploration decreases
        end_epsilon (float): Minimum exploration rate
        model (nn.Module): The Q-network model
        target_model (nn.Module): The target Q-network model
        device (str): Device to use for training ('cuda' or 'cpu')
        gamma (float, optional): Discount factor for future rewards. Defaults to 0.99.
    """
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
        # List of possible enemy types in VizDoom for detection
        self.possible_enemies = [
            "Zombieman", "ShotgunGuy", "ChaingunGuy", "Imp", "Demon", "Spectre",
            "LostSoul", "Cacodemon", "HellKnight", "BaronOfHell", "Arachnotron",
            "Mancubus", "Archvile", "Revenant", "Cyberdemon", "SpiderMastermind"
        ]

    def get_action(self, state: np.ndarray) -> int:
        """Select an action using epsilon-greedy policy.
        
        Args:
            state (np.ndarray): Current game state containing screen and game variables
            
        Returns:
            int: Selected action index
        """
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()   # Returns a random action from the action_space  

        else:
            image = torch.tensor(state["screen"], dtype=torch.float).to(self.device)
            game_values = torch.tensor(state["gamevariables"], dtype=torch.float).to(self.device)
            preds = self.model(image, game_values) # Returns list of probabilities with size of action_space
            prediction = torch.argmax(preds).item() # Returns the most probable action, represented by the index of the action space
            return prediction

    def remember(self, 
                 state: np.ndarray, 
                 action: int, 
                 reward: float, 
                 next_state: np.ndarray, 
                 done: bool):
        """Store experience in replay memory.
        
        Args:
            state (np.ndarray): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.ndarray): Next state
            done (bool): Whether episode terminated
        """
        self.memory.append((state, action, reward, next_state, done))

    def train_long(self, batch_size: int):
        """Train on a batch of experiences from replay memory.
        
        Args:
            batch_size (int): Number of experiences to sample from memory
        """
        if len(self.memory) > batch_size:
            random_samples = random.sample(self.memory, batch_size)
        else:
            random_samples = list(self.memory)

        for state, action, reward, next_state, done in random_samples:
            self.train(state, action, reward, next_state, done)

    def train(self, 
              state: np.ndarray, 
              action: int, 
              reward: float, 
              next_state: np.ndarray, 
              done: bool):
        """Perform a single training step using the Bellman equation.
        
        Args:
            state (np.ndarray): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.ndarray): Next state
            done (bool): Whether episode terminated
        """
        image = torch.tensor(state["screen"], dtype=torch.float).to(self.device) / 255.0
        game_values = torch.tensor(state["gamevariables"], dtype=torch.float).to(self.device)
        next_image = torch.tensor(next_state["screen"], dtype=torch.float).to(self.device) / 255.0
        next_game_values = torch.tensor(next_state["gamevariables"], dtype=torch.float).to(self.device)

        q_values = self.model(image, game_values)
        q_values = q_values.squeeze()
        q_value = q_values[action]

        with torch.no_grad():
            target_q_values = self.target_model(next_image, next_game_values).to(self.device)

            target_q_value = torch.max(target_q_values).to(self.device).detach()
            if not done:
                target_q_value = reward + self.gamma * torch.max(target_q_values)
            else:
                target_q_value = torch.tensor(reward, dtype=torch.float).to(self.device)

        self.loss = self.trainer.optimize_model(q_value, target_q_value)

    def update_target_model(self):
        """Updates the target_models' parameters to the models' parameters."""
        state_dict = self.model.state_dict()
        self.target_model.load_state_dict(state_dict)  
        self.target_model.eval()     

    def decay_epsilon(self):
        """Decay exploration rate according to epsilon_decay, with minimum end_epsilon."""
        self.epsilon = max(self.end_epsilon, self.epsilon * self.epsilon_decay)

    def enemies_on_screen(self) -> bool:
        """Check if any enemies are visible on screen.
        
        Returns:
            bool: True if enemies are detected, False otherwise
        """
        unwrapped_state = self.env.unwrapped.game.get_state()
        num_doom_players = 0
        
        if unwrapped_state:
            labels = unwrapped_state.labels
            
            for label in labels:
                if label.object_name in self.possible_enemies:
                    return True
                if label.object_name == "DoomPlayer":
                    num_doom_players+=1

                if num_doom_players > 1:
                    return True
        
        return False
