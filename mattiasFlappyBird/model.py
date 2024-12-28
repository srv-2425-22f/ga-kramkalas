# IMPORTS
import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np

# SETUP

torch.manual_seed(42)

# MODEL CLASS
class CNN_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super().__init__()
        self.device = device
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size,
                      out_channels=hidden_size,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_size,
                      out_channels=hidden_size,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_size,
                      out_channels=hidden_size,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_size,
                      out_channels=hidden_size,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.to(self.device)

        dummy_input = torch.randn(1, input_size, 72, 128).to(self.device)
        dummy_output = self._forward_conv(dummy_input)
        flattened_size = dummy_output.numel()
        # print(flattened_size)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=flattened_size, 
                      out_features=output_size)
        )

    def _forward_conv(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return x
    
    def forward(self, x):
        x = self._forward_conv(x)
        x = self.classifier(x)
        return x

    def save(self, file_name, path):
        if not os.path.exists(path):
            os.makedirs(path)
        file_name = os.path.join(path, file_name)
        torch.save(self.state_dict(), file_name)

# test_image = torch.rand(3, 288, 512).to(device)
# model = CNN_QNet(3, 8, 1).to(device)
# print(test_image.unsqueeze(dim=0).shape)
# print(model(test_image.unsqueeze(dim=0)))

# TRAINING CLASS
class QTrainer:
    def __init__(self, model, lr, gamma, device):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.device = device
        self.optimzier = optim.Adam(model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = np.array(state)
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.float).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        next_state = np.array(next_state)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)

        # print("State new shape:", next_state.shape)
        # print("State new shape length:", len(next_state.shape))

        if len(state.shape) < 4:
            state = torch.unsqueeze(state, dim=0)
            action = torch.unsqueeze(action, dim=0)
            reward = torch.unsqueeze(reward, dim=0)
            next_state = torch.unsqueeze(next_state, dim=0)
            done = (done, )
            # print("State shape:", state.shape)
            # print("State new shape:", next_state.shape)
        
        pred = self.model(state)

        target = pred.clone()
        for i in range(len(done)):
            Q_new = reward[i]
            if not done[i]:
                # print(next_state.shape)
                # print("next state", next_state[i])
                # print("next state [i]",next_state[i])
                # print(self.model(next_state[i]).shape)
                Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i].unsqueeze(dim=0)))

            target[i][torch.argmax(action[i]).item()] = Q_new
        
        self.optimzier.zero_grad()
        loss = self.loss_fn(pred, target)
        loss.backward()
        self.optimzier.step()