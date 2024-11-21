import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
torch.manual_seed(42)

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.linear3 = nn.Linear(in_features=hidden_size, out_features=output_size)
        # self.model = nn.Sequential(
        #     nn.Linear(in_features=input_size, out_features=hidden_size),
        #     F.ReLU(),
        #     nn.Linear(in_features=hidden_size, out_features=hidden_size),
        #     F.ReLU(),
        #     nn.Linear(in_features=hidden_size, out_features=output_size),
        # )

    def forward(self, x: torch.tensor):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, file_name="model.pth"):
        path = "./saved_models"
        if not os.path.exists(path):
            os.makedirs(path)
        file_name = os.path.join(path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, dim=0)
            next_state = torch.unsqueeze(next_state, dim=0)
            action = torch.unsqueeze(action, dim=0)
            reward = torch.unsqueeze(reward, dim=0)
            done = (done, )

        # predicted Q values with state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.loss_fn(pred,target)
        loss.backward()
        self.optimizer.step()