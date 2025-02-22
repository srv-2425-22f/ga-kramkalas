import torch
from torch import nn
import torch.optim as optim
import os


class CNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        discrete_action_space: float,
        continuous_action_space: float,
    ):
        """
        Initializes a Convolutional Neural Network model which takes in an image

        Args:
            in_channels: Number of color channels of the image
            hidden_size
        """
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_size,
                kernel_size=8,
                stride=4,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_size,
                out_channels=hidden_size * 2,
                kernel_size=4,
                stride=2,
                padding=0,
            ),
            nn.ReLU(),
        )
        dummy_input = torch.randn(3, 240, 320)
        dummy_input = self._forward(dummy_input)
        flattened_size = dummy_input.numel()
        flattened_size = int(flattened_size)

        self.discrete_classifier = nn.Sequential(
            nn.Flatten(start_dim=0, end_dim=-1),
            nn.Linear(in_features=flattened_size, out_features=discrete_action_space),
            # nn.ReLU(),
            # nn.Linear(in_features=1024, out_features=action_space),
        )
        self.continuous_classifier = nn.Sequential(
            nn.Flatten(start_dim=0, end_dim=-1),
            nn.Linear(in_features=flattened_size, out_features=continuous_action_space),
            nn.Tanh(),
        )

    def _forward(self, x):
        x = self.conv(x)
        return x

    def forward(self, x):
        x = self._forward(x)
        discrete_values = self.discrete_classifier(x)
        continuous_values = self.continuous_classifier(x)
        return discrete_values, continuous_values

    def save(self, file_name="model.pth"):
        path = "./saved_models"
        if not os.path.exists(path):
            os.makedirs(path)
        file_name = os.path.join(path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr):
        self.lr = lr
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.discrete_loss_fn = nn.MSELoss()
        self.continuous_loss_fn = nn.L1Loss()

    def optimize_model(
        self, discrete_pred, continuous_pred, discrete_target, continuous_target
    ):
        discrete_loss = self.discrete_loss_fn(discrete_pred, discrete_target)
        continuous_loss = self.continuous_loss_fn(continuous_pred, continuous_target)

        loss = discrete_loss + continuous_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.cpu().detach().numpy()
