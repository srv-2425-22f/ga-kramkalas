import torch
from torch import nn
import torch.optim as optim
import os

class CNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        action_space: float,
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
        dummy_input = torch.randn(3, 120, 160)
        dummy_input = self._forward(dummy_input)
        flattened_size = dummy_input.numel()
        flattened_size = int(flattened_size)

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=0, end_dim=-1),
            nn.Linear(in_features=flattened_size, out_features=action_space),
            # nn.ReLU(),
            # nn.Linear(in_features=1024, out_features=action_space),
        )

    def _forward(self, x):
        x = self.conv(x)
        return x

    def forward(self, x):
        x = self._forward(x)
        x = self.classifier(x)
        return x
    
    def save(self, file_name="model.pth"):
        path = "./saved_models"
        if not os.path.exists(path):
            os.makedirs(path)
        file_name = os.path.join(path, file_name)
        torch.save(self.state_dict(), file_name)

class DQN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        action_space: float,
        game_value_size: int
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
        dummy_input = torch.randn(3, 120, 160)
        dummy_input = self._forward(dummy_input)
        flattened_size = dummy_input.numel()
        flattened_size = int(flattened_size)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=flattened_size + game_value_size, out_features=action_space),
            # nn.ReLU(),
            # nn.Linear(in_features=1024, out_features=action_space),
        )

    def _forward(self, x):
        x = self.conv(x)
        return x

    def forward(self, image: torch.Tensor, game_values: torch.Tensor):
        print(f"image: {image}")
        print(f"game_values: {game_values}")
        x = self._forward(image)
        x = x.view(x.size(0), -1)
        x = torch.concat((x, game_values.unsqueeze(0)), dim=1)
        x = self.classifier(x)
        torch.cat
        return x
    
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
        self.loss_fn = nn.MSELoss()

    def optimize_model(self, pred, target):
        # pred = torch.tensor(pred, dtype=torch.float)
        # target = torch.tensor(target, dtype=torch.float)
        # print(pred.type(), target.type())
        loss = self.loss_fn(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print(f"Loss: {loss:.4f} | Prediction: {pred:.4f}, Target: {target:.4f}")
        return loss.cpu().detach().numpy()