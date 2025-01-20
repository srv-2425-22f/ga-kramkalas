import torch
from torch import nn
import torch.optim as optim

class CNN(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        hidden_size: int, 
        action_space: int,
        device
    ):
        """
        Initializes a Convolutional Nueral Network model which takes in an image
        
        Args:
            in_channels: Number of color channels of the image
            hidden_size
        """
        super().__init__()
        

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=hidden_size,
                      kernel_size=8,
                      stride=4,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_size,
                      out_channels=hidden_size*2,
                      kernel_size=4,
                      stride=2,
                      padding=0),
            nn.ReLU(),
        )

        self.to(device)
    
        dummy_input = torch.randn(3, 240, 320)
        flattened_size = self._forward(dummy_input.to(device)).numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=flattened_size,
                      out_features=512),
            nn.Linear(in_features=512,
                      out_features=action_space)
        )

    def _forward(self, x):
        x = self.conv(x)
        return x
    
    def forward(self, x):
        x = self._forward(x)
        x = self.classifier(x)
        return x
    
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()