import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

EPOCHS = 1
LEARNING_RATE = 0.001

train_transform = transforms.Compose(transforms.ToTensor())
test_transform = transforms.Compose(transforms.ToTensor())
train_data = datasets.MNIST(root="data", train=True, transform=train_transform, target_transform=None, download=True)
test_data = datasets.MNIST(root="data", train=False, transform=test_transform, target_transform=None, download=True)

train_dataloader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=32, shuffle=False)

class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size,
                      out_channels=hidden_size,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_size,
                      out_channels=hidden_size,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_size,
                      out_channels=hidden_size,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_size,
                      out_channels=hidden_size,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_size*256,
                      out_features=output_size)
        )

    def forward(self, x):
        return self.classifier(self.conv2(self.conv1(x)))
    
model = CNN(3, 8, 10)

optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

images = torch.randn(size=(32, 3, 64, 64)) # [batch_size, color_channels, height, width]
test_image = images[0]

print(model(test_image.unsqueeze(dim=0)).shape)