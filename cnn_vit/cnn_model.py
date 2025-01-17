import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt 
from helper import accuracy_fn, plot, plot_live

import timeit
from tqdm import tqdm

from pathlib import Path

torch.cuda.manual_seed(42)
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")

class CNN(nn.Module):
    def __init__(self, input_size: int, hidden_units: int, output_size: int):
        super().__init__()       

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.to(device)

        dummy_input = torch.randn(1, input_size, 224, 224).to(device)
        dummy_output = self._forward(dummy_input)
        flattened_size = dummy_output.numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=flattened_size,
                      out_features=output_size)
        )

    def _forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return x
    
    def forward(self, x):
        x = self._forward(x)
        x = self.classifier(x)
        return x

class Trainer():
    def __init__(self, model: nn.Module, epochs: int, loss_fn, optimizer: torch.optim, accuracy_fn):
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.model = model
        self.accuracy_fn = accuracy_fn

    def train_step(self, data_loader) -> tuple[float, float]:       
        self.train_loss, self.train_acc = 0, 0

        for batch, (X, y) in enumerate(tqdm(data_loader, position=0, leave=True)):
            X, y = X.to(device), y.to(device)
            pred = self.model(X)
            pred_label = torch.argmax(pred, dim=1)

            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.train_loss += loss.item()
            self.train_acc += self.accuracy_fn(y, pred_label)
        self.train_loss /= (batch+1)
        self.train_acc /= len(data_loader)

        return self.train_loss, self.train_acc

    def test_step(self, data_loader) -> tuple[float, float]:
        self.test_loss, self.test_acc = 0, 0

        with torch.no_grad():
            for batch, (X, y) in enumerate(tqdm(data_loader, position=0, leave=True)):
                X, y = X.to(device), y.to(device)
                pred = self.model(X)
                pred_label = torch.argmax(pred, dim=1)

                loss = self.loss_fn(pred, y)

                self.test_loss += loss.item()
                self.test_acc += self.accuracy_fn(y, pred_label)
        self.test_loss /= (batch+1)
        self.test_acc /= len(data_loader)

        return self.test_loss, self.test_acc
    
    def train_loop(self, train_dataloader: DataLoader, test_dataloader: DataLoader) -> None:
        plot_train_loss = []
        plot_test_loss = []
        plot_train_acc = []
        plot_test_acc = []
        start_time = timeit.default_timer()
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            train_loss, train_acc = self.train_step(train_dataloader)
            self.model.eval()
            test_loss, test_acc = self.test_step(test_dataloader)
            print("-"*30)
            print(f"Train loss epoch {epoch+1}: {train_loss:.4f}")
            print(f"Test loss epoch {epoch+1}: {test_loss:.4f}")
            print(f"Train accuracy epoch {epoch+1}: {train_acc:.4f}%")
            print(f"Test accuracy epoch {epoch+1}: {test_acc:.4f}%")
            print("-"*30)
            plot_train_loss.append(train_loss)
            plot_test_loss.append(test_loss)
            plot_train_acc.append(train_acc)
            plot_test_acc.append(test_acc)
            plot_live(plot_train_loss, plot_test_loss, plot_train_acc, plot_test_acc)
        end_time = timeit.default_timer()

        train_time = end_time-start_time

        print(f"Training time: {train_time:.2f}s")
        return plot_train_loss, plot_test_loss, plot_train_acc, plot_test_acc, train_time
    
def main():
    image_path = Path("hugo/data/vegetables")
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    data_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(root=train_dir,transform=data_transform,target_transform=None)
    test_data = datasets.ImageFolder(root=test_dir,transform=data_transform)
    
    train_dataloader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=32, shuffle=False)

    classes = train_data.classes

    model = CNN(3, 8, len(classes)).to(device)
    trainer = Trainer(model=model,  
                      epochs=40, 
                      loss_fn=nn.CrossEntropyLoss(),
                      optimizer=torch.optim.SGD(params=model.parameters(), lr=0.001),
                      accuracy_fn=accuracy_fn)
    
    plot_train_loss, plot_test_loss, plot_train_acc, plot_test_acc, train_time = trainer.train_loop(train_dataloader, test_dataloader)
    plot(plot_train_loss, plot_test_loss, plot_train_acc, plot_test_acc, train_time)

if __name__ == "__main__":
    main()