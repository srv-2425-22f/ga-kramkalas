import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import timeit
from tqdm import tqdm
from helper import accuracy_fn, plot

EPOCHS = 40
LEARNING_RATE = 0.001

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

torch.manual_seed(42)
torch.cuda.manual_seed(42)

train_transform = transforms.Compose([transforms.RandomRotation(15), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
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
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_size,
                      out_channels=hidden_size*2,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_size*2,
                      out_channels=hidden_size*2,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_size*16*2,
                      out_features=output_size)
        )

    def forward(self, x):
        return self.classifier(self.conv2(self.conv1(x)))
    
model = CNN(1, 16, 10)
model.to(device)

optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

# images = torch.randn(size=(32, 3, 64, 64)) # [batch_size, color_channels, height, width]
# test_image = images[0]

# print(model(test_image.unsqueeze(dim=0)).shape)

plot_train_loss = []
plot_train_acc = []
plot_test_loss = []
plot_test_acc = []

start = timeit.default_timer()
for epoch in tqdm(range(EPOCHS)):
    model.train()
    train_labels = []
    train_preds = []
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(tqdm(train_dataloader, position=0, leave=True)):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        y_pred_label = torch.argmax(y_pred, dim=1)

        train_labels.extend(y_pred.cpu().detach())
        train_preds.extend(y_pred_label.cpu().detach())

        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += accuracy_fn(y, y_pred_label)
    train_loss = train_loss / (batch+1)

    model.train()
    test_labels = []
    test_preds = []
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(tqdm(test_dataloader, position=0, leave=True)):
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            test_pred_label = torch.argmax(test_pred, dim=1)

            test_labels.extend(test_pred.cpu().detach())
            test_preds.extend(test_pred_label.cpu().detach())

            loss = loss_fn(test_pred, y)
            test_loss += loss.item()
            test_acc += accuracy_fn(y, test_pred_label)
    test_loss = test_loss / (batch+1)

    train_acc /= len(train_dataloader)
    test_acc /= len(test_dataloader)

    print("-"*30)
    print(f"Train loss epoch {epoch+1}: {train_loss:.4f}")
    print(f"Test loss epoch {epoch+1}: {test_loss:.4f}")
    print(f"Train accuracy epoch {epoch+1}: {train_acc:.4f}")
    print(f"Test accuracy epoch {epoch+1}: {test_acc:.4f}")
    print("-"*30)

    plot_train_loss.append(train_loss)
    plot_test_loss.append(test_loss)
    plot_train_acc.append(train_acc*100)
    plot_test_acc.append(test_acc*100)

stop = timeit.default_timer()
print(f"Training time: {stop-start:.2f}s")

torch.cuda.empty_cache()


plot(plot_train_loss, plot_test_loss, plot_train_acc, plot_test_acc, stop-start)