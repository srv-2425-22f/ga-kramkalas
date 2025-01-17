# Images are divided into patches to learn on
# The patches are fed into a Transformer Encoder
# Number of pacthes is the width of an image (square) divided by the width of a patch (square) and to the power of 2 -> N = (W/P)^2 ?

import torch
from torch import nn
import pandas as pd
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import random
import timeit
from tqdm import tqdm


RANDOM_SEED = 42
BATCH_SIZE = 512
EPOCHS = 40
LEARNING_RATE = 0.0001
NUM_CLASSES = 10
PATCH_SIZE = 4
IMG_SIZE = 28
IN_CHANNELS = 1 # Color channels, 1 for grey-scale
NUM_HEADS = 8 # How many attention heads are going to be used
DROPOUT = 0.001
ADAM_WEIGHT_DECAY = 0
ADAM_BETAS = (0.9, 0.999)
ACTIVATION = "gelu"
NUM_ENCODER = 4
EMBED_DIM = (PATCH_SIZE ** 2) * IN_CHANNELS # 16
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2 # 49

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"

class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels):
        super().__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size
            ),
            nn.Flatten(2)
        )
    
        self.cls_token = nn.Parameter(torch.randn(size=(1, in_channels, embed_dim)), requires_grad=True)
        self.position_embeddings = nn.Parameter(torch.randn(size=(1, num_patches+1, embed_dim)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        x = self.patcher(x).permute(0, 2, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.position_embeddings + x
        x = self.dropout(x)
        
        return x
    
model = PatchEmbedding(EMBED_DIM, PATCH_SIZE, NUM_PATCHES, DROPOUT, IN_CHANNELS).to(device)
x = torch.randn(512, 1, 28, 28).to(device)
# print(model(x).shape) # [512, 50, 16]

class ViT(nn.Module):
    def __init__(self, num_patches, img_size, num_classes, patch_size, embed_dim, num_encoders, num_heads, dropout, activation, in_channels):
        super().__init__()
        self.embeddings_block = PatchEmbedding(embed_dim, patch_size, num_patches, dropout, in_channels)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, 
                                                   nhead=num_heads, 
                                                   dropout=dropout, 
                                                   activation=activation, 
                                                   batch_first=True)
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        )

    def forward(self, x):
        x = self.embeddings_block(x)
        x = self.encoder_blocks(x)
        x = self.mlp_head(x[:, 0, :])

        return x
    
model = ViT(NUM_PATCHES, IMG_SIZE, NUM_CLASSES, PATCH_SIZE, EMBED_DIM, NUM_ENCODER, NUM_HEADS, DROPOUT, ACTIVATION, IN_CHANNELS).to(device)
x = torch.randn(512, 1, 28, 28).to(device)
# print(model(x).shape) # [512, 10]

train_transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_data = datasets.MNIST(
    root="data", 
    train=True, 
    download=True, 
    transform=train_transform, 
    target_transform=None 
)
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=test_transform,
)

train_dataloader = DataLoader(train_data, BATCH_SIZE, shuffle=True)
test_datalaoder = DataLoader(test_data, BATCH_SIZE, shuffle=False)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), betas=ADAM_BETAS, lr=LEARNING_RATE, weight_decay=ADAM_WEIGHT_DECAY)

plot_train_loss = []
plot_test_loss = []
plot_train_acc = []
plot_test_acc = []

start = timeit.default_timer()
for epoch in tqdm(range(EPOCHS), position=0, leave=True):
    model.train()    
    train_labels = []
    train_preds = []
    train_running_loss = 0

    for batch, (X, y) in enumerate(tqdm(train_dataloader, position=0, leave=True)):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        y_pred_label = torch.argmax(y_pred, dim=1)

        train_labels.extend(y.cpu().detach())
        train_preds.extend(y_pred_label.cpu().detach())

        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_running_loss += loss.item()
    train_loss = train_running_loss / (batch + 1)

    model.eval()
    test_labels = []
    test_preds = []
    test_running_loss = 0
    with torch.no_grad():
        for batch, (X,y) in enumerate(tqdm(test_datalaoder, position=0, leave=True)):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            y_pred_label = torch.argmax(y_pred, dim=1)

            test_labels.extend(y.cpu().detach())
            test_preds.extend(y_pred_label.cpu().detach())

            loss = criterion(y_pred, y)
            test_running_loss += loss.item()
    test_loss = test_running_loss / (batch + 1)

    train_acc = sum(1 for x, y in zip(train_preds, train_labels) if x == y) / len(train_labels)
    test_acc = sum(1 for x, y in zip(test_preds, test_labels) if x == y) / len(test_labels)

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

labels = []
imgs = []
model.eval()
with torch.no_grad():
    for batch, (X, y) in enumerate(tqdm(test_datalaoder, position=0, leave=True)):
        X, y = X.to(device), y.to(device)

        outputs = model(X)

        imgs.extend(X.cpu().detach())
        labels.extend([int(i) for i in torch.argmax(outputs, dim=1)])

from helper import plot
plot(plot_train_loss, plot_test_loss, plot_train_acc, plot_test_acc, stop-start)

f, axarr = plt.subplots(4, 6)
counter = 0
for i in range(4):
    for j in range(6):
        axarr[i][j].imshow(imgs[counter].squeeze(), cmap="gray")
        axarr[i][j].set_title(f"Predicted {labels[counter]}")
        counter += 1

plt.show()