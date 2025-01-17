import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from helper import accuracy_fn, plot, plot_live
import timeit
from tqdm import tqdm
from pathlib import Path

torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = "cuda" if torch.cuda.is_available else "cpu"
torch.cuda.empty_cache()

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

BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 0.001
CLASSES = train_data.classes
PATCH_SIZE = 8
IMG_SIZE = 224
IN_CHANNELS = 3
NUM_HEADS = 4
DROPOUT = 0.001
ADAM_WEIGHT_DECAY = 0
ADAM_BETAS = (0.9, 0.999)
ACTIVATION = "gelu"
NUM_ENCODER = 2
EMBED_DIM = (PATCH_SIZE ** 2) * IN_CHANNELS
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2

class PatchEmbedding(nn.Module):
    def __init__(self, input_size, embed_dim, patch_size, num_patches, dropout):
        super().__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels=input_size,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size
            ),
            nn.Flatten(2)
        )

        self.class_token = nn.Parameter(torch.randn(size=(1, input_size, embed_dim)), requires_grad=True)
        self.position_embeddings = nn.Parameter(torch.randn(size=(1, num_patches+3, embed_dim)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        
        x = self.patcher(x).permute(0, 2, 1)
        x = torch.cat([class_token, x], dim=1)
        x = self.position_embeddings + x
        x = self.dropout(x)

        return x
    
class ViT(nn.Module):
    def __init__(self, input_size, embed_dim, patch_size, num_patches, dropout, num_classes, num_encoders, num_heads, activation):
        super().__init__()
        self.embeddings_block = PatchEmbedding(input_size, embed_dim, patch_size, num_patches, dropout)

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
    model = ViT(input_size=IN_CHANNELS,
                embed_dim=EMBED_DIM,
                patch_size=PATCH_SIZE,
                num_patches=NUM_PATCHES,
                dropout=DROPOUT,
                num_classes=len(CLASSES),
                num_encoders=NUM_ENCODER,
                num_heads=NUM_HEADS,
                activation=ACTIVATION).to(device)
    trainer = Trainer(model=model,  
                      epochs=40, 
                      loss_fn=nn.CrossEntropyLoss(),
                      optimizer=torch.optim.Adam(params=model.parameters(), betas=ADAM_BETAS, lr=LEARNING_RATE, weight_decay=ADAM_WEIGHT_DECAY),
                      accuracy_fn=accuracy_fn)
    
    plot_train_loss, plot_test_loss, plot_train_acc, plot_test_acc, train_time = trainer.train_loop(train_dataloader, test_dataloader)
    plot(plot_train_loss, plot_test_loss, plot_train_acc, plot_test_acc, train_time)

if __name__ == "__main__":
    main()