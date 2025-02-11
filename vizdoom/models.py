import torch
from torch import nn
import torch.optim as optim

# CNN GREJER

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

        dummy_input = torch.randn(3, 240, 320)
        dummy_input = self._forward(dummy_input)
        flattened_size = dummy_input.numel()
        flattened_size = int(flattened_size)

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=0, end_dim=-1),
            nn.Linear(in_features=flattened_size, out_features=1024),
            nn.Linear(in_features=1024, out_features=action_space),
        )

    def _forward(self, x):
        x = self.conv(x)
        return x

    def forward(self, x):
        x = self._forward(x)
        x = self.classifier(x)
        return x

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

# ViT GREJER
# Image size - 240 x 320
# Patch size - 16
# Num patches - 15 x 20 = 300, (image_x // patch_size) * (image_y // patch_size) -> (image_x * image_y) // (patch_size ** 2)
# Embed dim - 512

class ViT(nn.Module):
    def __init__(self, image_size: tuple[int, int], num_classes=4, depth=12, embed_dim=512, num_heads=8):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size)
        self.pos_embed = PositionalEncoding()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.encoder = nn.Sequential(*[TransformerEncoder(embed_dim, num_heads) for _ in range(depth)])
        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # [B, 1, embed_dim]

        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches + 1, embed_dim]
        x = self.pos_embed(x)
        x = self.encoder(x)
        return self.mlp_head(x[:, 0])

class PatchEmbedding(nn.Module):
    def __init__(self, image_size: tuple[int, int], patch_size=16, embed_dim=512):
        super().__init__()
        self.num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.projection = nn.Conv2d(in_channels=3,
                                    out_channels=embed_dim,
                                    kernel_size=patch_size,
                                    stride=patch_size)
        
    def forward(self, x: torch.Tensor):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, num_patches=300, embed_dim=512):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim)) # num_pactehs + 1 för att få med en classification token (CLS token)

    def forward(self, x: torch.Tensor):
        return x + self.pos_embedding[:, :x.shape[1], :]   
                                        # För att modellen ska lära sig var varje patch är
                                        # Kanske betyder att den kan inse om fienden är till vänster eller höger om sig
                                        # Verkar väldigt bra då
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim,
                                          num_heads=num_heads)
        
    def forward(self, x: torch.Tensor):
        x = x.transpose(0, 1) # [Batch, Seq, Dim] -> [Seq, Batch, Dim], pytorch kräver det av nån anledning
        x, _ = self.attn(x, x, x)
        return x.transpose(0, 1) # Original shape
    
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, mlp_dim=2048, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x