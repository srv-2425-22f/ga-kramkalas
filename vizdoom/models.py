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
        """_summary_

        Args:
            in_channels (int): _description_
            hidden_size (int): _description_
            action_space (float): _description_
            game_value_size (int): _description_
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
            # nn.Tanh()
        )
        dummy_input = torch.randn(3, 120, 160)
        dummy_input = self._forward(dummy_input)
        flattened_size = dummy_input.numel()
        flattened_size = int(flattened_size)

        self.flatten = nn.Flatten(start_dim=0, end_dim=-1)

        self.linear = nn.Sequential(
            nn.Linear(in_features=flattened_size + game_value_size, out_features=256),
            nn.LayerNorm(256),
            nn.Tanh()
        )

        self.temporal_encoder = nn.LSTMCell(256, 256)

        self.classifier = nn.Linear(256, action_space)

        self.hidden_state = None
        self.cell_state = None   

    def _forward(self, x):
        x = self.conv(x)
        return x

    def forward(self, image: torch.Tensor, game_values: torch.Tensor):
        x = self._forward(image)
        # print(f"Conv mean: {x.mean().item()}")
        x = self.flatten(x)
        x = torch.cat((x.unsqueeze(0), game_values.unsqueeze(0)), dim=1)
        # print(f"Flatten mean: {x.mean().item()}")
        x = self.linear(x)
        # print(f"Linear mean: {x.mean().item()}")
        x, _ = self.temporal_encoder(x) # Returns short-term memory and long-term memory. Short-term is for prediction
        # print(f"LSTM mean: {x.mean().item()}\n")
        x = self.classifier(x)
        return x
    
    def initialize_hidden(self, batch_size: int):
        """Call this at the start of each episode"""
        device = next(self.parameters()).device
        self.hidden_state = torch.zeros(batch_size, self.temporal_encoder.hidden_size, device=device)
        self.cell_state = torch.zeros(batch_size, self.temporal_encoder.hidden_size, device=device)
    
    def save(self, file_name="model.pth"):
        path = "./saved_models"
        if not os.path.exists(path):
            os.makedirs(path)
        file_name = os.path.join(path, file_name)
        torch.save(self.state_dict(), file_name)

class ViT(nn.Module):
    def __init__(self, 
                 image_size: tuple[int, int], # (H=240, W=320)
                 in_channels: int, 
                 action_space: int, 
                 game_value_size: int,
                 patch_size: int=16,
                 embed_dim: int=768, 
                 depth: int=3,
                 n_heads: int=3,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))
        self.transformer = nn.Sequential(*[
            TransformerEncoderLayer(embed_dim, n_heads) for _ in range(depth)
        ])

        # OM MAN ANVÄNDER LSTM #
        # self.fc_before_lstm = nn.Sequential(
        #     nn.Linear(embed_dim + game_value_size, 256), # linear uses images data and game values
        #     nn.LayerNorm(256),
        #     nn.Tanh(),
        # )
        # self.temporal_encoder = nn.LSTMCell(256, 256)
        # self.classifier = nn.Linear(256, action_space)
        #
        # self.hidden_state = None
        # self.cell_state = None

        # OM MAN INTE ANVÄNDER LSTM #
        self.classifier = nn.Linear(embed_dim + game_value_size, action_space)

    
    # def initialize_hidden(self, batch_size: int):
    #     """Call this at the start of each episode"""
    #     # model.initialize_hidden(batch_size=32)
    #     device = next(self.parameters()).device
    #     self.hidden_state = torch.zeros(batch_size, self.lstm.hidden_size, device=device)
    #     self.cell_state = torch.zeros(batch_size, self.lstm.hidden_size, device=device)

    def forward(self, image: torch.Tensor, game_values: torch.Tensor):
        B = image.shape[0]
        x = self.patch_embed(image)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x += self.pos_embed
        x = self.transformer(x)
        x = x[:, 0]
        x = torch.cat([x, game_values], dim=1)

        # OM MAN ANVÄNDER LSTM #
        # x = self.fc_before_lstm(x)
        # x = self.temporal_encoder(x)

        return self.classifier(x)

class PatchEmbedding(nn.Module):
    def __init__(self,
                 image_size: tuple[int, int], # (H, W)
                 patch_size: int, 
                 in_channels: int, 
                 embed_dim: int, 
    ):
        super().__init__()
        self.proj = nn.Conv2d(in_channels=in_channels,
                              out_channels=embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        self.n_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
    
    def forward(self, x: torch.Tensor):
        x = self.proj(x) # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2) # (B, N, D)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 n_heads, 
                 mlp_ratio=4.0, 
                 dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class QTrainer:
    def __init__(self, model, lr):
        self.lr = lr
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def optimize_model(self, pred, target):
        loss = self.loss_fn(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.cpu().detach().numpy()