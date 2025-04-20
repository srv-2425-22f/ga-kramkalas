import torch
from torch import nn
import torch.optim as optim
import os

class CNN(nn.Module):
    """A Convolutional Neural Network for processing image inputs and outputting action predictions.
    
    Args:
        in_channels (int): Number of input channels (e.g., 3 for RGB images)
        hidden_size (int): Base size for hidden layers (will be doubled in second conv layer)
        action_space (float): Size of the output action space
    """
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        action_space: float,
    ):      
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
        )

    def _forward(self, x: torch.Tensor):
        """Forward pass through convolutional layers only.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output after convolutional layers
        """
        x = self.conv(x)
        return x

    def forward(self, x: torch.Tensor):
        """Complete forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output predictions
        """
        x = self._forward(x)
        x = self.classifier(x)
        return x
    
    def save(self, file_name: str="model.pth"):
        path = "./saved_models"
        if not os.path.exists(path):
            os.makedirs(path)
        file_name = os.path.join(path, file_name)
        torch.save(self.state_dict(), file_name)

class DQN(nn.Module):
    """Deep Q-Network with convolutional layers and LSTM temporal encoding.
    
    Args:
        in_channels (int): Number of input channels
        hidden_size (int): Base size for hidden layers
        action_space (float): Size of the output action space
        game_value_size (int): Size of additional game state values to incorporate
    """
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        action_space: float,
        game_value_size: int
    ):
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

    def _forward(self, x: torch.Tensor):
        """Forward pass through convolutional layers only.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output after convolutional layers
        """
        x = self.conv(x)
        return x

    def forward(self, image: torch.Tensor, game_values: torch.Tensor):
        """Complete forward pass through the network.
        
        Args:
            image (torch.Tensor): Input image tensor
            game_values (torch.Tensor): Additional game state values
            
        Returns:
            torch.Tensor: Output predictions
        """
        x = self._forward(image)
        x = self.flatten(x)
        x = torch.cat((x.unsqueeze(0), game_values.unsqueeze(0)), dim=1)
        x = self.linear(x)
        x, _ = self.temporal_encoder(x) # Returns short-term memory and long-term memory. Short-term is for prediction
        x = self.classifier(x)
        return x
    
    def save(self, file_name: str="model.pth"):
        """Save the model's state dictionary to file.
        
        Args:
            file_name (str): Name of the file to save the model to
        """
        path = "./saved_models"
        if not os.path.exists(path):
            os.makedirs(path)
        file_name = os.path.join(path, file_name)
        torch.save(self.state_dict(), file_name)

class ViT(nn.Module):
    """Vision Transformer for processing image inputs and game state values.
    
    Args:
        image_size (tuple[int, int]): Input image dimensions (height, width)
        in_channels (int): Number of input channels
        action_space (int): Size of the output action space
        game_value_size (int): Size of additional game state values
        patch_size (int): Size of image patches (default: 16)
        embed_dim (int): Dimension of embedding (default: 768)
        depth (int): Number of transformer layers (default: 3)
        n_heads (int): Number of attention heads (default: 3)
    """
    def __init__(self, 
                 image_size: tuple[int, int], # (H=240, W=320)
                 in_channels: int, 
                 action_space: int, 
                 game_value_size: int,
                 patch_size: int=16,
                 embed_dim: int=768, 
                 depth: int=3,
                 n_heads: int=3
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))
        self.transformer = nn.Sequential(*[
            TransformerEncoderLayer(embed_dim, n_heads) for _ in range(depth)
        ])

        self.classifier = nn.Linear(embed_dim + game_value_size, action_space)

    def forward(self, image: torch.Tensor, game_values: torch.Tensor):
        """Forward pass through the Vision Transformer.
        
        Args:
            image (torch.Tensor): Input image tensor
            game_values (torch.Tensor): Additional game state values
            
        Returns:
            torch.Tensor: Output predictions
        """
        image = image.unsqueeze(dim=0)
        game_values = game_values.unsqueeze(dim=0)
        B = image.shape[0]
        x = self.patch_embed(image)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x += self.pos_embed
        x = self.transformer(x)
        x = x[:, 0]
        x = torch.cat([x, game_values], dim=1)

        return self.classifier(x)

class PatchEmbedding(nn.Module):
    """Module for converting images into patch embeddings.
    
    Args:
        image_size (tuple[int, int]): Input image dimensions (height, width)
        patch_size (int): Size of image patches
        in_channels (int): Number of input channels
        embed_dim (int): Dimension of embedding
    """
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
        """Convert input image into patch embeddings.
        
        Args:
            x (torch.Tensor): Input image tensor
            
        Returns:
            torch.Tensor: Patch embeddings
        """
        x = self.proj(x) # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2) # (B, N, D)
        return x

class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with multi-head self-attention.
    
    Args:
        embed_dim (int): Dimension of embeddings
        n_heads (int): Number of attention heads
        mlp_ratio (float): Ratio for hidden dimension in MLP (default: 4.0)
        dropout (float): Dropout rate (default: 0.1)
    """
    def __init__(self, 
                 embed_dim: int, 
                 n_heads: int, 
                 mlp_ratio: float=4.0, 
                 dropout: float=0.1):
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
        """Forward pass through the transformer layer.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output after self-attention and MLP
        """
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class QTrainer:
    """Training utility for Q-learning models.
    
    Args:
        model (nn.Module): The model to train
        lr (float): Learning rate for optimizer
    """
    def __init__(self, model: nn.Module, lr: float):
        self.lr = lr
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def optimize_model(self, pred: torch.Tensor, target: torch.Tensor):
        """Perform one optimization step.
        
        Args:
            pred (torch.Tensor): Model predictions
            target (torch.Tensor): Target values
            
        Returns:
            float: Loss value
        """
        loss = self.loss_fn(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.cpu().detach().numpy()
    