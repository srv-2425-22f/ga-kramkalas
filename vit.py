import torch
from torch import nn

torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = "cuda" if torch.cuda.is_available else "cpu"

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

        self.cls_token = nn.Parameter(torch.randn(size=(1, input_size, embed_dim)), requires_grad=True)
        self.position_embeddings = nn.Parameter(torch.randn(size=(1, num_patches+1, embed_dim)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)