"""Classification and contrastive heads (from VN-3DGS/src/models/classifier.py)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """MLP head: embed_dim -> embed_dim//2 -> num_classes."""

    def __init__(self, embed_dim: int = 128, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class ContrastiveHead(nn.Module):
    """Contrastive projection head, L2-normalised output."""

    def __init__(self, embed_dim: int = 128, proj_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)


class SimCLRProjectionHead(nn.Module):
    """SimCLR projection head: d -> h -> h -> out, BN+ReLU, L2-normalised."""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)
