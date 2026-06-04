import torch
import torch.nn as nn

from .soc import SOCPooling


class SOCClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        spd_dim: int = 24,
        hidden_dim: int = 128,
        num_classes: int = 8,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.pool = SOCPooling(input_dim, spd_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.pool.output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(x))

