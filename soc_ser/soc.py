import torch
import torch.nn as nn


class SOCPooling(nn.Module):
    """Second-Order Correlation pooling layer."""

    def __init__(
        self,
        in_dim: int = 768,
        spd_dim: int = 24,
        eps: float = 1e-5,
        eig_floor: float = 1e-7,
    ):
        super().__init__()
        self.spd_dim = spd_dim
        self.eps = eps
        self.eig_floor = eig_floor
        self.projection = nn.Linear(in_dim, spd_dim, bias=False)
        nn.init.orthogonal_(self.projection.weight)

    @property
    def output_dim(self) -> int:
        return self.spd_dim * (self.spd_dim + 1) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected input shape [B, T, D], got {tuple(x.shape)}")
        if x.size(1) < 2:
            raise ValueError("SOC pooling needs at least two frames per utterance.")

        z = self.projection(x)
        z = z - z.mean(dim=1, keepdim=True)

        cov = torch.bmm(z.transpose(1, 2), z) / (z.size(1) - 1)
        trace = torch.diagonal(cov, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        cov = cov / (trace + self.eps)

        eye = torch.eye(self.spd_dim, device=z.device, dtype=z.dtype).unsqueeze(0)
        cov = (cov + cov.transpose(1, 2)) * 0.5 + self.eps * eye

        eig_vals, eig_vecs = torch.linalg.eigh(cov)
        log_vals = torch.log(torch.clamp(eig_vals, min=self.eig_floor))
        log_cov = torch.bmm(
            torch.bmm(eig_vecs, torch.diag_embed(log_vals)),
            eig_vecs.transpose(1, 2),
        )

        idx = torch.triu_indices(self.spd_dim, self.spd_dim, device=x.device)
        return log_cov[:, idx[0], idx[1]]

