import torch
import torch.nn as nn

class SOCPooling(nn.Module):
    """
    Second-Order Correlation (SOC) Pooling Layer.

    Paper: Second-order correlation learning for self-supervised speech emotion recognition.
    """

    def __init__(self, in_dim=768, spd_dim=24):
        super().__init__()
        self.spd_dim = spd_dim

        # Manifold Projection (Linear)
        self.projection = nn.Linear(in_dim, spd_dim, bias=False)
        nn.init.orthogonal_(self.projection.weight)

    def forward(self, x):
        """
        Args:
            x: Input features [Batch, Time, in_dim]
        Returns:
            vec: Vectorized covariance descriptors [Batch, spd_dim*(spd_dim+1)/2]
        """

        # 1. Projection
        x = self.projection(x)  # [B, T, spd_dim]

        # 2. Centering
        x = x - torch.mean(x, dim=1, keepdim=True)

        # 3. Covariance Descriptor
        n = x.shape[1]
        cov = torch.bmm(x.transpose(1, 2), x) / (n - 1 + 1e-6)  # [B, spd_dim, spd_dim]

        # 4. Trace Normalization
        trace = torch.diagonal(cov, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]
        cov = cov / (trace + 1e-6)

        # 5. Riemannian Regularization (Symmetry + Positive Definiteness)
        cov = (cov + cov.transpose(1, 2)) / 2
        cov = cov + 1e-4 * torch.eye(self.spd_dim, device=x.device).unsqueeze(0)

        # 6. Log-Euclidean Mapping (Tangent Space Projection)
        eig_vals, eig_vecs = torch.linalg.eigh(cov)
        log_vals = torch.log(torch.clamp(eig_vals, min=1e-6))

        # Reconstruct: U * log(S) * U^T
        log_cov = torch.bmm(
            torch.bmm(eig_vecs, torch.diag_embed(log_vals)),
            eig_vecs.transpose(1, 2)
        )  # [B, spd_dim, spd_dim]

        # 7. Vectorization (Upper Triangular Flattening)
        idx = torch.triu_indices(self.spd_dim, self.spd_dim, device=x.device)
        vec = log_cov[:, idx[0], idx[1]]  # [B, spd_dim*(spd_dim+1)/2]

        return vec
