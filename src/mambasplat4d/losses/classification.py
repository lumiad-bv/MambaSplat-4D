"""Classification losses (from VN-3DGS/src/training/losses.py)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """SupCon loss (Khosla et al., 2020)."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]

        sim_matrix = features @ features.T / self.temperature

        labels = labels.view(-1, 1)
        mask_pos = (labels == labels.T).float().to(device)
        mask_pos.fill_diagonal_(0)

        sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()

        exp_sim = torch.exp(sim_matrix)

        mask_self = torch.eye(batch_size, device=device)
        denom = (exp_sim * (1 - mask_self)).sum(dim=1, keepdim=True)

        log_prob = sim_matrix - torch.log(denom + 1e-8)

        num_pos = mask_pos.sum(dim=1)
        loss = -(mask_pos * log_prob).sum(dim=1) / (num_pos + 1e-8)

        valid = num_pos > 0
        if valid.sum() > 0:
            loss = loss[valid].mean()
        else:
            loss = torch.tensor(0.0, device=device)

        return loss
