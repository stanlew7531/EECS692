import torch

def weighted_mse_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        size_avg: bool = True) -> torch.Tensor:
    out = (pred - target) ** 2
    out = out * weight
    if size_avg:
        return out.sum() / len(pred)
    else:
        return out.sum()


def weighted_l1_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        size_avg: bool = True) -> torch.Tensor:
    out = torch.abs(pred - target)
    out = out * weight
    if size_avg:
        return out.sum() / len(pred)
    else:
        return out.sum()