import math
import torch

def global_grad_norm(grads):
    """Computes the global gradient norm across all parameters."""
    total_norm = 0.0
    for g in grads.values():
        if g is not None:
            total_norm += g.norm().item() ** 2
    return total_norm ** 0.5

def gradient_clipping(grads, max_norm=1.0):
    """Apply gradient clipping to all gradients."""
    total_norm = global_grad_norm(grads)

    if math.isnan(total_norm) or math.isinf(total_norm):  # <- Use math instead of torch
        raise ValueError(f"NaN or Inf detected in global grad norm!")

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for g in grads.values():
            g *= clip_coef

    return grads

def check_for_nan_inf(grads):
    """Check for NaN/Inf gradients in any parameter."""
    for name, grad in grads.items():
        if torch.isnan(grad).any() or torch.isinf(grad).any():
            print(f"🚨 Gradient explosion detected in {name}")
            raise ValueError(f"Gradient in {name} contains NaN/Inf")
        