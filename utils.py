import torch
import torch.nn.functional as F

def second_derivative_conv(x, dx=1.0):
    # x : (B, 1, L)  -- batch, channel=1, length
    # discrete kernel [1, -2, 1] / dx^2
    kernel = torch.tensor([1.0, -2.0, 1.0], device=x.device, dtype=x.dtype).view(1,1,3) / (dx*dx)
    # padding='replicate' approx by F.pad then conv
    x_pad = F.pad(x, (1,1), mode='replicate')
    d2 = F.conv1d(x_pad, kernel)    # shape (B, 1, L)
    return d2

def local_energy(d2, window_size):
    # d2 : (B,1,L)
    # window_size should be odd for symmetric window
    w = torch.ones(1,1,window_size, device=d2.device, dtype=d2.dtype)
    # pad to keep same length
    pad = (window_size//2, window_size//2)
    d2_sq = d2 * d2
    d2_pad = F.pad(d2_sq, pad, mode='replicate')
    E_local = F.conv1d(d2_pad, w)   # (B,1,L) sum of squared second deriv in window
    return E_local

def huber_like(x, delta=1e-6):
    # Charbonnier-like: sqrt(x^2 + eps^2)
    eps = delta
    return torch.sqrt(x*x + eps*eps)

def smoothness_loss(recon, input_signal=None, dx=1.0, window_size=9,
                    use_weights=True, gamma=10.0, huber_eps=1e-6):
    # recon, input_signal: tensors (B, L) or (B,1,L); returns scalar loss
    if recon.dim()==2:
        recon = recon.unsqueeze(1)
    if input_signal is not None and input_signal.dim()==2:
        input_signal = input_signal.unsqueeze(1)

    d2 = second_derivative_conv(recon, dx=dx)       # (B,1,L)
    # robustify
    robust = huber_like(d2, delta=huber_eps)        # (B,1,L)

    E_local = local_energy(robust, window_size=window_size)  # (B,1,L)

    if use_weights and input_signal is not None:
        # compute gradient magnitude on input for edge-preserving weight
        # simple central diff (no dx factor; scale by dx if needed)
        grad_kernel = torch.tensor([-0.5, 0.0, 0.5], device=recon.device, dtype=recon.dtype).view(1,1,3)
        inp_pad = F.pad(input_signal, (1,1), mode='replicate')
        grad = F.conv1d(inp_pad, grad_kernel)       # (B,1,L)
        weight = torch.exp(-gamma * torch.abs(grad))  # small weight where grad large
    else:
        weight = torch.ones_like(E_local)

    # average (weighted)
    numerator = (weight * E_local).sum()
    denom = weight.sum().clamp_min(1.0)
    loss = numerator / denom
    return loss.squeeze()

# Example usage in training step:
# recon: (B, L), input: (B, L)
# recon_loss = F.mse_loss(recon, input)
# smooth_loss = smoothness_loss(recon, input_signal=input, dx=1.0, window_size=9, use_weights=True)
# loss = recon_loss + lambda_smooth * smooth_loss
