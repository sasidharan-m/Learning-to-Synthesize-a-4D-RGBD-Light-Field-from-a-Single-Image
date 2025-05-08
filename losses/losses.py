# Python script that defines the losses used
# Author: Sasidharan Mahalingam
# Date Created: May 7 2025

# Import required packages
import torch
import torch.nn.functional as F


def transformRayDepths(ray_depths, u_step, v_step, lfsize):
    """
    Function that resamples ray depths for depth consistency regularization

    Arguments:
    ---------- 
    ray_depths - Tensor that contains the predicted ray depths. Expect shape is [B, H, W, V, U]
    u_step - Float scalar
    v_step - Float scalar
    lfsize - Tuple containing the light filed dimensions (Y, X, V, U)

    Returns:
    --------
    Returns a tensor of the warped ray depths. Output tensor is of shape [B, H, W, V, U]
    """
    B, H, W, V, U = ray_depths.shape
    device = ray_depths.device
    dtype  = torch.float32

    # 1) meshgrid for b, y, x, v, u
    b_idx, y_idx, x_idx, v_idx, u_idx = torch.meshgrid(
        torch.arange(B, device=device, dtype=dtype),
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        torch.arange(V, device=device, dtype=dtype) - (V/2),
        torch.arange(U, device=device, dtype=dtype) - (U/2),
        indexing='ij'
    )

    # 2) warp spatial coords by ray depths
    #    y_t, x_t shape [B,H,W,V,U]
    y_t = y_idx + v_step * ray_depths
    x_t = x_idx + u_step * ray_depths

    # 3) compute reparameterized angular coords for indexing
    v_t = v_idx - v_step + (V/2)
    u_t = u_idx - u_step + (U/2)

    # 4) floor/ceil and clamp
    y0 = torch.clamp(torch.floor(y_t), 0, H-1)
    y1 = torch.clamp(y0 + 1, 0, H-1)
    x0 = torch.clamp(torch.floor(x_t), 0, W-1)
    x1 = torch.clamp(x0 + 1, 0, W-1)

    v0 = torch.clamp(torch.floor(v_t), 0, V-1)
    u0 = torch.clamp(torch.floor(u_t), 0, U-1)

    # cast to long for indexing
    y0l, y1l = y0.long(), y1.long()
    x0l, x1l = x0.long(), x1.long()
    v0l = v0.long()
    u0l = u0.long()

    # 5) gather helper
    def gather(g, y_idx, x_idx, v_idx, u_idx):
        # g: [B,H,W,V,U]
        # idx arrays all [B,H,W,V,U] longs
        # flatten spatial+angular dims
        g_flat = g.reshape(B, -1)              # [B, H*W*V*U]
        lin_idx = (
            y_idx * (W*V*U) +
            x_idx * (V*U) +
            v_idx * U +
            u_idx
        ).reshape(B, -1)                        # [B, H*W*V*U]
        out = torch.stack([g_flat[b, lin_idx[b]] for b in range(B)], dim=0)
        return out.view(B, H, W, V, U)

    lf_1 = gather(ray_depths, y0l, x0l, v0l, u0l)
    lf_2 = gather(ray_depths, y1l, x0l, v0l, u0l)
    lf_3 = gather(ray_depths, y0l, x1l, v0l, u0l)
    lf_4 = gather(ray_depths, y1l, x1l, v0l, u0l)

    # 6) interpolation weights
    dy = y_t - y0
    dx = x_t - x0
    w1 = (1 - dy) * (1 - dx)
    w2 = (    dy) * (1 - dx)
    w3 = (1 - dy) * (    dx)
    w4 = (    dy) * (    dx)

    # 7) combine
    warped = w1*lf_1 + w2*lf_2 + w3*lf_3 + w4*lf_4
    return warped

def depthConsistencyLoss(x, lfsize):
    """
    Function that defines the depth consistency loss

    Arguments:
    ----------
    x - Input tensor of shape [B, H, W, V, U]
    lfsize - Tuple that contains the lightfield dimensions (Y, X, V, U)

    Returns:
    --------
    Returns the scalar loss, loss = mean(|d1| + |d2| + |d3|)
    """
    # warp along u, v, and both
    x_u  = transformRayDepths(x, u_step=1.0, v_step=0.0, lfsize=lfsize)
    x_v  = transformRayDepths(x, u_step=0.0, v_step=1.0, lfsize=lfsize)
    x_uv = transformRayDepths(x, u_step=1.0, v_step=1.0, lfsize=lfsize)

    # slice off the first row and column in angular dims (so indices 1: onward)
    # note: x has dims [B, H, W, V, U]
    #       we compare x[...,1:,1:] with each warped version’s same slice
    d1 = x[..., 1:, 1:] - x_u[..., 1:, 1:]
    d2 = x[..., 1:, 1:] - x_v[..., 1:, 1:]
    d3 = x[..., 1:, 1:] - x_uv[..., 1:, 1:]

    # mean absolute error
    loss = torch.mean(torch.abs(d1) + torch.abs(d2) + torch.abs(d3))
    return loss

def imageDerivatives(x, nc):
    """
    Function th calculates the image derivatives

    Arguments:
    ----------
    x -  Input tensor of shape [B, C, H, W]
    nc - number of channels (should equal C)

    Returns:
    --------
    Returns a tuple of derivatives dy, dx. Both are tensors of dimensions [B, C, H-2, W-2] (VALID padding)
    """
    # Sobel kernels
    k_y = torch.tensor([[ 1.0,  2.0,  1.0],
                        [ 0.0,  0.0,  0.0],
                        [-1.0, -2.0, -1.0]], dtype=x.dtype, device=x.device)
    k_x = torch.tensor([[ 1.0,  0.0, -1.0],
                        [ 2.0,  0.0, -2.0],
                        [ 1.0,  0.0, -1.0]], dtype=x.dtype, device=x.device)

    # shape [1,1,3,3], then repeat to [C,1,3,3] for depthwise
    ky = k_y.view(1,1,3,3).repeat(nc,1,1,1)
    kx = k_x.view(1,1,3,3).repeat(nc,1,1,1)

    # grouped conv: input channels=C, groups=C
    dy = F.conv2d(x, ky, bias=None, stride=1, padding=0, groups=nc)
    dx = F.conv2d(x, kx, bias=None, stride=1, padding=0, groups=nc)
    return dy, dx

def tvLoss(x, lfsize):
    """
    Function that calculated the spatial TV loss

    Arguments:
    ----------
    x - Input tensor of shape [B, H, W, V, U]
    lfsize - tuple that defines the lightfield dimensions (Y,X,V,U) for channels

    Returns:
    --------
    Returns the scalar total variation loss
    """
    B, H, W, V, U = x.shape
    # reshape to [B, C, H, W] with C = V*U
    C = V*U
    x_flat = x.permute(0,3,4,1,2).reshape(B, C, H, W)

    # compute derivatives
    dy, dx = imageDerivatives(x_flat, nc=C)

    # mean absolute gradient
    loss = torch.mean(torch.abs(dy) + torch.abs(dx))
    return loss