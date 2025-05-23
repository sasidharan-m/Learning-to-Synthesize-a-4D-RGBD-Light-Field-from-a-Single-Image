# Python script that defines the losses used
# Author: Sasidharan Mahalingam
# Date Created: May 7 2025

# Import required packages
import torch
import torch.nn.functional as F
import numpy as np


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

    # Meshgrid for b, y, x, v, u
    b_idx, y_idx, x_idx, v_idx, u_idx = torch.meshgrid(
        torch.arange(B, device=device, dtype=dtype),
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        torch.arange(V, device=device, dtype=dtype) - (float(V)/2.0),
        torch.arange(U, device=device, dtype=dtype) - (float(U)/2.0),
        indexing='ij'
    )

    # Warp spatial coords by ray depths
    #    y_t, x_t shape [B,H,W,V,U]
    y_t = y_idx + v_step * ray_depths
    x_t = x_idx + u_step * ray_depths

    # Compute reparameterized angular coords for indexing
    v_t = v_idx - v_step + (float(V)/2.0)
    u_t = u_idx - u_step + (float(U)/2.0)

    b_1 = b_idx.to(torch.int32)
    y_1 = (torch.floor(y_t)).to(torch.int32)
    y_2 = y_1 + 1
    x_1 = (torch.floor(x_t)).to(torch.int32)
    x_2 = x_1 + 1
    v_1 = (v_t).to(torch.int32)
    u_1 = (u_t).to(torch.int32)

    # Clamp to valid range
    y_1 = torch.clamp(y_1, 0, H - 1)
    y_2 = torch.clamp(y_2, 0, H - 1)
    x_1 = torch.clamp(x_1, 0, W - 1)
    x_2 = torch.clamp(x_2, 0, W - 1)
    v_1 = torch.clamp(v_1, 0, V - 1)
    u_1 = torch.clamp(u_1, 0, U - 1)

    # Assemble interpolation indices
    interp_pts_1 = torch.stack([b_1, y_1, x_1, v_1, u_1], dim=-1)
    interp_pts_2 = torch.stack([b_1, y_2, x_1, v_1, u_1], dim=-1)
    interp_pts_3 = torch.stack([b_1, y_1, x_2, v_1, u_1], dim=-1)
    interp_pts_4 = torch.stack([b_1, y_2, x_2, v_1, u_1], dim=-1)

    # Helper to gather using advanced indexing
    def gather_nd(params, indices):
        """
        A PyTorch equivalent of TensorFlow's tf.gather_nd.

        Args:
            params (torch.Tensor): The source tensor to gather values from.
            indices (torch.LongTensor): Index tensor of shape [..., index_depth], where each entry specifies
                                        an index into `params`.

        Returns:
            torch.Tensor: Gathered values.
        """
        # indices shape: [*, index_depth]
        orig_shape = indices.shape[:-1]  # Shape of the output
        index_depth = indices.shape[-1]
        
        # Convert indices to tuple of slices
        indices = indices.reshape(-1, index_depth).T  # Shape: [index_depth, num_indices]
        gathered = params[tuple(indices)]  # Advanced indexing

        return gathered.reshape(orig_shape)

    
    lf_1 = gather_nd(ray_depths, interp_pts_1)
    lf_2 = gather_nd(ray_depths, interp_pts_2)
    lf_3 = gather_nd(ray_depths, interp_pts_3)
    lf_4 = gather_nd(ray_depths, interp_pts_4)

    # Compute interpolation weights
    y_1f = y_1.to(dtype)
    x_1f = x_1.to(dtype)
    d_y_1 = 1.0 - (y_t - y_1f)
    d_y_2 = 1.0 - d_y_1
    d_x_1 = 1.0 - (x_t - x_1f)
    d_x_2 = 1.0 - d_x_1

    # Interpolation
    w1 = d_y_1 * d_x_1
    w2 = d_y_2 * d_x_1
    w3 = d_y_1 * d_x_2
    w4 = d_y_2 * d_x_2

    lf = w1 * lf_1 + w2 * lf_2 + w3 * lf_3 + w4 * lf_4

    return lf

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
    x_flat = x.view(B, H, W, C)
    x_flat = x_flat.permute(0, 3, 1, 2).contiguous()

    # compute derivatives
    dy, dx = imageDerivatives(x_flat, nc=C)

    # mean absolute gradient
    loss = torch.mean(torch.abs(dy) + torch.abs(dx))
    return loss