# Python script that defines the function that renders the lamertian light field by warping the calculated depths and center view
# Author: Sasidharan Mahalingam
# Date Created: May 7 2025

# Import the required packages
import torch
import numpy as np

def depthRendering(central, ray_depths, lfsize):
    """
    Function that generates the lambertian light field by warping depths

    Arguments:
    ----------
    central -  Tensor that contains the centeral view of the light file. Expected shape is [B, H, W]
    ray_depths - Tensor that contains the ray depths. Expect shape is shape [B, H, W, V, U]
    lfsize - Tuple that defines the shape of the light field (Y, X, V, U)

    Returns:
    --------
    lf: Tensor that conatains the lambertian light field. Output shape is [B, H, W, V, U]
    """
    B, H, W = central.shape
    _, _, _, V, U = ray_depths.shape

    # 1) expand central to [B, H, W, V, U]
    c = central.unsqueeze(3).unsqueeze(4)

    # 2) coordinate grids
    device = central.device
    dtype = torch.float32

    # Create coordinate grids
    b_vals = torch.arange(B, device=device, dtype=torch.float32)
    y_vals = torch.arange(H, device=device, dtype=torch.float32)
    x_vals = torch.arange(W, device=device, dtype=torch.float32)
    v_vals = torch.arange(V, device=device, dtype=torch.float32) - float(V) / 2.0
    u_vals = torch.arange(U, device=device, dtype=torch.float32) - float(U) / 2.0

    b, y, x, v, u = torch.meshgrid(b_vals, y_vals, x_vals, v_vals, u_vals, indexing='ij')

    # Warp coordinates by ray depths
    y_t = y + v * ray_depths
    x_t = x + u * ray_depths

    # Integer interpolation indices
    y_1 = (torch.floor(y_t)).to(torch.int32)
    y_2 = y_1 + 1
    x_1 = (torch.floor(x_t)).to(torch.int32)
    x_2 = x_1 + 1

    # Clamp to valid range
    y_1 = torch.clamp(y_1, 0, H - 1)
    y_2 = torch.clamp(y_2, 0, H - 1)
    x_1 = torch.clamp(x_1, 0, W - 1)
    x_2 = torch.clamp(x_2, 0, W - 1)

    # For batch, v, u indices
    b_1 = b.to(torch.int32)
    v_1 = torch.zeros_like(b_1, dtype=torch.int32)
    u_1 = torch.zeros_like(b_1, dtype=torch.int32)

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



    # Gather neighbors
    lf_1 = gather_nd(c, interp_pts_1)
    lf_2 = gather_nd(c, interp_pts_2)
    lf_3 = gather_nd(c, interp_pts_3)
    lf_4 = gather_nd(c, interp_pts_4)

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