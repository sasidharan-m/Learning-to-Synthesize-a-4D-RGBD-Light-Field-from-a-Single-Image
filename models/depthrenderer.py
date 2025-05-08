# Python script that defines the function that renders the lamertian light field by warping the calculated depths and center view
# Author: Sasidharan Mahalingam
# Date Created: May 7 2025

# Import the required packages
import torch

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
    v_vals = torch.arange(V, device=device, dtype=torch.float32) - V / 2.0
    u_vals = torch.arange(U, device=device, dtype=torch.float32) - U / 2.0

    b, y, x, v, u = torch.meshgrid(b_vals, y_vals, x_vals, v_vals, u_vals, indexing='ij')

    # Warp coordinates by ray depths
    y_t = y + v * ray_depths
    x_t = x + u * ray_depths

    # Integer interpolation indices
    y_1 = torch.floor(y_t).to(torch.int32)
    y_2 = y_1 + 1
    x_1 = torch.floor(x_t).to(torch.int32)
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

    # Helper to gather using advanced indexing
    def gather_nd(img, y_idx, x_idx):
        """
        Function that gathers the light field data

        Arguments:
        ----------
        img - Tensor that contains the image values. Expected shape is [B, H, W, 1, 1]
        y_idx - Tensor that contains the y indices to gather. Expected shape is [B, H, W, V, U]
        x_idx - Tensor that contains the x indices to gather. Expected shape is [B, H, W, V, U]

        Returns:
        -------
        Returns the calulated lambertian light field. Output shape is [B, H, W, V, U]
        """
        B, H, W, _, _ = img.shape
        img = img.squeeze(-1).squeeze(-1)  # [B, H, W]
        val = []
        for b in range(B):
            val.append(img[b, y_idx[b], x_idx[b]])
        return torch.stack(val)

    # Gather neighbors
    lf_1 = gather_nd(c, y_1, x_1)
    lf_2 = gather_nd(c, y_2, x_1)
    lf_3 = gather_nd(c, y_1, x_2)
    lf_4 = gather_nd(c, y_2, x_2)

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