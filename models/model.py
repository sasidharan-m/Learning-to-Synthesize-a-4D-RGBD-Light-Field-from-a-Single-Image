# Python script that defines the forward model 
# Author: Sasidharan Mahalingam
# Date Created: May 7 2025

# Import the required packages
import torch
import torch.nn as nn
from depthnet import DepthNetwork
from occlusionnet import OcclusionsNetwork
from depthrenderer import depthRendering

class ForwardModel(nn.Module):
    """
    Class that defines the complete forward model
    """
    def __init__(self, lfsize, disp_mult):
        """
        Constructor of the ForwardModel class

        Arguments:
        ----------
        lfsize - Tuple that has the lightfield dimensions
        disp_mult - Range for the disparity estimates

        Returns:
        --------
        Nothing
        """
        super().__init__()
        # depth prediction network
        self.depth_net = DepthNetwork(lfsize=lfsize, disp_mult=disp_mult)
        # occlusions / residual network
        self.occ_net   = OcclusionsNetwork(lfsize=lfsize)
        self.lfsize = lfsize

    def forward(self, x):
        """
        Member function for the forward pass of the complete network

        Arguments:
        ----------
        x - Input tensor of shape [B, 3, H, W]  (center RGB image)

        Returns:
        --------
        ray_depths - Tensor that contains the predicted depth. Output is of shape [B, H, W, V, U]
        lf_shear - Tensor that contains the claculated shear. Output is of shape [B, H, W, V, U, 3]
        y - Tensor that contains the predicted light field data. Output is of shape [B, H, W, V, U, 3]  (occlusion‐corrected output)
        """
        B, C, H, W = x.shape
        _, _, v_sz, u_sz = self.depth_net.v_sz, self.depth_net.u_sz, self.depth_net.v_sz, self.depth_net.u_sz
        
        # 1) predict ray depths from the center image
        #    DepthNetwork expects [B,3,H,W], returns [B,H,W,V,U]
        ray_depths = self.depth_net(x)  # [B,H,W,V,U]

        # 2) render each color channel separately
        #    our depth_rendering expects central in [B,H,W] and outputs [B,H,W,V,U]
        c = x.permute(0,2,3,1)       # → [B, H, W, 3]
        lf_r = depthRendering(c[...,0], ray_depths, self.lfsize)  # [B,H,W,V,U]
        lf_g = depthRendering(c[...,1], ray_depths, self.lfsize)
        lf_b = depthRendering(c[...,2], ray_depths, self.lfsize)

        # stack into [B,H,W,V,U,3]
        lf_shear = torch.stack([lf_r, lf_g, lf_b], dim=-1)

        # 3) prepare input for occlusions network:
        #    stack [R,G,B, stop_gradient(depths)] along channel dim
        #    occlusions_network expects x: [B,H,W,V,U,4] and shear: [B,H,W,V,U,3]
        #    so we need to detach ray_depths to mimic tf.stop_gradient
        depths_det = ray_depths.detach().unsqueeze(-1)  # [B,H,W,V,U,1]
        shear_and_depth = torch.cat([lf_shear, depths_det], dim=-1)  # [B,H,W,V,U,4]

        # 4) occlusions / residual prediction
        y = self.occ_net(shear_and_depth, lf_shear)  # [B,H,W,V,U,3]

        return ray_depths, lf_shear, y