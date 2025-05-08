# Python script that defines the Neural Network that predicts the refined lambertian light field
# Author: Sasidharan Mahalingam
# Date Created: May 7 2025

# Import the required packages
import torch
import torch.nn as nn
from layers import CNNLayer3d, CNNLayer3dPlain

class OcclusionsNetwork(nn.Module):
    """
    Class that defines the network for refining the lambertian light field
    """
    def __init__(self, lfsize):
        """
        Constructor for the OcclusionNetwork class

        Arguments:
        ----------
        lfsize - tuple that has the light field dimensions (y, x, V, U)

        Returns:
        --------
        Nothing
        """
        super().__init__()
        _, _, v_sz, u_sz = lfsize
        # num_views = u_sz * v_sz

        # 3D conv stages
        self.c1 = CNNLayer3d(in_channels=4, out_channels=8, kernel_size=(3,3,3))
        self.c2 = CNNLayer3d(in_channels=8, out_channels=8, kernel_size=(3,3,3))
        self.c3 = CNNLayer3d(in_channels=8, out_channels=8, kernel_size=(3,3,3))
        self.c4 = CNNLayer3d(in_channels=8, out_channels=8, kernel_size=(3,3,3))
        # plain final layer to 3 channels
        self.c5 = CNNLayer3dPlain(in_channels=8, out_channels=3, kernel_size=(3,3,3))

        self.v_sz = v_sz
        self.u_sz = u_sz

    def forward(self, x, shear):
        """
        Member function for the forward pass of the occlusion network

        Arguments:
        ----------
        x - Input tensor of shape [B, H, W, V, U, 4]
        shear - Input tensor that contains the shear values of shape [B, H, W, V, U, 3]

        Returns:
        --------
        Returns the output tensor of shape [B, H, W, V, U, 3]
        """
        B, H, W, V, U, _ = x.shape
        uv = U * V

        # 1) reshape into 5D for conv3d:
        #    [B,H,W,V,U,4] → transpose to [B,4,H,W,V,U]
        x1 = x.permute(0,5,1,2,3,4)
        #    → reshape to [B,4,H,W,uv]
        x1 = x1.reshape(B, 4, H, W, uv)
        #    → permute to [B,4, uv, H, W]
        x1 = x1.permute(0,1,4,2,3)

        # 2) pass through 3D CNN layers
        h = self.c1(x1)   # [B,8, uv, H, W]
        h = self.c2(h)    # [B,8, uv, H, W]
        h = self.c3(h)    # [B,8, uv, H, W]
        h = self.c4(h)    # [B,8, uv, H, W]
        h = torch.tanh(self.c5(h))  # [B,3, uv, H, W]

        # 3) reshape back to [B, H, W, V, U, 3]
        #    permute [B,3, uv, H, W] → [B, uv, H, W, 3]
        h = h.permute(0,2,3,4,1)
        #    reshape to [B, V, U, H, W, 3]
        h = h.reshape(B, V, U, H, W, 3)
        #    permute to [B, H, W, V, U, 3]
        output = h.permute(0,3,4,1,2,5)

        # 4) add shear
        return output + shear