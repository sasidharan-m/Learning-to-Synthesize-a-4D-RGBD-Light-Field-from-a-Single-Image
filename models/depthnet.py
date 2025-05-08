# Python script that defines the Neural Network that predicts the depths for all viewpoints
# Author: Sasidharan Mahalingam
# Date Created: May 7 2025

# Import the required packages
import torch.nn as nn
from layers import CNNLayer2d, CNNLayer2dPlain

class DepthNetwork(nn.Module):
    """
    Class that defines the depth prediction network
    """
    def __init__(self, lfsize, disp_mult):
        """
        Constructor for the DepthNetwork class

        Arguments:
        ----------
        lfsize - tuple that has the light field dimensions (y, x, V, U)
        disp_mult - scalar multiplier for the final tanh output

        Returns:
        --------
        Nothing
        """
        super().__init__()
        _, _, v_sz, u_sz = lfsize
        num_views = v_sz * u_sz

        # convolutional backbone
        self.c1 = CNNLayer2d(3,    16,   kernel_size=(3,3), stride=1, dilation=1)
        self.c2 = CNNLayer2d(16,   64,   kernel_size=(3,3), stride=1, dilation=1)
        self.c3 = CNNLayer2d(64,  128,   kernel_size=(3,3), stride=1, dilation=1)
        self.c4 = CNNLayer2d(128, 128,   kernel_size=(3,3), stride=1, dilation=2)
        self.c5 = CNNLayer2d(128, 128,   kernel_size=(3,3), stride=1, dilation=4)
        self.c6 = CNNLayer2d(128, 128,   kernel_size=(3,3), stride=1, dilation=8)
        self.c7 = CNNLayer2d(128, 128,   kernel_size=(3,3), stride=1, dilation=16)
        self.c8 = CNNLayer2d(128, 128,   kernel_size=(3,3), stride=1, dilation=1)

        # project to num_views channels
        self.c9 = CNNLayer2d(128, num_views, kernel_size=(3,3), stride=1, dilation=1)

        # final plain conv + tanh
        self.c10 = CNNLayer2dPlain(
            in_channels=num_views,
            out_channels=num_views,
            kernel_size=(3,3),
            stride=1, dilation=1
        )

        self.disp_mult = disp_mult
        self.v_sz = v_sz
        self.u_sz = u_sz

    def forward(self, x):
        """
        Member function for the forward pass of the depth network

        Arguments:
        ----------
        x - Input tensor of shape [B,3,H,W]

        Returns:
        --------
        Returns the output tensor of shape [B, H, W, V, U]
        """
        B, C, H, W = x.shape

        h = self.c1(x)   # [B,16,H,W]
        h = self.c2(h)   # [B,64,H,W]
        h = self.c3(h)   # [B,128,H,W]
        h = self.c4(h)   # [B,128,H,W]
        h = self.c5(h)   # [B,128,H,W]
        h = self.c6(h)   # [B,128,H,W]
        h = self.c7(h)   # [B,128,H,W]
        h = self.c8(h)   # [B,128,H,W]

        h = self.c9(h)   # [B, V*U, H, W]
        h = self.c10(h)  # [B, V*U, H, W]

        # scaled tanh
        h = self.disp_mult * torch.tanh(h)

        # reshape to [B, V, U, H, W]
        h = h.view(B, self.v_sz, self.u_sz, H, W)

        # permute to [B, H, W, V, U]
        disp = h.permute(0, 3, 4, 1, 2)
        return disp