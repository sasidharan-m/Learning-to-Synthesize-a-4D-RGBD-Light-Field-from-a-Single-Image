# Python script that defines the different layers used in the model
# Author: Sasidharan Mahalingam
# Date Created: May 7 2025

# Import the required packages
import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_variable(shape):
    """
    Function that creates a conv weight tensor initialized with Xavier (Glorot) uniform.

    Arguments:
    ----------
    shape - Shape of the weights to be initialized

    Returns:
    --------
    Returns the initialized weights
    """
    w = nn.Parameter(torch.empty(shape))
    # For conv layers: fan_in/out based on kernel + in/out channels
    nn.init.xavier_uniform_(w)
    return w


def bias_variable(shape, init_bias=0.0):
    """
    Function that creates a bias tensor initialized to a constant.

    Arguments:
    ---------
    shape - Shape of the bias variables to be initialized

    Returns:
    --------
    Returns the initialized bias variables
    """
    b = nn.Parameter(torch.full(shape, init_bias))
    return b


class CNNLayer2d(nn.Module):
    """
    Class that defines a 2D conv + bias + instance-norm + leaky-ReLU layer, with symmetric padding.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, negative_slope=0.2):
        """
        Default constructor for the CNNLayer2d class

        Arguments:
        ----------
        in_channels - Number of channels in the input tensor to the layer
        out_channels - Number of channels in the output tensor of the layer
        kernel_size - Kernel size for the convolutional neural network
        stride - Stride for the convolutional neural network
        dilation - Dilation value for the convolutional neural network
        negative_slope - Negative slope setting for the Leaky ReLU activation

        Returns:
        --------
        None
        """
        super().__init__()
        # weight and bias as raw Parameters
        self.weight = weight_variable((out_channels, in_channels, *kernel_size))
        self.bias   = bias_variable((out_channels,))
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size

        self.instnorm = nn.InstanceNorm2d(out_channels, affine=False)
        self.act      = nn.LeakyReLU(negative_slope, inplace=True)

    def forward(self, x):
        """
        Member function for the forward pass of the CNN layer

        Arguments:
        ----------
        x - Tensor that holds the input values to the CNN layer

        Returns:
        --------
        Returns the output tensor
        """
        # compute symmetric pad amounts
        pad_h = self.dilation * (self.kernel_size[0] - 1) // 2
        pad_w = self.dilation * (self.kernel_size[1] - 1) // 2
        # PyTorch reflection pad ≈ TF symmetric
        x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='reflect')

        # convolution via F.conv2d
        x = F.conv2d(x,
                     self.weight,
                     bias=None,
                     stride=self.stride,
                     dilation=self.dilation)

        # add bias
        x = x + self.bias.view(1, -1, 1, 1)

        # instance norm + activation
        x = self.instnorm(x)
        x = self.act(x)
        return x


class CNNLayer2dPlain(nn.Module):
    """
    Class that defines a 2D conv + bias only layer, with symmetric padding.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        """
        Default constructor for the CNNLayer2dPlain class

        Arguments:
        ----------
        in_channels - Number of channels in the input tensor to the layer
        out_channels - Number of channels in the output tensor of the layer
        kernel_size - Kernel size for the convolutional neural network
        stride - Stride for the convolutional neural network
        dilation - Dilation value for the convolutional neural network

        Returns:
        --------
        None
        """
        super().__init__()
        self.weight = weight_variable((out_channels, in_channels, *kernel_size))
        self.bias   = bias_variable((out_channels,))
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size

    def forward(self, x):
        """
        Member function for the forward pass of the CNN layer

        Arguments:
        ----------
        x - Tensor that holds the input values to the CNN layer

        Returns:
        --------
        Returns the output tensor
        """
        pad_h = self.dilation * (self.kernel_size[0] - 1) // 2
        pad_w = self.dilation * (self.kernel_size[1] - 1) // 2
        x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
        x = F.conv2d(x,
                     self.weight,
                     bias=None,
                     stride=self.stride,
                     dilation=self.dilation)
        x = x + self.bias.view(1, -1, 1, 1)
        return x


class CNNLayer3d(nn.Module):
    """
    Class that defines a 3D conv + bias + instance-norm + leaky-ReLU layer, with symmetric padding.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, negative_slope=0.2):
        """
        Default constructor for the CNNLayer3d class

        Arguments:
        ----------
        in_channels - Number of channels in the input tensor to the layer
        out_channels - Number of channels in the output tensor of the layer
        kernel_size - Kernel size for the convolutional neural network
        stride - Stride for the convolutional neural network
        dilation - Dilation value for the convolutional neural network
        negative_slope - Negative slope setting for the Leaky ReLU activation

        Returns:
        --------
        None
        """
        super().__init__()
        self.weight = weight_variable((out_channels, in_channels, *kernel_size))
        self.bias   = bias_variable((out_channels,))
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size

        self.instnorm = nn.InstanceNorm3d(out_channels, affine=False)
        self.act      = nn.LeakyReLU(negative_slope, inplace=True)

    def forward(self, x):
        """
        Member function for the forward pass of the CNN layer

        Arguments:
        ----------
        x - Tensor that holds the input values to the CNN layer

        Returns:
        --------
        Returns the output tensor
        """
        # pad dims: (D, H, W)
        pad_d = self.dilation * (self.kernel_size[0] - 1) // 2
        pad_h = self.dilation * (self.kernel_size[1] - 1) // 2
        pad_w = self.dilation * (self.kernel_size[2] - 1) // 2
        # F.pad for 5D: (Wl, Wr, Hl, Hr, Dl, Dr)
        x = F.pad(x, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d), mode='reflect')

        x = F.conv3d(x,
                     self.weight,
                     bias=None,
                     stride=self.stride,
                     dilation=self.dilation)

        x = x + self.bias.view(1, -1, 1, 1, 1)
        x = self.instnorm(x)
        x = self.act(x)
        return x


class CNNLayer3dPlain(nn.Module):
    """
    Class that defines a 3D conv + bias only layer, with symmetric padding.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        """
        Default constructor for the CNNLayer3dPlain class

        Arguments:
        ----------
        in_channels - Number of channels in the input tensor to the layer
        out_channels - Number of channels in the output tensor of the layer
        kernel_size - Kernel size for the convolutional neural network
        stride - Stride for the convolutional neural network
        dilation - Dilation value for the convolutional neural network

        Returns:
        --------
        None
        """
        super().__init__()
        self.weight = weight_variable((out_channels, in_channels, *kernel_size))
        self.bias   = bias_variable((out_channels,))
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size

    def forward(self, x):
        """
        Member function for the forward pass of the CNN layer

        Arguments:
        ----------
        x - Tensor that holds the input values to the CNN layer

        Returns:
        --------
        Returns the output tensor
        """
        pad_d = self.dilation * (self.kernel_size[0] - 1) // 2
        pad_h = self.dilation * (self.kernel_size[1] - 1) // 2
        pad_w = self.dilation * (self.kernel_size[2] - 1) // 2
        x = F.pad(x, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d), mode='reflect')
        x = F.conv3d(x,
                     self.weight,
                     bias=None,
                     stride=self.stride,
                     dilation=self.dilation)
        x = x + self.bias.view(1, -1, 1, 1, 1)
        return x