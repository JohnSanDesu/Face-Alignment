"""
This py file contains the initiation of CNN.
"""
import torch
import torch.nn as nn

class Net(nn.Module):
    """
    Define CNN

    Attributes:
        block_1 (nn.Sequential): The first convolutional block. It contains 2 convolutional layers and 2 activation function (ReLU) layer and 1 max pooling layer
        block_2 (nn.Sequential): The second convolutional block. The structure of the block is the same as block_1.
        classifier (nn.Sequential):  Full connected layer. Each layer consists with flatten function and linear transformation function
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        """
        Initialize the layer of the network

        Args:
            input_shape (int): The number of input channel
            hidden_units (int): The number of hidden units
            output_shape (int): The size of the output
        """
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 56 * 56,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        """
        Define the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: output of the network
        """
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)

        return x
