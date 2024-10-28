from torch import Tensor
from torch.nn import Module, Conv2d, Linear, functional


class Network(Module):
    def __init__(self, inFrames: int, outputDimension: int):
        super().__init__()  # type: ignore
        # initializes layers with kaiming uniform
        self.conv2d_1 = Conv2d(
            in_channels=inFrames, out_channels=16, kernel_size=8, stride=4
        )
        self.conv2d_2 = Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.linear_1 = Linear(2592, 256)
        self.linear_2 = Linear(256, outputDimension)

    def forward(self, input: int):
        # input (1,84,84)
        x = self.conv2d_1(input)
        x = functional.relu(x)
        x = self.conv2d_2(x)
        x = functional.relu(x)
        x = x.flatten()
        x = self.linear_1(x)
        x = functional.relu(x)
        x = self.linear_2(x)
        # output (outputDimension)
        return x

    def forward_batch(self, input: Tensor):
        # input (Batch,1,84,84)
        x = self.conv2d_1(input)
        x = functional.relu(x)
        x = self.conv2d_2(x)
        x = functional.relu(x)
        x = x.flatten(start_dim=1)
        x = self.linear_1(x)
        x = functional.relu(x)
        x = self.linear_2(x)
        # output (Batch, outputDimension)
        return x
