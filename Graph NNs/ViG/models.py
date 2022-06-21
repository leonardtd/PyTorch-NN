import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import *


class IsotropicVIG(nn.Module):
    def __init__(self, in_channels, num_blocks, patch_size=4, kernel_size=16):
        super().__init__()

        self.num_blocks = num_blocks
        self.patch_size = patch_size

        self.patchifier = Patchifier(
            patch_size=self.patch_size, hidden_channels=in_channels
        )

        self.network = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_blocks):
            self.network.append(Block(in_channels, kernel_size))
            if i < num_blocks-1:
                self.norms.append(nn.BatchNorm2d(in_channels))

        self.reset_parameters()

    def forward(self, x):

        x = self.patchifier(x)

        for i, block in enumerate(self.network):
            x = block(x)

            if i < self.num_blocks - 1:
                x = F.gelu(x)
                x = self.norms[i](x)

        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def test_iso_vig():
    # Tiny config
    model = IsotropicVIG(in_channels=192, num_blocks=1, patch_size=4, kernel_size=16)

    from torchsummary import summary
    summary(model, (3, 224, 224))

    x = torch.randn(1, 3, 224, 224)
    print(model(x).shape)  # torch.Size([`batch_size`, `in_channels`, 56, 56])


if __name__ == "__main__":
    test_iso_vig()
