import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Basic building blocks (BasicBlock and Bottleneck) - standard ResNet blocks
# -----------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


# -----------------------------------------------------------------------------
# ResNet variant tailored for STL-10 (96x96 inputs)
# - Stem: Conv3x3 (stride=2) + BN + ReLU + MaxPool(2)  -> reduces 96 -> 48 -> 24
# - Standard ResNet stages with downsampling at stage 2/3/4
# - AdaptiveAvgPool -> Linear classifier
# -----------------------------------------------------------------------------
class ResNetSTL(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, channels=3):
        """
        ResNet variant tuned for STL-10 (96x96).
        Args:
            block: block class (BasicBlock or Bottleneck)
            num_blocks: list of block counts per stage, e.g. [2,2,2,2] for ResNet18
            num_classes: number of output classes
            channels: input channels (3 for RGB)
        """
        super(ResNetSTL, self).__init__()
        base_channels = [64, 128, 256, 512]

        self.in_planes = base_channels[0]
        # Stem: small downsampling to handle 96x96 inputs efficiently
        # Conv3x3 stride=2 reduces 96 -> 48
        self.conv1 = nn.Conv2d(
            channels, base_channels[0], kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(base_channels[0])
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # 48 -> 24

        # Residual stages
        self.layer1 = self._make_layer(block, base_channels[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base_channels[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, base_channels[3], num_blocks[3], stride=2)

        # Pooling and classifier
        out_channels = base_channels[3] * block.expansion
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(out_channels, num_classes)

        # expose embedding dim (useful for selection/AL code)
        self.embDim = out_channels

        # weight initialization will typically be handled by the outer code,
        # but include a safe default initialization here as well.
        # self._init_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, stride=s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, last=False, freeze=False):
        """
        forward(..., last=True) -> returns (logits, embeddings)
        forward(..., last=False) -> returns logits
        freeze=True runs forward with torch.no_grad() for embedding extraction
        """
        if freeze:
            with torch.no_grad():
                out = F.relu(self.bn1(self.conv1(x)), inplace=True)
                out = self.maxpool(out)
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
                out = self.avgpool(out)
                emb = out.view(out.size(0), -1)
        else:
            out = F.relu(self.bn1(self.conv1(x)), inplace=True)
            out = self.maxpool(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.avgpool(out)
            emb = out.view(out.size(0), -1)

        logits = self.linear(emb)
        if last:
            return logits, emb
        return logits

    def get_embedding_dim(self):
        return self.embDim

    # def _init_weights(self):
    #     # He / Kaiming initialization for convs, ones for BN weights, zeros for biases
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    #         elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
    #             if getattr(m, "weight", None) is not None:
    #                 nn.init.ones_(m.weight)
    #             if getattr(m, "bias", None) is not None:
    #                 nn.init.zeros_(m.bias)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 1e-3)
    #             if getattr(m, "bias", None) is not None:
    #                 nn.init.zeros_(m.bias)


# -----------------------------------------------------------------------------
# Convenience constructors
# -----------------------------------------------------------------------------
def ResNet18_STL(num_classes=10, channels=3):
    return ResNetSTL(BasicBlock, [2, 2, 2, 2], num_classes, channels)


def ResNet34_STL(num_classes=10, channels=3):
    return ResNetSTL(BasicBlock, [3, 4, 6, 3], num_classes, channels)


def ResNet50_STL(num_classes=10, channels=3):
    return ResNetSTL(Bottleneck, [3, 4, 6, 3], num_classes, channels)



