import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


def conv3x3(in_planes, out_planes):
    """
    conv2d with kernel_size 3.
    padding = 1, stride = 1, bias=False.
    """
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=1, padding=1, bias=False)

def conv5x5(in_planes, out_planes):
    """
    conv2d with kernel_size 5.
    padding = 0, stride = 2, bias=False.
    """
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=5, stride=2, padding=0, bias=False)

class residualBlock(nn.Module):
    """
    Simple ResNet residual block.
    """
    def __init__(self, in_planes, out_planes):
        super(residualBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.elu1 = nn.ELU(inplace=True)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.elu2 = nn.ELU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.elu2(out)

        return out


class EmbeddingNet(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(EmbeddingNet, self).__init__()

        self.conv32 = conv5x5(in_planes=1, out_planes=32)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.elu1 = nn.ELU()
        self.residual_block1 = self._make_layer(in_planes=32, out_planes=32)
        self.residual_block1 = nn.DataParallel(self.residual_block1)

        self.conv64 = conv5x5(in_planes=32, out_planes=64)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.elu2 = nn.ELU()
        self.residual_block2 = self._make_layer(in_planes=64, out_planes=64)
        self.residual_block2 = nn.DataParallel(self.residual_block2)

        self.conv128 = conv5x5(in_planes=64, out_planes=128)
        self.bn3 = nn.BatchNorm2d(num_features=128)
        self.elu3 = nn.ELU()
        self.residual_block3 = self._make_layer(in_planes=128, out_planes=128)
        self.residual_block3 = nn.DataParallel(self.residual_block3)

        self.conv256 = conv5x5(in_planes=128, out_planes=256)
        self.bn4 = nn.BatchNorm2d(num_features=256)
        self.elu4 = nn.ELU()
        self.residual_block4 = self._make_layer(in_planes=256, out_planes=256)
        self.residual_block4 = nn.DataParallel(self.residual_block4)

        self.avgpool = nn.AdaptiveAvgPool2d((1, None))
        self.fc = nn.Linear(256, embedding_size)
        self.classifier = nn.Linear(embedding_size, num_classes)
        self.classifier = nn.DataParallel(self.classifier)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        self.alpha = 16

    def _make_layer(self, in_planes, out_planes):
        layers = []
        layers.append(residualBlock(in_planes, out_planes))

        for _ in range(2):
            layers.append(residualBlock(in_planes, out_planes))

        return nn.Sequential(*layers)

    def l2_norm(self, out):
        input_size = out.size()
        buffer = torch.pow(out, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(out, norm.view(-1,1).expand_as(out))
        output = _output.view(input_size)
        return output

    def embedding_forward(self, x):
        out = self.conv32(x)
        out = self.bn1(out)
        out = self.elu1(out)
        out = self.residual_block1(out)

        out = self.conv64(out)
        out = self.bn2(out)
        out = self.elu1(out)
        out = self.residual_block2(out)

        out = self.conv128(out)
        out = self.bn3(out)
        out = self.elu3(out)
        out = self.residual_block3(out)

        out = self.conv256(out)
        out = self.bn4(out)
        out = self.elu4(out)
        out = self.residual_block4(out)

        # temporal pooling layer
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        # linear layer
        out = self.fc(out)
        # length normalization layer
        out = self.l2_norm(out)
        # out = out / F.normalize(out, p=2)
        out = out * self.alpha
        # out is the embedding of an utterance
        return out

    def forward(self, x):
        embedding = self.embedding_forward(x)
        speaker = self.classifier(embedding)
        return speaker



