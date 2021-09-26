import torch
import torch.nn as nn
import torch.nn.functional as F


# courtesy: https://github.com/darkstar112358/fast-neural-style/blob/master/neural_style/transformer_net.py
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ImgEncoder(nn.Module):
    def __init__(self, img_size, embedding_size, ngf=64):
        super(ImgEncoder, self).__init__()
        input_nc = 3
        self.layer1 = nn.Sequential(nn.ReflectionPad2d((3,3,3,3)),
                                    nn.Conv2d(input_nc,ngf,kernel_size=7,stride=1),
                                    nn.InstanceNorm2d(ngf),
                                    nn.ReLU(True))
        
        self.layer2 = nn.Sequential(nn.Conv2d(ngf,ngf*2,kernel_size=3,stride=2,padding=1),
                                   nn.InstanceNorm2d(ngf*2),
                                   nn.ReLU(True))
        
        self.layer3 = nn.Sequential(nn.Conv2d(ngf*2,ngf*4,kernel_size=3,stride=2,padding=1),
                                   nn.InstanceNorm2d(ngf*4),
                                   nn.ReLU(True))

        self.layer4 = nn.Sequential(ResidualBlock(ngf*4,ngf*4),
                                    ResidualBlock(ngf*4,ngf*4),
                                    ResidualBlock(ngf*4,ngf*4))

        self.flatten = nn.Flatten()

        self.layer5 = nn.Linear(ngf * 4 * img_size * img_size // 16, embedding_size)
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.flatten(out)
        out = F.relu(self.layer5(out))
        return out


if __name__ == '__main__':
    # batch_size = 4
    # img_size = 128
    # embedding_size = 128
    # model = ImgEncoder(img_size, embedding_size)
    # inputs = torch.rand(batch_size, img_size, img_size, 3)
    # outputs = model(inputs)
    # print(len(outputs))
    # print(outputs.shape)

    batch_size = 4
    img_size = 128
    embedding_size = 128
    model = ImgEncoder(img_size, embedding_size)
    print(model)
    # inputs = torch.rand(batch_size, 3, img_size, img_size)
    inputs = torch.rand(batch_size, img_size, img_size, 3)
    outputs = model(inputs)
    print(outputs.shape)
