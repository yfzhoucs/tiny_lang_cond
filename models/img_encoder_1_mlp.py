import torch
import torch.nn as nn
import torch.nn.functional as F


class ImgEncoder(nn.Module):
    def __init__(self, img_size, embedding_size):
        super(ImgEncoder, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(img_size * img_size * 3, 64 * 64)
        self.layer2 = nn.Linear(64 * 64, 32 * 32)
        self.layer3 = nn.Linear(32 * 32, embedding_size)

    def forward(self, img):
        x = self.flatten(img)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return x


if __name__ == '__main__':
    batch_size = 4
    img_size = 128
    embedding_size = 128
    model = ImgEncoder(img_size, embedding_size)
    inputs = torch.rand(batch_size, img_size, img_size, 3)
    outputs = model(inputs)
    print(len(outputs))
    print(outputs.shape)
