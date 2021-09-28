import torch
import torch.nn as nn
import torch.nn.functional as F


class Controller(nn.Module):
    def __init__(self, embedding_size, num_joints):
        super(Controller, self).__init__()
        self.layer1 = nn.Linear(embedding_size, 16 * 16)
        self.layer2 = nn.Linear(16 * 16, num_joints)

    def forward(self, x):
        x = F.selu(self.layer1(x))
        x = self.layer2(x)
        return x


if __name__ == '__main__':
    batch_size = 4
    num_joints = 2
    embedding_size = 128
    model = Controller(embedding_size, num_joints)
    inputs = torch.rand(batch_size, embedding_size)
    outputs = model(inputs)
    print(len(outputs))
    print(outputs.shape)
