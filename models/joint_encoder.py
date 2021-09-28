import torch
import torch.nn as nn
import torch.nn.functional as F


class JointEncoder(nn.Module):
    def __init__(self, num_joints, embedding_size):
        super(JointEncoder, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(num_joints, 32 * 32)
        self.layer2 = nn.Linear(32 * 32, 16 * 16)
        self.layer3 = nn.Linear(16 * 16, embedding_size)

    def forward(self, joints):
        x = self.flatten(joints)
        x = F.selu(self.layer1(x))
        x = F.selu(self.layer2(x))
        x = F.selu(self.layer3(x))
        return x


if __name__ == '__main__':
    batch_size = 4
    num_joints = 2
    embedding_size = 128
    model = JointEncoder(num_joints, embedding_size)
    inputs = torch.rand(batch_size, 2)
    outputs = model(inputs)
    print(len(outputs))
    print(outputs.shape)
