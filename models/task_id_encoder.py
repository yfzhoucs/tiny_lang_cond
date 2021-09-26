import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskIDEncoder(nn.Module):
    def __init__(self, num_tasks, embedding_size):
        super(TaskIDEncoder, self).__init__()
        self.id_to_embedding = nn.Embedding(num_tasks, 32 * 32)
        self.layer1 = nn.Linear(32 * 32, 16 * 16)
        self.layer2 = nn.Linear(16 * 16, embedding_size)

    def forward(self, task_id):
        x = self.id_to_embedding(task_id)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return x


if __name__ == '__main__':
    num_tasks = 3
    model = TaskIDEncoder(num_tasks, 128)
    batch_size = 4
    inputs = torch.randint(num_tasks, (batch_size,))
    outputs = model(inputs)
    print(len(outputs))
    print(outputs.shape)
