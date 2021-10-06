import numpy as np
from utils.load_data import ComprehensiveRobotDataset
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import matplotlib.pyplot as plt
import time


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class JointsToEndPosition(nn.Module):
    def __init__(self, num_joints=2):
        super(JointsToEndPosition, self).__init__()
        self.layer1 = nn.Linear(num_joints * 2, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        # self.layer4 = nn.Linear(256, 256)
        # self.layer6 = nn.Linear(256, 256)
        self.layer5 = nn.Linear(128, 2)

    def forward(self, x):
        # a = torch.asin(x[:, 0:1])
        # b = torch.acos(x[:, 1:2])
        # c = torch.asin(x[:, 2:3])
        # d = torch.acos(x[:, 3:])
        # print(a.shape)
        # x = torch.cat((a, b, c, d), dim=1)
        x = F.selu(self.layer1(x))
        x = F.selu(self.layer2(x))
        x = F.selu(self.layer3(x))
        # x = F.selu(self.layer4(x))
        # x = F.selu(self.layer6(x))
        x = self.layer5(x)
        return x


def train(writer, name, epoch_idx, data_loader, model, 
    optimizer, criterion, ckpt_path, save_ckpt):

    for idx, (img, joints, task_id, end_position, object_list, target_position, next_joints, displacement) in enumerate(data_loader):
        # Prepare data
        img = img.to(device)
        joints = joints.to(device) / 100
        task_id = task_id.to(device)
        end_position = end_position.to(device)
        object_list = object_list.to(device)
        target_position = target_position.to(device)
        next_joints = next_joints.to(device)
        displacement = displacement.to(device)

        # Forward pass
        optimizer.zero_grad()
        # action_pred, target_position_pred, displacement_pred, displacement_embed, attn_map = model(img, joints, task_id)
        # action_pred2, target_position_pred2, displacement_pred2, displacement_embed2, attn_map2 = model(img, joints, task_id, displacement_embed)
        # loss1 = (criterion(action_pred, next_joints) + criterion(action_pred2, next_joints)) / 2
        # loss5 = (criterion(target_position_pred, target_position) + criterion(target_position_pred2, target_position)) / 2
        # loss6 = (criterion(displacement_pred, displacement) + criterion(displacement_pred2, displacement)) / 2
        position_pred = model(joints)
        loss = criterion(position_pred, end_position)
        # loss = loss1 + loss5
        
        # Backward pass
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        # torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)
        optimizer.step()

        # Log and print
        writer.add_scalar('train loss', loss, global_step=epoch_idx * len(data_loader) + idx)
        print(f'epoch {epoch_idx}, step {idx}, loss {loss.item():.2f}')
        print(joints.detach().cpu().numpy()[0], end_position.detach().cpu().numpy()[0], position_pred.detach().cpu().numpy()[0])
        # print(f'epoch {epoch_idx}, step {idx}, loss1 {loss1.item():.2f}, loss5 {loss5.item():.2f}')
        # print(displacement.detach().cpu().numpy()[0], displacement_pred.detach().cpu().numpy()[0])
        # if epoch_idx * len(data_loader) + idx > 600:
        #     fig = plt.figure(figsize=(10, 5))
        #     attn_map = attn_map.sum(axis=1)
        #     attn_map = attn_map.detach().cpu().numpy()[0].reshape((32, 32))
        #     fig.add_subplot(1, 2, 1)
        #     plt.imshow(attn_map, cmap='Greys')
        #     plt.colorbar()
        #     fig.add_subplot(1, 2, 2)
        #     plt.imshow(img.detach().cpu().numpy()[0])
        #     plt.title(str(task_id.detach().cpu().numpy()[0]))
        #     plt.show()

            # plt.close()
            # plt.cla()

    # Save checkpoint
    if save_ckpt:
        if not os.path.isdir(os.path.join(ckpt_path, name)):
            os.mkdir(os.path.join(ckpt_path, name))
        torch.save(model.state_dict(), os.path.join(ckpt_path, name, f'{epoch_idx}.pth'))


def main(writer, name, batch_size=192):
    ckpt_path = r'/share/yzhou298'
    save_ckpt = False
    add_displacement = True
    accumulate_angles = True

    # load data
    dataset = ComprehensiveRobotDataset(
        data_dir='./data_position_2000/', 
        use_trigonometric_representation=True, 
        use_delta=True,
        normalize=False,
        ending_angles_as_next=True,
        amplify_trigonometric=True,
        add_displacement=add_displacement,
        accumulate_angles=accumulate_angles)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
    # load model
    model = JointsToEndPosition()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    # train n epoches
    for i in range(30):
        train(writer, name, i, data_loader, model, optimizer, 
            criterion, ckpt_path, save_ckpt)


if __name__ == '__main__':
    # Debussy
    name = 'train10-1-joints-to-end-position'
    writer = SummaryWriter('runs/' + name)

    main(writer, name)
