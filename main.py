import numpy as np
from models.backbone import Backbone
from utils.load_data import RobotDataset
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import torch
import os


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def train(writer, name, epoch_idx, data_loader, model, optimizer, criterion, ckpt_path, save_ckpt):
    for idx, (img, joints, task_id, next_joints) in enumerate(data_loader):
        img = img.to(device)
        joints = joints.to(device)
        task_id = task_id.to(device)
        next_joints = next_joints.to(device)
        optimizer.zero_grad()
        pred_joints = model(img, joints, task_id)
        loss = criterion(pred_joints, next_joints)
        loss.backward()
        optimizer.step()
        writer.add_scalar('train loss', loss, global_step=epoch_idx * len(data_loader) + idx)
        print(f'epoch {epoch_idx}, step {idx}, loss {loss.item()}')
    if not os.path.isdir(os.path.join(ckpt_path, name)):
        os.mkdir(os.path.join(ckpt_path, name))
    if save_ckpt:
        torch.save(model.state_dict(), os.path.join(ckpt_path, name, f'{epoch_idx}.pth'))


def main(writer, name, batch_size=4):
    ckpt_path = r'/share/yzhou298'
    save_ckpt = False
    # load data
    dataset = RobotDataset(data_dir='./data_small/')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
    # load model
    model = Backbone(128, 2, 3, 128)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    # train n epoches
    for i in range(100):
        train(writer, name, i, data_loader, model, optimizer, criterion, ckpt_path, save_ckpt)
    # test
    # save checkpoint


if __name__ == '__main__':
    name = 'train2-2-conv-img-encoder-relu-5-traces'
    writer = SummaryWriter('runs/' + name)

    main(writer, name)