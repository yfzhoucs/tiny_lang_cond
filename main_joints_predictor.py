import numpy as np
from models.backbone_joint_head import Backbone
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


def train(writer, name, epoch_idx, data_loader, model, 
    optimizer, criterion, ckpt_path, save_ckpt,
    train_loss1=True, train_loss2=True):
    assert train_loss1 or train_loss2
    for idx, (img, joints, task_id, next_joints) in enumerate(data_loader):
        img = img.to(device)
        joints = joints.to(device)
        task_id = task_id.to(device)
        next_joints = next_joints.to(device)

        optimizer.zero_grad()
        pred_joints, recognized_joints = model(img, joints, task_id)
        loss1 = criterion(pred_joints, next_joints)
        loss2 = criterion(recognized_joints, joints)
        loss = 0
        if train_loss1:
            loss = loss1
        if train_loss2:
            loss = loss + loss2
        loss.backward()
        optimizer.step()

        writer.add_scalar('train loss', loss, global_step=epoch_idx * len(data_loader) + idx)
        print(f'epoch {epoch_idx}, step {idx}, loss1 {loss1.item()}, loss2 {loss2.item()}')
    if save_ckpt:
        if not os.path.isdir(os.path.join(ckpt_path, name)):
            os.mkdir(os.path.join(ckpt_path, name))
        torch.save(model.state_dict(), os.path.join(ckpt_path, name, f'{epoch_idx}.pth'))


def main(writer, name, batch_size=4):
    ckpt_path = r'/share/yzhou298'
    save_ckpt = True
    # load data
    dataset = RobotDataset(
        data_dir='./data/', 
        use_trigonometric_representation=True, 
        use_delta=True,
        normalize=True,
        ending_angles_as_next=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
    # load model
    model = Backbone(128, 2, 3, 128)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.L1Loss()

    # train n epoches
    for i in range(1):
        train(writer, name, 2 * i, data_loader, model, optimizer, 
            criterion, ckpt_path, save_ckpt,
            train_loss1=False, train_loss2=True)
        train(writer, name, 2 * i + 1, data_loader, model, optimizer, 
            criterion, ckpt_path, save_ckpt,
            train_loss1=True, train_loss2=False)
    # test
    # save checkpoint


if __name__ == '__main__':
    name = 'train5-4-pred-ending-motion'
    # name = 'train4-1-traces-aux-loss-joint-angle-l1'
    writer = SummaryWriter('runs/' + name)

    main(writer, name)