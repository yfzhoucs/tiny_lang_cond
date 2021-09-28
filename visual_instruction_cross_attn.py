import numpy as np
from models.target_position_predictor import TargetPositionPredictor
from utils.load_data import TargetPositionlDataset
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
    for idx, (img, joints, task_id, end_position, object_list, target_position) in enumerate(data_loader):
        # Prepare data
        img = img.to(device)
        joints = joints.to(device)
        task_id = task_id.to(device)
        end_position = end_position.to(device)
        object_list = object_list.to(device)
        target_position = target_position.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        cross_attn_embedding, target_position_pred, joints_pred, end_position_pred, object_list_pred = model(img, task_id)
        
        # Calculate loss
        loss_joints = criterion(joints_pred, joints)
        loss_end_position = criterion(end_position_pred, end_position)
        loss_object_list = criterion(object_list_pred, object_list)
        loss_target_position = criterion(target_position_pred, target_position)
        loss = loss_joints + loss_end_position + loss_object_list + loss_target_position
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Log and print
        writer.add_scalar('train loss', loss, global_step=epoch_idx * len(data_loader) + idx)
        print(f'epoch {epoch_idx}, step {idx}, l_joints {loss_joints.item():.3f}, l_end_pos {loss_end_position.item():.3f}, l_objects {loss_object_list.item():.3f}, l_targ_pos {loss_target_position:.3f}')
        print(f'all obj {object_list.detach().cpu().numpy()[0]} task_id {task_id.detach().cpu().numpy()[0]} targ_pos {target_position.detach().cpu().numpy()[0]} pred {target_position_pred.detach().cpu().numpy()[0]}')
        print(f'pre obj {object_list_pred.detach().cpu().numpy()[0]}')
        print(f'joints {joints.detach().cpu().numpy()[0]} joints_pred {joints_pred.detach().cpu().numpy()[0]}')

    # Save checkpoint
    if save_ckpt:
        if not os.path.isdir(os.path.join(ckpt_path, name)):
            os.mkdir(os.path.join(ckpt_path, name))
        torch.save(model.state_dict(), os.path.join(ckpt_path, name, f'{epoch_idx}.pth'))


def main(writer, name, batch_size=50):
    ckpt_path = r'/share/yzhou298'
    save_ckpt = False
    # load data
    dataset = TargetPositionlDataset(
        data_dir='./data_position/', use_trigonometric_representation=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
    # load model
    model = TargetPositionPredictor(img_size=128, num_joints=2, embedding_size=128, num_tasks=3)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    # train n epoches
    for i in range(1):
        train(writer, name, i, data_loader, model, optimizer, criterion, ckpt_path, save_ckpt)


if __name__ == '__main__':
    name = 'train7-2-target-position-mse-cat-bs50-angle-amp'
    writer = SummaryWriter('runs/' + name)

    main(writer, name)