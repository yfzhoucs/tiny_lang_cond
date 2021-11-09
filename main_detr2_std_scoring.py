import numpy as np
from models.backbone_detr2 import Backbone
from utils.load_data import ComprehensiveRobotDataset
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import torch
import os
import matplotlib.pyplot as plt
import time


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def train(writer, name, epoch_idx, data_loader, model, 
    optimizer, criterion, ckpt_path, save_ckpt, add_displacement):

    if not add_displacement:
        for idx, (img, joints, task_id, end_position, object_list, target_position, next_joints) in enumerate(data_loader):
            # Prepare data
            img = img.to(device)
            joints = joints.to(device)
            task_id = task_id.to(device)
            end_position = end_position.to(device)
            object_list = object_list.to(device)
            target_position = target_position.to(device)
            next_joints = next_joints.to(device)

            # Forward pass
            optimizer.zero_grad()
            action_pred, target_position_pred, attn_map = model(img, joints, task_id)
            loss1 = criterion(action_pred, next_joints)
            loss5 = criterion(target_position_pred, target_position)
            loss = loss1 + loss5
            
            # Backward pass
            loss.backward()
            optimizer.step()

            # Log and print
            writer.add_scalar('train loss', loss, global_step=epoch_idx * len(data_loader) + idx)
            print(f'epoch {epoch_idx}, step {idx}, loss1 {loss1.item():.2f}, loss2 {loss2.item():.2f}, loss3 {loss3.item():.2f}, loss4 {loss4.item():.2f}, loss5 {loss5.item():.2f}')
    
    else:
        for idx, (img, joints, task_id, end_position, object_list, target_position, next_joints, displacement) in enumerate(data_loader):
            # Prepare data
            img = img.to(device)
            joints = joints.to(device)
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
            action_pred, target_position_pred, displacement_pred, displacement_embed, attn_map = model(img, joints, task_id)
            loss1 = criterion(action_pred, next_joints)
            loss5 = criterion(target_position_pred, target_position)
            loss6 = criterion(displacement_pred, displacement)
            loss = loss1 + loss5 + loss6
            # loss = loss1 + loss5
            
            # Backward pass
            loss.backward()
            optimizer.step()

            # Log and print
            writer.add_scalar('train loss', loss, global_step=epoch_idx * len(data_loader) + idx)
            print(f'epoch {epoch_idx}, step {idx}, loss1 {loss1.item():.2f}, loss5 {loss5.item():.2f}, loss6 {loss6.item():.2f}')
            # print(f'epoch {epoch_idx}, step {idx}, loss1 {loss1.item():.2f}, loss5 {loss5.item():.2f}')
            # print(displacement.detach().cpu().numpy()[0], displacement_pred.detach().cpu().numpy()[0])
            print(task_id.detach().cpu().numpy()[0], target_position.detach().cpu().numpy()[0])
            # if epoch_idx * len(data_loader) + idx > 300:
            #     fig = plt.figure(figsize=(10, 5))
            #     print(attn_map.shape)
            #     attn_map = attn_map.sum(axis=1).detach().cpu().numpy()[0][0].reshape((32, 32))
            #     # attn_map = attn_map.detach().cpu().numpy()[0].reshape((32, 32))
            #     fig.add_subplot(1, 2, 1)
            #     # plt.imshow(attn_map, cmap='Greys')
            #     plt.imshow(attn_map)
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


def test(writer, name, epoch_idx, data_loader, model, criterion, train_dataset_size, print_attention_map=False):
    with torch.no_grad():
        model.eval()
        loss = 0
        loss5_accu = 0
        idx = 0
        for img, joints, task_id, end_position, object_list, target_position, next_joints, displacement in data_loader:
            global_step = epoch_idx * len(data_loader) + idx

            # Prepare data
            img = img.to(device)
            joints = joints.to(device)
            task_id = task_id.to(device)
            end_position = end_position.to(device)
            object_list = object_list.to(device)
            target_position = target_position.to(device)
            next_joints = next_joints.to(device)
            displacement = displacement.to(device)

            # Forward pass
            action_pred, target_position_pred, displacement_pred, displacement_embed, attn_map = model(img, joints, task_id)
            # target_position_pred, attn_map = model(img, joints, task_id)
            loss1 = criterion(action_pred, next_joints)
            loss5 = criterion(target_position_pred, target_position)
            loss6 = criterion(displacement_pred, displacement)
            loss += loss1.item()
            loss5_accu += loss5.item()

            # Print Attention Map
            if print_attention_map:
                if epoch_idx * len(data_loader) + idx > 1:
                    fig = plt.figure(figsize=(20, 5))
                    # attn_map = attn_map.sum(axis=1)
                    print(attn_map4.shape)
                    attn_map = attn_map4.detach().cpu().numpy()[0][3][:4]
                    fig.add_subplot(1, 4, 1)
                    plt.imshow(attn_map)
                    plt.colorbar()
                    attn_map = attn_map4.detach().cpu().numpy()[0][3][256:]
                    fig.add_subplot(1, 4, 2)
                    plt.imshow(attn_map)
                    plt.colorbar()
                    attn_map = attn_map4.detach().cpu().numpy()[0][3][4:260].reshape((16, 16))
                    fig.add_subplot(1, 4, 3)
                    plt.imshow(attn_map)
                    plt.colorbar()
                    fig.add_subplot(1, 4, 4)
                    plt.imshow(img.detach().cpu().numpy()[0])
                    plt.title(str(task_id.detach().cpu().numpy()[0]))
                    # plt.show()
                    plt.savefig(f"figs/attn_map4_{global_step}.png")
                    print('fig saved')


            idx += 1

            # Print
            print(f'test: epoch {epoch_idx}, step {idx}, loss1 {loss1.item():.2f}, loss5 {loss5.item():.2f}, loss6 {loss6.item():.2f}')

        # Log
        loss /= idx
        loss5_accu /= idx
        writer.add_scalar('test loss', loss, global_step=epoch_idx * train_dataset_size)
        writer.add_scalar('test loss target_position', loss5_accu, global_step=epoch_idx * train_dataset_size)


def main(writer, name, batch_size=128):
    ckpt_path = r'/share/yzhou298'
    save_ckpt = False
    add_displacement = True

    # load data
    dataset = ComprehensiveRobotDataset(
        data_dir='./data_position_2000/', 
        use_trigonometric_representation=True, 
        use_delta=True,
        normalize=False,
        ending_angles_as_next=True,
        amplify_trigonometric=True,
        add_displacement=add_displacement,
        accumulate_angles=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
    dataset_test = ComprehensiveRobotDataset(
        data_dir='./data_position_part3/', 
        use_trigonometric_representation=True, 
        use_delta=True,
        normalize=False,
        ending_angles_as_next=True,
        amplify_trigonometric=True,
        add_displacement=add_displacement,
        accumulate_angles=False)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                          shuffle=False, num_workers=2)
    # load model
    model = Backbone(128, 2, 3, 128, add_displacement=add_displacement)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    # train n epoches
    for i in range(50):
        train(writer, name, i, data_loader, model, optimizer, 
            criterion, ckpt_path, save_ckpt, add_displacement)
        test(writer, name, i + 1, data_loader_test, model, criterion, len(data_loader))


if __name__ == '__main__':
    # Debussy
    name = 'train9-8-attn2-detr2-std-scoring'
    writer = SummaryWriter('runs/' + name)

    main(writer, name)
