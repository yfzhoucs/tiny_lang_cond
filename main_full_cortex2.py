import numpy as np
from models.backbone_full_cortex2 import Backbone
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


def pixel_position_to_attn_index(pixel_position, attn_map_offset=4):
    pixel_position = pixel_position.detach().cpu().numpy() + 64
    index = (pixel_position[:, 0]) // 8 + attn_map_offset + 16 * (15 - pixel_position[:, 1] // 8)
    index = index.astype(int)
    index = torch.tensor(index).to(device).unsqueeze(1)
    return index


def train(writer, name, epoch_idx, data_loader, model, 
    optimizer, criterion, ckpt_path, save_ckpt, loss_stage,
    print_attention_map=False, curriculum_learning=False, supervised_attn=False):
    model.train()
    criterion2 = nn.CrossEntropyLoss()
    loss_array = np.array([100000.0] * 50)
    for idx, (img, joints, task_id, end_position, object_list, target_position, next_joints, displacement) in enumerate(data_loader):
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
        optimizer.zero_grad()
        action_pred, target_position_pred, displacement_pred, attn_map, attn_map2, attn_map3, attn_map4, joints_pred = model(img, joints, task_id)
        # target_position_pred, attn_map = model(img, joints, task_id)
        loss1 = criterion(action_pred, next_joints)
        loss5 = criterion(target_position_pred, target_position)
        loss6 = criterion(displacement_pred, displacement)
        loss7 = criterion(joints_pred, joints)
        # loss8 = criterion2(task_pred, task_id)

        # Attention Supervison for Target Position Task
        target_attn = attn_map[:, 0, -1]
        # target_attn = torch.gather(attn_map[:, -1, :], 1, pixel_position_to_attn_index(target_position, attn_map_offset=4))
        loss_attn_target = criterion(target_attn, torch.ones(attn_map.shape[0], 1, dtype=torch.float32).to(device))
        target_position_attn = torch.gather(attn_map2[:, 0, :], 1, pixel_position_to_attn_index(target_position, attn_map_offset=4))
        # target_position_attn = attn_map2[:, 0, -1]
        loss_attn_target_position = criterion(target_position_attn, torch.ones(attn_map2.shape[0], 1, dtype=torch.float32).to(device))

        # Attention Supervision for Displacement Task
        end_effector_attn = torch.gather(attn_map2[:, 2, :], 1, pixel_position_to_attn_index(end_position, attn_map_offset=4))
        loss_end_effector_attn = criterion(end_effector_attn, torch.ones(attn_map2.shape[0], 1, dtype=torch.float32).to(device))
        displacement_attn = attn_map3[:, 2, 0]
        loss_displacement_attn = criterion(displacement_attn, torch.ones(attn_map3.shape[0], 1, dtype=torch.float32).to(device))

        # Attention Supervision for Joints Task
        joints_attn = attn_map3[:, 3, -2]
        loss_joints_attn = criterion(displacement_attn, torch.ones(attn_map3.shape[0], 1, dtype=torch.float32).to(device))

        # Calculate loss
        if curriculum_learning:
            if loss_array.mean() < 75:
                loss_stage += 1
            if loss_stage == -1:
                if supervised_attn:
                    loss = loss8 + loss_attn_target * 5000
                else:
                    loss = loss8
            elif loss_stage == 0:
                if supervised_attn:
                    # loss = loss5 + loss_attn_target_position * 5000# + loss_attn_target * 5000

                    #######################################################
                    # This line works perfectly for target position prediction and displacement prediction
                    # loss = loss_attn_target_position * 5000 + loss_attn_target * 5000 + loss5 + loss6 + loss_end_effector_attn * 5000# + loss_displacement_attn * 5000
                    #######################################################

                    #######################################################
                    # This line works well for tar pos, disp and joints. Removing the last attn loss will make it converge faster
                    # loss = loss_attn_target_position * 5000 + loss_attn_target * 5000 + loss5 + loss6 + loss_end_effector_attn * 5000 + loss7 + loss_joints_attn * 5000
                    #######################################################

                    loss = loss_attn_target_position * 5000 + loss_attn_target * 5000 + loss5 + loss6 + loss_end_effector_attn * 5000 + loss7 + loss1

                    print(loss_attn_target.item() * 5000, loss_attn_target_position.item() * 5000)
                else:
                    loss = loss5
                # print(loss5.item())
            elif loss_stage == 1:
                loss = loss5 + loss6# + loss_attn_end_position * 1000 + loss_target_position_attn2 * 1000
            elif loss_stage == 2:
                loss = loss5 + loss6 + loss7
            else:
                loss = loss1 + loss5 + loss6 + loss7
            loss_array[global_step % len(loss_array)] = loss.item()
            # if loss_stage == 0:
            #     loss_array[global_step % len(loss_array)] -= loss_attn_target_position * 5000
            # elif loss_stage == 1:
            #     loss_array[global_step % len(loss_array)] -= loss_attn_end_position * 1000 + loss_target_position_attn2 * 1000
        else:
            if not supervised_attn:
                loss = loss1 + loss5 + loss6 + loss7
            else:
                loss = loss_attn_target_position * 5000 + loss_attn_target * 5000 + loss5 + loss6 + loss_end_effector_attn * 5000 + loss7 + loss1
            loss_array[global_step % len(loss_array)] = loss.item()
        
        # Backward pass
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        # torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)
        optimizer.step()

        # Log and print
        writer.add_scalar('train loss', loss_array[global_step % len(loss_array)], global_step=epoch_idx * len(data_loader) + idx)
        # print(f'epoch {epoch_idx}, step {idx}, stage {loss_stage}, l_all {loss.item():.2f}, l1 {loss1.item():.2f}, l5 {loss5.item():.2f}, l6 {loss6.item():.2f}, l7 {loss7.item():.2f}, l8 {loss8.item():.2f}')
        print(f'epoch {epoch_idx}, step {idx}, stage {loss_stage}, l_all {loss.item():.2f}, l1 {loss1.item():.2f}, l5 {loss5.item():.2f}, l6 {loss6.item():.2f}, l7 {loss7.item():.2f}')

        # Print Attention Map
        if print_attention_map:
            if epoch_idx * len(data_loader) + idx > 1:
                fig = plt.figure(figsize=(10, 5))
                # attn_map = attn_map.sum(axis=1)
                print(attn_map.shape)
                # attn_map = attn_map.detach().cpu().numpy()[0][0][4:].reshape((16, 16))
                supervised_attn_map_idx = pixel_position_to_attn_index(target_position, attn_map_offset=0)
                supervised_attn_map = np.zeros((256))
                supervised_attn_map[supervised_attn_map_idx[0]] = 1
                supervised_attn_map = supervised_attn_map.reshape((16, 16))
                print(supervised_attn_map.shape)
                fig.add_subplot(1, 2, 1)
                plt.imshow(supervised_attn_map)
                plt.colorbar()
                fig.add_subplot(1, 2, 2)
                plt.imshow(img.detach().cpu().numpy()[0])
                plt.title(str(task_id.detach().cpu().numpy()[0]))
                plt.show()

    # Save checkpoint
    if save_ckpt:
        if not os.path.isdir(os.path.join(ckpt_path, name)):
            os.mkdir(os.path.join(ckpt_path, name))
        torch.save(model.state_dict(), os.path.join(ckpt_path, name, f'{epoch_idx}.pth'))
    return loss_stage


def test(writer, name, epoch_idx, data_loader, model, criterion, train_dataset_size, print_attention_map=False):
    with torch.no_grad():
        model.eval()
        loss = 0
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
            action_pred, target_position_pred, displacement_pred, attn_map, attn_map2, attn_map3, attn_map4, joints_pred = model(img, joints, task_id)
            # target_position_pred, attn_map = model(img, joints, task_id)
            loss1 = criterion(action_pred, next_joints)
            loss5 = criterion(target_position_pred, target_position)
            loss6 = criterion(displacement_pred, displacement)
            loss7 = criterion(joints_pred, joints)
            loss += loss1.item()

            # Print Attention Map
            if print_attention_map:
                if epoch_idx * len(data_loader) + idx > 200:
                    fig = plt.figure(figsize=(10, 5))
                    # attn_map = attn_map.sum(axis=1)
                    print(attn_map.shape)
                    attn_map = attn_map.detach().cpu().numpy()[0][0][4:].reshape((16, 16))
                    fig.add_subplot(1, 2, 1)
                    plt.imshow(attn_map)
                    plt.colorbar()
                    fig.add_subplot(1, 2, 2)
                    plt.imshow(img.detach().cpu().numpy()[0])
                    plt.title(str(task_id.detach().cpu().numpy()[0]))
                    plt.show()

            idx += 1

            # Print
            print(f'test: epoch {epoch_idx}, step {idx}, loss1 {loss1.item():.2f}, loss5 {loss5.item():.2f}, loss6 {loss6.item():.2f}, loss7 {loss7.item():.2f}')

        # Log
        loss /= idx
        writer.add_scalar('test loss', loss, global_step=epoch_idx * train_dataset_size)


def main(writer, name, batch_size=128):
    ckpt_path = r'/share/yzhou298'
    save_ckpt = False
    add_displacement = True
    accumulate_angles = False
    supervised_attn = True
    curriculum_learning = False
    print_attention_map = False

    # load data
    dataset_train = ComprehensiveRobotDataset(
        data_dir='./data_position_2000/', 
        use_trigonometric_representation=True, 
        use_delta=True,
        normalize=False,
        ending_angles_as_next=True,
        amplify_trigonometric=True,
        add_displacement=add_displacement,
        accumulate_angles=accumulate_angles)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
    dataset_test = ComprehensiveRobotDataset(
        data_dir='./data_position_part3/', 
        use_trigonometric_representation=True, 
        use_delta=True,
        normalize=False,
        ending_angles_as_next=True,
        amplify_trigonometric=True,
        add_displacement=add_displacement,
        accumulate_angles=accumulate_angles)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                          shuffle=False, num_workers=2)
    # load model
    model = Backbone(128, 2, 3, 192, add_displacement=add_displacement)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters())
    # optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    # train n epoches
    loss_stage = 0
    for i in range(500):
        loss_stage = train(writer, name, i, data_loader_train, model, optimizer, 
            criterion, ckpt_path, save_ckpt, loss_stage, supervised_attn=supervised_attn, curriculum_learning=curriculum_learning, print_attention_map=print_attention_map)
        test(writer, name, i + 1, data_loader_test, model, criterion, len(data_loader_train))

if __name__ == '__main__':
    # Debussy
    name = 'train14-1-full-cortex-all-losses-attn-head8'
    writer = SummaryWriter('runs/' + name)

    main(writer, name)
