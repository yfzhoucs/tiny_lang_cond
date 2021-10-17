import numpy as np
from models.backbone_supervised_attn import Backbone
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
    # print(pixel_position[0, 0], pixel_position[0, 0] // 8)
    # print(pixel_position[0, 1], 15 - pixel_position[0, 1] // 8)
    index = index.astype(int)
    index = torch.tensor(index).to(device).unsqueeze(1)
    return index


def train(writer, name, epoch_idx, data_loader, model, 
    optimizer, criterion, ckpt_path, save_ckpt, loss_stage):
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
        # action_pred, target_position_pred, displacement_pred, displacement_embed, attn_map = model(img, joints, task_id)
        # action_pred2, target_position_pred2, displacement_pred2, displacement_embed2, attn_map2 = model(img, joints, task_id, displacement_embed)
        # loss1 = (criterion(action_pred, next_joints) + criterion(action_pred2, next_joints)) / 2
        # loss5 = (criterion(target_position_pred, target_position) + criterion(target_position_pred2, target_position)) / 2
        # loss6 = (criterion(displacement_pred, displacement) + criterion(displacement_pred2, displacement)) / 2
        # if epoch_idx < 1:
        #     attn_mask = np.ones((260, 260))
        #     attn_mask[:, 3] = -10000000
        #     attn_mask[3, :] = -10000000
        #     attn_mask[:, 1] = -10000000
        #     attn_mask[1, :] = -10000000
        #     attn_mask = torch.tensor(attn_mask, dtype=torch.float32).to(device)
        # elif epoch_idx < 3:
        #     attn_mask = np.ones((260, 260))
        #     attn_mask[:, 3] = -10000000
        #     attn_mask[3, :] = -10000000
        #     attn_mask[:, 1] = -10000000
        #     attn_mask[1, :] = -10000000
        #     attn_mask = torch.tensor(attn_mask, dtype=torch.float32).to(device)
        # elif epoch_idx < 5:
        #     attn_mask = np.ones((260, 260))
        #     # attn_mask[:, 3] = -10000000
        #     # attn_mask[3, :] = -10000000
        #     attn_mask = torch.tensor(attn_mask, dtype=torch.float32).to(device)
        # else:
        #     attn_mask = np.ones((260, 260))
        #     # attn_mask[:, 3] = 0
        #     # attn_mask[3, :] = 0
        #     attn_mask = torch.tensor(attn_mask, dtype=torch.float32).to(device)
        # if loss_stage == 0:
        #     attn_mask = np.ones((260, 260), dtype=float) * 10000000
        #     attn_mask[:, 3] = -10000000
        #     attn_mask[3, :] = -10000000
        #     attn_mask[:, 1] = -10000000
        #     attn_mask[1, :] = -10000000
        #     attn_mask = torch.tensor(attn_mask, dtype=torch.float32).to(device)
        # elif loss_stage == 1:
        #     attn_mask = np.ones((260, 260), dtype=float)
        #     attn_mask[:, 3] = -10000000
        #     attn_mask[3, :] = -10000000
        #     attn_mask[:, 1] = -10000000
        #     attn_mask[1, :] = -10000000
        #     attn_mask = torch.tensor(attn_mask, dtype=torch.float32).to(device)
        # elif loss_stage == 2:
        #     attn_mask = np.ones((260, 260), dtype=float)
        #     attn_mask = torch.tensor(attn_mask, dtype=torch.float32).to(device)
        # else:
        #     attn_mask = np.ones((260, 260), dtype=float)
        #     attn_mask = torch.tensor(attn_mask, dtype=torch.float32).to(device)



        action_pred, target_position_pred, displacement_pred, displacement_embed, attn_map, joints_pred = model(img, joints, task_id)
        # target_position_pred, attn_map = model(img, joints, task_id)
        loss1 = criterion(action_pred, next_joints)
        loss5 = criterion(target_position_pred, target_position)
        loss6 = criterion(displacement_pred, displacement)
        loss7 = criterion(joints_pred, joints)

        target_position_attn = torch.gather(attn_map[:, 0, :], 1, pixel_position_to_attn_index(target_position, attn_map_offset=4))
        loss_attn_target_position = criterion(target_position_attn, torch.ones(attn_map.shape[0], 1, dtype=torch.float32).to(device))
        end_position_attn = torch.gather(attn_map[:, 3, :], 1, pixel_position_to_attn_index(end_position, attn_map_offset=4))
        loss_attn_end_position = criterion(end_position_attn, torch.ones(attn_map.shape[0], 1, dtype=torch.float32).to(device) / 2)
        target_position_attn2 = torch.gather(attn_map[:, 3, :], 1, pixel_position_to_attn_index(target_position, attn_map_offset=4))
        loss_target_position_attn2 = criterion(target_position_attn2, torch.ones(attn_map.shape[0], 1, dtype=torch.float32).to(device) / 2)


        if loss_array.mean() < 75:
            loss_stage += 1

        if loss_stage == 0:
            loss = loss5 + loss_attn_target_position * 5000
            # print(loss5.item())
        elif loss_stage == 1:
            loss = loss5 + loss6# + loss_attn_end_position * 1000 + loss_target_position_attn2 * 1000
        elif loss_stage == 2:
            loss = loss5 + loss6 + loss7
        else:
            loss = loss1 + loss5 + loss6 + loss7
        loss_array[global_step % len(loss_array)] = loss.item()
        if loss_stage == 0:
            loss_array[global_step % len(loss_array)] -= loss_attn_target_position * 5000
        # elif loss_stage == 1:
        #     loss_array[global_step % len(loss_array)] -= loss_attn_end_position * 1000 + loss_target_position_attn2 * 1000

        # loss = loss5 + loss_attn_target_position * 5000
        
        # Backward pass
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        # torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)
        optimizer.step()

        # Log and print
        writer.add_scalar('train loss', loss_array[global_step % len(loss_array)], global_step=epoch_idx * len(data_loader) + idx)
        print(f'epoch {epoch_idx}, step {idx}, loss_stage {loss_stage}, loss1 {loss1.item():.2f}, loss5 {loss5.item():.2f}, loss6 {loss6.item():.2f}, loss7 {loss7.item():.2f}')
        # print(f'epoch {epoch_idx}, step {idx}, loss_stage {loss_stage}, loss5 {loss5.item():.2f}, loss_array {loss_array.mean()}')
        # print(f'epoch {epoch_idx}, step {idx}, loss1 {loss1.item():.2f}, loss5 {loss5.item():.2f}')
        # print(displacement.detach().cpu().numpy()[0], displacement_pred.detach().cpu().numpy()[0])
        print(task_id.detach().cpu().numpy()[0], target_position.detach().cpu().numpy()[0])
        # if epoch_idx * len(data_loader) + idx > 200:
        #     fig = plt.figure(figsize=(10, 5))
        #     # attn_map = attn_map.sum(axis=1)
        #     print(attn_map.shape)
        #     attn_map = attn_map.detach().cpu().numpy()[0][0][4:].reshape((16, 16))
        #     fig.add_subplot(1, 2, 1)
        #     # plt.imshow(attn_map, cmap='Greys')
        #     plt.imshow(attn_map)
        #     plt.colorbar()
        #     fig.add_subplot(1, 2, 2)
        #     plt.imshow(img.detach().cpu().numpy()[0])
        #     plt.title(str(task_id.detach().cpu().numpy()[0]))
        #     plt.show()


        # if global_step > 1000:
        #     fig = plt.figure(figsize=(10, 5))
        #     attn_supervision = np.zeros((256))
        #     attn_supervision[pixel_position_to_attn_index(target_position, attn_map_offset=0).detach().cpu().numpy()[0]] = 1
        #     attn_supervision = attn_supervision.reshape((16, 16))
        #     fig.add_subplot(1, 2, 1)
        #     plt.imshow(attn_supervision, cmap='Greys')
        #     plt.colorbar()
        #     fig.add_subplot(1, 2, 2)
        #     plt.imshow(img.detach().cpu().numpy()[0])
        #     plt.title(str(task_id.detach().cpu().numpy()[0]))
        #     plt.show()

    # Save checkpoint
    if save_ckpt:
        if not os.path.isdir(os.path.join(ckpt_path, name)):
            os.mkdir(os.path.join(ckpt_path, name))
        torch.save(model.state_dict(), os.path.join(ckpt_path, name, f'{epoch_idx}.pth'))
    return loss_stage


def main(writer, name, batch_size=144):
    ckpt_path = r'/share/yzhou298'
    save_ckpt = True
    add_displacement = True
    accumulate_angles = False

    # load data
    dataset = ComprehensiveRobotDataset(
        data_dir='./data_position_5000/', 
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
    model = Backbone(128, 2, 3, 192, add_displacement=add_displacement)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    # train n epoches
    loss_stage = 0
    for i in range(5):
        loss_stage = train(writer, name, i, data_loader, model, optimizer, 
            criterion, ckpt_path, save_ckpt, loss_stage)


if __name__ == '__main__':
    # Debussy
    name = 'train11-3-supervised-attn5-bs144'
    writer = SummaryWriter('runs/' + name)

    main(writer, name)
