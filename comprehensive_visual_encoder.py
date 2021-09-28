import numpy as np
# from models.img_encoder import ImgEncoder
from models.comprehensive_visual_encoder import ComprehensiveVisualEncoder
from utils.load_data import ComprehensiveVisualDataset
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
    for idx, (img, joints, end_position, object_list) in enumerate(data_loader):
        # Prepare data
        img = img.to(device)
        joints = joints.to(device)
        end_position = end_position.to(device)
        object_list = object_list.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        img_embed, joints_pred, end_position_pred, object_list_pred = model(img)
        
        # Calculate loss
        loss_joints = criterion(joints_pred, joints)
        loss_end_position = criterion(end_position_pred, end_position)
        loss_object_list = criterion(object_list_pred, object_list)
        loss = loss_joints + loss_end_position + loss_object_list
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Log and print
        writer.add_scalar('train loss', loss, global_step=epoch_idx * len(data_loader) + idx)
        print(f'epoch {epoch_idx}, step {idx}, loss_joints {loss_joints.item():.3f}, loss_end_position {loss_end_position.item():.3f}, loss_object_list {loss_object_list.item():.3f}')
    
    # Save checkpoint
    if save_ckpt:
        if not os.path.isdir(os.path.join(ckpt_path, name)):
            os.mkdir(os.path.join(ckpt_path, name))
        torch.save(model.state_dict(), os.path.join(ckpt_path, name, f'{epoch_idx}.pth'))


# class ComprehensiveVisualEncoder(nn.Module):
#     def __init__(self, img_size, num_joints, embedding_size, device=torch.device('cuda')):
#         super(ComprehensiveVisualEncoder, self).__init__()
#         self.img_encoder = ImgEncoder(img_size, embedding_size)

#         self.joints_predictor = nn.Sequential(
#             nn.Linear(embedding_size, 32), 
#             nn.ReLU(), 
#             nn.Linear(32, num_joints * 2))

#         self.end_position_predictor = nn.Sequential(
#             nn.Linear(embedding_size, 32), 
#             nn.ReLU(), 
#             nn.Linear(32, 2))

#         self.object_detector = nn.Sequential(
#             nn.Linear(embedding_size, 64), 
#             nn.ReLU(), 
#             nn.Linear(64, 6))

#     def forward(self, img):
#         img_embedding = self.img_encoder(img)
#         joints_pred = self.joints_predictor(img_embedding)
#         end_position_pred = self.end_position_predictor(img_embedding)
#         object_list_pred = self.object_detector(img_embedding)
#         return joints_pred, end_position_pred, object_list_pred


def main(writer, name, batch_size=4):
    ckpt_path = r'/share/yzhou298'
    save_ckpt = True
    # load data
    dataset = ComprehensiveVisualDataset(
        data_dir='./data_position/', use_trigonometric_representation=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
    # load model
    model = ComprehensiveVisualEncoder(img_size=128, num_joints=2, embedding_size=128)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    # train n epoches
    for i in range(1):
        train(writer, name, i, data_loader, model, optimizer, criterion, ckpt_path, save_ckpt)
    # test
    # save checkpoint


if __name__ == '__main__':
    name = 'train6-3-comprehensive-visual-encoder-mse'
    writer = SummaryWriter('runs/' + name)

    main(writer, name)