from models.img_encoder import ImgEncoder
import torch.nn as nn


class ComprehensiveVisualEncoder(nn.Module):
    def __init__(self, img_size, num_joints, embedding_size):
        super(ComprehensiveVisualEncoder, self).__init__()
        self.img_encoder = ImgEncoder(img_size, embedding_size)

        self.joints_predictor = nn.Sequential(
            nn.Linear(embedding_size, 32), 
            nn.ReLU(), 
            nn.Linear(32, num_joints * 2))

        self.end_position_predictor = nn.Sequential(
            nn.Linear(embedding_size, 32), 
            nn.ReLU(), 
            nn.Linear(32, 2))

        self.object_detector = nn.Sequential(
            nn.Linear(embedding_size, 64), 
            nn.ReLU(), 
            nn.Linear(64, 6))

    def forward(self, img):
        img_embedding = self.img_encoder(img)
        joints_pred = self.joints_predictor(img_embedding)
        end_position_pred = self.end_position_predictor(img_embedding)
        object_list_pred = self.object_detector(img_embedding)
        return img_embedding, joints_pred, end_position_pred, object_list_pred