import torch
import torch.nn as nn
from models.img_encoder import ImgEncoder
from models.joint_encoder import JointEncoder
from models.task_id_encoder import TaskIDEncoder
from models.cross_attention_bert import CrossAttentionBERT
from models.controller import Controller


class Backbone(nn.Module):
	def __init__(self, img_size, num_joints, num_tasks, embedding_size, device=torch.device('cuda')):
		super(Backbone, self).__init__()
		self.img_encoder = ImgEncoder(img_size, embedding_size)
		self.joint_encoder = JointEncoder(num_joints * 2, embedding_size)
		self.task_id_encoder = TaskIDEncoder(num_tasks, embedding_size)
		self.cross_attention_bert = CrossAttentionBERT(embedding_size, device)
		self.controller = Controller(embedding_size, num_joints * 2)
		self.joints_predictor = nn.Sequential(nn.Linear(embedding_size, 32), nn.ReLU(), nn.Linear(32, num_joints * 2))

	def forward(self, img, joints, target_id):
		img_embedding = self.img_encoder(img)
		joints_pred = self.joints_predictor(img_embedding)
		joint_embedding = self.joint_encoder(joints)
		task_embedding = self.task_id_encoder(target_id)
		_, state_embedding = self.cross_attention_bert(img_embedding, joint_embedding, task_embedding)
		action = self.controller(state_embedding)
		return action, joints_pred

# import torch
# import torch.nn as nn
# from models.img_encoder import ImgEncoder
# from models.joint_encoder import JointEncoder
# from models.task_id_encoder import TaskIDEncoder
# from models.cross_attention_bert import CrossAttentionBERT
# from models.controller import Controller


# class Backbone(nn.Module):
# 	def __init__(self, img_size, num_joints, num_tasks, embedding_size):
# 		super(Backbone, self).__init__()
# 		self.img_encoder = ImgEncoder(img_size, embedding_size)
# 		self.joint_encoder = JointEncoder(num_joints, embedding_size)
# 		self.task_id_encoder = TaskIDEncoder(num_tasks, embedding_size)
# 		self.cross_attention_bert = CrossAttentionBERT(embedding_size)
# 		self.controller = Controller(embedding_size, 2)

# 	def forward(self, img, joints, target_id):
# 		img_embedding = self.img_encoder(img)
# 		joint_embedding = self.joint_encoder(joints)
# 		task_embedding = self.task_id_encoder(target_id)
# 		_, state_embedding = self.cross_attention_bert(img_embedding, joint_embedding, task_embedding)
# 		action = self.controller(state_embedding)
# 		return action