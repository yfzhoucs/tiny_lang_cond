import torch
import torch.nn as nn
from models.comprehensive_visual_encoder import ComprehensiveVisualEncoder
from models.joint_encoder import JointEncoder
from models.task_id_encoder import TaskIDEncoder
from models.cross_attention_bert import CrossAttentionBERT
from models.controller import Controller


class Backbone(nn.Module):
	def __init__(self, img_size, num_joints, num_tasks, embedding_size, add_displacement=False):
		super(Backbone, self).__init__()

		self.add_displacement = add_displacement

		self.visual_encoder = ComprehensiveVisualEncoder(img_size, num_joints, embedding_size)
		self.joint_encoder = JointEncoder(num_joints * 2, embedding_size)
		self.task_id_encoder = TaskIDEncoder(num_tasks, embedding_size)
		self.fusion = nn.Sequential(
			nn.Linear(embedding_size * 3, embedding_size),
			nn.SELU(),
			nn.Linear(embedding_size, embedding_size),
			nn.SELU())
		self.controller = Controller(embedding_size, num_joints * 2)
		self.embed_to_target_position = nn.Sequential(
            nn.Linear(embedding_size, 64), 
            nn.SELU(), 
            nn.Linear(64, 2))
		if add_displacement:
			self.embed_to_displacement = nn.Sequential(
            nn.Linear(embedding_size, 64), 
            nn.SELU(), 
            nn.Linear(64, 2))

	def forward(self, img, joints, target_id):
		img_embedding, joints_pred, end_position_pred, object_list_pred = self.visual_encoder(img)
		joint_embedding = self.joint_encoder(joints)
		task_embedding = self.task_id_encoder(target_id)
		state_embedding = self.fusion(torch.cat((img_embedding, joint_embedding, task_embedding), dim=1))
		target_position_pred = self.embed_to_target_position(state_embedding)
		action_pred = self.controller(state_embedding)
		if self.add_displacement:
			displacement_pred = self.embed_to_displacement(state_embedding)
			return action_pred, joints_pred, end_position_pred, object_list_pred, target_position_pred, displacement_pred
		return action_pred, joints_pred, end_position_pred, object_list_pred, target_position_pred

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