from models.comprehensive_visual_encoder import ComprehensiveVisualEncoder
from models.task_id_encoder import TaskIDEncoder
import torch
import torch.nn as nn
import transformers


class CrossAttention(nn.Module):
    def __init__(self, embedding_size, device=torch.device('cuda')):
        super(CrossAttention, self).__init__()
        config = transformers.BertConfig(
            hidden_size = embedding_size,
            intermediate_size = embedding_size * 4,
            num_attention_heads = 16,
            type_vocab_size = 3,
            return_dict = False,
            num_hidden_layers = 6,
        )
        self.cross_attention_bert = transformers.BertModel(config)
        self.device = device

    def forward(self, img_embedding, task_embedding):
        inputs_embeds = torch.stack([img_embedding, task_embedding], dim=1)
        output_embeds = self.cross_attention_bert(
            token_type_ids = torch.tensor([0, 1]).to(self.device),
            position_ids = torch.tensor([0, 1]).to(self.device),
            inputs_embeds = inputs_embeds
        )

        return output_embeds


class TargetPositionPredictor(nn.Module):
	def __init__(self, img_size, num_joints, embedding_size, num_tasks, device=torch.device('cuda')):
		super(TargetPositionPredictor, self).__init__()
		self.visual_encoder = ComprehensiveVisualEncoder(img_size, num_joints, embedding_size)
		self.task_id_encoder = TaskIDEncoder(num_tasks, embedding_size)
		self.fusion = nn.Sequential(
			nn.Linear(embedding_size * 2, embedding_size),
			nn.SELU(),
			nn.Linear(embedding_size, embedding_size),
			nn.SELU())
		self.embed_to_target_position = nn.Sequential(
            nn.Linear(embedding_size, 64), 
            nn.SELU(), 
            nn.Linear(64, 2))

	def forward(self, img, task_id):
		img_embedding, joints_pred, end_position_pred, object_list_pred = self.visual_encoder(img)
		task_embedding = self.task_id_encoder(task_id)

		cross_attn_embedding = self.fusion(torch.cat((img_embedding, task_embedding), dim=1))

		target_position_pred = self.embed_to_target_position(cross_attn_embedding)

		return cross_attn_embedding, target_position_pred, joints_pred, end_position_pred, object_list_pred
