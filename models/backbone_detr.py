import torch
import torch.nn as nn
# from models.comprehensive_visual_encoder import ComprehensiveVisualEncoder
from models.img_encoder_square import ImgEncoder
from models.joint_encoder import JointEncoder
from models.task_id_encoder import TaskIDEncoder
from models.cross_attention_bert import CrossAttentionBERT
from models.controller import Controller
import numpy as np
import math
import torch.nn.functional as F


# Based on https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.q_proj = nn.Linear(input_dim, embed_dim)
        self.k_proj = nn.Linear(input_dim, embed_dim)
        self.v_proj = nn.Linear(input_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.q_proj.weight)
        # self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        # self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        # self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        # self.o_proj.bias.data.fill_(0)
    
    def _scaled_dot_product(self, q, k, v, mask=None):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention
    
    def forward(self, q, k, v, mask=None, return_attention=False):
        batch_size, seq_length_q, embed_dim = q.size()
        _, seq_length_kv, _ = k.size()

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = q.reshape(batch_size, seq_length_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_length_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_length_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Determine value outputs
        values, attention = self._scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length_q, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


# Some code from DETR
class Backbone(nn.Module):
    def __init__(self, img_size, num_joints, num_tasks, embedding_size, add_displacement=False, ngf=64, channel_multiplier=4, device=torch.device('cuda')):
        super(Backbone, self).__init__()

        self.add_displacement = add_displacement
        self.device = device

        self.visual_encoder = ImgEncoder(img_size, ngf=ngf, channel_multiplier=channel_multiplier)
        # self.conv = nn.Conv2d(ngf * channel_multiplier, embedding_size, 1)
        self.visual_encoder_narrower = nn.Sequential(
            nn.Linear(ngf * channel_multiplier, embedding_size),
            nn.ReLU())

        self.img_position_embed_w = nn.Embedding(img_size // 4, 8)
        self.img_position_embed_h = nn.Embedding(img_size // 4, 8)
        self.img_embed_merge_pos_embed = nn.Sequential(
            nn.Linear(embedding_size + 2, embedding_size),
            nn.ReLU())

        self.img_embed_prepro = nn.Sequential(
            nn.Linear(embedding_size, embedding_size), 
            nn.SELU())
        self.pos_value_embed_prepro = nn.Sequential(
            nn.Linear(embedding_size, embedding_size), 
            nn.SELU())
        
        self.attn = MultiheadAttention(input_dim=embedding_size, embed_dim=embedding_size, num_heads=16)

        self.joint_encoder = JointEncoder(num_joints * 2, embedding_size)
        self.task_id_encoder = TaskIDEncoder(num_tasks, embedding_size)

        self.controller = Controller(embedding_size, num_joints * 2)
        self.embed_to_target_position = nn.Sequential(
            nn.Linear(embedding_size, 128), 
            nn.SELU(), 
            nn.Linear(128, 64), 
            nn.SELU(), 
            nn.Linear(64, 2))
        if add_displacement:
            self.embed_to_displacement = nn.Sequential(
            nn.Linear(embedding_size, 64), 
            nn.SELU(), 
            nn.Linear(64, 2))

        self.img_embedding_converter = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(64 * 2 * img_size * img_size // 16, embedding_size), 
            nn.ReLU())

        self.joints_predictor = nn.Sequential(
            nn.Linear(embedding_size, 64), 
            nn.SELU(), 
            nn.Linear(64, 32), 
            nn.SELU(), 
            nn.Linear(32, num_joints * 2))

        self.end_position_predictor = nn.Sequential(
            nn.Linear(embedding_size, 64), 
            nn.SELU(), 
            nn.Linear(64, 32), 
            nn.SELU(), 
            nn.Linear(32, 2))

        self.object_detector = nn.Sequential(
            nn.Linear(embedding_size, 128), 
            nn.SELU(), 
            nn.Linear(128, 64), 
            nn.SELU(), 
            nn.Linear(64, 6))

        # self.state_embedding_converter = nn.Linear(embedding_size * 10, embedding_size)

    def forward(self, img, joints, target_id):
        # Comprehensive Visual Encoder. img_embedding is the square token list
        img_embedding = self.visual_encoder(img)
        _, _, H, W = img_embedding.shape
        img_embedding = img_embedding.reshape(img_embedding.shape[0], img_embedding.shape[1], -1).permute(0, 2, 1)

        # joints_pred = self.joints_predictor(img_embedding_converted)
        # end_position_pred = self.end_position_predictor(img_embedding_converted)
        # object_list_pred = self.object_detector(img_embedding_converted)
        # img_embedding = img_embedding.reshape(img_embedding.shape[0], img_embedding.shape[1], -1).permute(0, 2, 1)

        # Prepare the image for attention
        img_embedding = self.visual_encoder_narrower(img_embedding)
        batch_size, H_W, _ = img_embedding.shape
        # pos_embed = torch.tensor(np.arange(H_W), dtype=torch.int32).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos_embed = self.img_position_embed(pos_embed)
        # pos_embed_w = self.img_position_embed_w(torch.tensor(np.arange(W), dtype=torch.int32).unsqueeze(0).unsqueeze(1).repeat(batch_size, H, 1).reshape(batch_size, -1).to(self.device))
        # pos_embed_h = self.img_position_embed_h(torch.tensor(np.arange(H), dtype=torch.int32).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, W).reshape(batch_size, -1).to(self.device))
        pos_embed_w = torch.tensor(np.arange(-W//2, W-W//2), dtype=torch.float32).unsqueeze(0).unsqueeze(1).repeat(batch_size, H, 1).reshape(batch_size, -1).unsqueeze(2).repeat(1, 1, 1).to(self.device)
        pos_embed_h = torch.tensor(np.arange(H-1 - H//2, -1 - H//2, -1), dtype=torch.float32).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, W).reshape(batch_size, -1).unsqueeze(2).repeat(1, 1, 1).to(self.device)
        # pos_embed = torch.tensor(np.arange(H_W), dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1).unsqueeze(2).to(self.device)
        
        img_embedding = torch.cat((img_embedding, pos_embed_w, pos_embed_h), dim=2)
        img_embedding = self.img_embed_merge_pos_embed(img_embedding)


        # # The converted img embedding is for auxiliary tasks
        img_embedding_converted = self.img_embedding_converter(img_embedding)
        joints_pred = self.joints_predictor(img_embedding_converted)
        end_position_pred = self.end_position_predictor(img_embedding_converted)
        object_list_pred = self.object_detector(img_embedding_converted)


        # Prepare task id for attention
        task_embedding = self.task_id_encoder(target_id)
        # print('task_embedding', task_embedding.shape)

        # pos_embed_w2 = torch.tensor(np.arange(-W//2, W-W//2), dtype=torch.float32).unsqueeze(0).unsqueeze(1).repeat(batch_size, H, 1).reshape(batch_size, -1).unsqueeze(2).repeat(1, 1, 64).to(self.device)
        # pos_embed_h2 = torch.tensor(np.arange(H-1 - H//2, -1 - H//2, -1), dtype=torch.float32).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, W).reshape(batch_size, -1).unsqueeze(2).repeat(1, 1, 64).to(self.device)
        # pos_value_embed = torch.cat((pos_embed_w2, pos_embed_h2), dim=2)

        # Attention itself
        img_embedding = self.img_embed_prepro(img_embedding)
        # pos_value_embed = self.pos_value_embed_prepro(pos_value_embed)
        state_embedding, attn_map = self.attn(task_embedding.unsqueeze(1), img_embedding, img_embedding, return_attention=True)

        state_embedding = state_embedding.squeeze(1)

        # joints_pred = self.joints_predictor(state_embedding)
        # end_position_pred = self.end_position_predictor(state_embedding)
        # object_list_pred = self.object_detector(state_embedding)

        target_position_pred = self.embed_to_target_position(state_embedding)
        action_pred = self.controller(state_embedding)
        if self.add_displacement:
            displacement_pred = self.embed_to_displacement(state_embedding)
            return action_pred, joints_pred, end_position_pred, object_list_pred, target_position_pred, displacement_pred, attn_map
        return action_pred, joints_pred, end_position_pred, object_list_pred, target_position_pred, attn_map
