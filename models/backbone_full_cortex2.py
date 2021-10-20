import torch
import torch.nn as nn
from models.task_id_encoder import TaskIDEncoder
from models.cross_attention_bert import CrossAttentionBERT
import numpy as np
import math
import torch.nn.functional as F


# courtesy: https://github.com/darkstar112358/fast-neural-style/blob/master/neural_style/transformer_net.py
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ImgEncoder(nn.Module):
    def __init__(self, img_size, ngf=64, channel_multiplier=4, input_nc=3):
        super(ImgEncoder, self).__init__()
        self.layer1 = nn.Sequential(nn.ReflectionPad2d((3,3,3,3)),
                                    nn.Conv2d(input_nc,ngf,kernel_size=7,stride=1),
                                    nn.InstanceNorm2d(ngf),
                                    nn.ReLU(True))
        
        self.layer2 = nn.Sequential(nn.Conv2d(ngf,ngf*channel_multiplier//2,kernel_size=3,stride=2,padding=1),
                                   nn.InstanceNorm2d(ngf*channel_multiplier//2),
                                   nn.ReLU(True))
        
        self.layer3 = nn.Sequential(nn.Conv2d(ngf*channel_multiplier // 2,ngf*channel_multiplier,kernel_size=3,stride=2,padding=1),
                                   nn.InstanceNorm2d(ngf*channel_multiplier),
                                   nn.ReLU(True))

        self.layer4 = nn.Sequential(nn.Conv2d(ngf*channel_multiplier,ngf*channel_multiplier,kernel_size=3,stride=2,padding=1),
                                   nn.InstanceNorm2d(ngf*channel_multiplier),
                                   nn.ReLU(True))

        self.layer5 = nn.Sequential(ResidualBlock(ngf*channel_multiplier,ngf*channel_multiplier),
                                    ResidualBlock(ngf*channel_multiplier,ngf*channel_multiplier),
                                    ResidualBlock(ngf*channel_multiplier,ngf*channel_multiplier))
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out


class JointEncoder(nn.Module):
    def __init__(self, num_joints, embedding_size):
        super(JointEncoder, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(num_joints, 2048)
        self.layer2 = nn.Linear(2048, 512)
        self.layer3 = nn.Linear(512, 256)
        self.layer4 = nn.Linear(256, embedding_size)

    def forward(self, joints):
        x = self.flatten(joints)
        x = F.selu(self.layer1(x))
        x = F.selu(self.layer2(x))
        x = F.selu(self.layer3(x))
        x = F.selu(self.layer4(x))
        return x


class Controller(nn.Module):
    def __init__(self, embedding_size, num_joints):
        super(Controller, self).__init__()
        self.layer1 = nn.Linear(embedding_size, 16 * 16)
        self.layer2 = nn.Linear(16 * 16, 16 * 16)
        self.layer3 = nn.Linear(16 * 16, 16 * 16)
        self.layer4 = nn.Linear(16 * 16, num_joints)

    def forward(self, x):
        x = F.selu(self.layer1(x))
        x = F.selu(self.layer2(x))
        x = F.selu(self.layer3(x))
        x = self.layer4(x)
        return x


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

        # self.v_proj_2 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.ReLU())

        # self.o_proj_2 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.ReLU())

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
        # v = self.v_proj_2(v)

        q = q.reshape(batch_size, seq_length_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_length_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_length_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Determine value outputs
        values, attention = self._scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length_q, embed_dim)
        o = self.o_proj(values)
        # o = self.o_proj_2(o)

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

        # Visual Pathway
        self.visual_encoder = ImgEncoder(img_size, ngf=ngf, channel_multiplier=channel_multiplier)
        self.visual_encoder_narrower = nn.Sequential(
            nn.Linear(ngf * channel_multiplier, embedding_size),
            nn.ReLU())
        self.img_position_embed = nn.Embedding((img_size // 4) ** 2, embedding_size)
        self.img_embed_merge_pos_embed = nn.Sequential(
            nn.Linear(embedding_size + 2, embedding_size),
            nn.ReLU())
        self.img_embed_to_qkv = []
        for i in range(3):
            self.img_embed_to_qkv.append(nn.Sequential(
                nn.Linear(embedding_size, embedding_size),
                nn.LayerNorm(embedding_size),
                nn.ReLU()
            ))
        self.img_embed_to_qkv = nn.ModuleList(self.img_embed_to_qkv)
        
        # Joints Pathway
        self.joint_encoder = JointEncoder(num_joints * 2, embedding_size)
        self.joint_embed_to_qkv = []
        for i in range(3):
            self.joint_embed_to_qkv.append(nn.Sequential(
                nn.Linear(embedding_size, embedding_size),
                nn.LayerNorm(embedding_size),
                nn.ReLU()
            ))
        self.joint_embed_to_qkv = nn.ModuleList(self.joint_embed_to_qkv)

        # Task Pathway
        self.task_id_encoder = TaskIDEncoder(num_tasks, embedding_size)
        self.task_embed_to_qkv = []
        for i in range(3):
            self.task_embed_to_qkv.append(nn.Sequential(
                nn.Linear(embedding_size, embedding_size),
                nn.LayerNorm(embedding_size),
                nn.ReLU()
            ))
        self.task_embed_to_qkv = nn.ModuleList(self.task_embed_to_qkv)

        # Requests and their mappings to qkv
        self.task_request = nn.Parameter(torch.rand(embedding_size))
        self.action_request = nn.Parameter(torch.rand(embedding_size))
        self.displacement_request = nn.Parameter(torch.rand(embedding_size))
        self.joints_request = nn.Parameter(torch.rand(embedding_size))
        self.requests = [self.task_request, self.action_request, self.displacement_request, self.joints_request]
        self.requests_to_queries = []
        self.requests_to_keys = []
        self.requests_to_values = []
        for i in range(len(self.requests)):
            self.requests_to_queries.append(nn.Sequential(
                nn.Linear(embedding_size, embedding_size),
                nn.LayerNorm(embedding_size),
                nn.SELU()
            ))
            self.requests_to_keys.append(nn.Sequential(
                nn.Linear(embedding_size, embedding_size),
                nn.LayerNorm(embedding_size),
                nn.SELU()
            ))
            self.requests_to_values.append(nn.Sequential(
                nn.Linear(embedding_size, embedding_size),
                nn.LayerNorm(embedding_size),
                nn.SELU()
            ))
        self.requests_to_queries = nn.ModuleList(self.requests_to_queries)
        self.requests_to_keys = nn.ModuleList(self.requests_to_keys)
        self.requests_to_values = nn.ModuleList(self.requests_to_values)

        # Cortex Module
        self.seg_embed = nn.Embedding(7, embedding_size)
        self.attn = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=8, device=device, batch_first=True)
        self.attn2 = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=8, device=device, batch_first=True)
        self.status_embed_to_qkv = []
        for i in range(3):
            self.status_embed_to_qkv.append(nn.Sequential(
                nn.Linear(embedding_size, embedding_size),
                nn.LayerNorm(embedding_size),
                nn.ReLU()
            ))
        self.status_embed_to_qkv = nn.ModuleList(self.status_embed_to_qkv)

        # Post Attention: action prediction
        self.controller = Controller(embedding_size, num_joints * 2)
        self.embed_to_target_position = nn.Sequential(
            nn.Linear(embedding_size, 128), 
            nn.SELU(), 
            nn.Linear(128, 2))

        # Post Attention: displacement prediction
        if add_displacement:
            self.embed_to_displacement = nn.Sequential(
            nn.Linear(embedding_size, 64), 
            nn.SELU(), 
            nn.Linear(64, 2))

        # Post Attention: Joints prediction
        self.joints_predictor = nn.Sequential(
            nn.Linear(embedding_size, 64), 
            nn.SELU(), 
            nn.Linear(64, 64), 
            nn.SELU(), 
            nn.Linear(64, num_joints * 2))

    def _embed_to_qkv_(self, embed, qkv_list):
        q = qkv_list[0](embed)
        k = qkv_list[1](embed)
        v = qkv_list[2](embed)
        return q, k, v

    def _img_pathway_(self, img):
        # Comprehensive Visual Encoder. img_embedding is the square token list
        img_embedding = self.visual_encoder(img)

        # Merge H and W dimensions
        _, _, H, W = img_embedding.shape
        img_embedding = img_embedding.reshape(img_embedding.shape[0], img_embedding.shape[1], -1).permute(0, 2, 1)

        # Narrow the embedding size
        img_embedding = self.visual_encoder_narrower(img_embedding)

        # Prepare the pos embedding for attention
        batch_size, H_W, _ = img_embedding.shape
        pos_embed_w = torch.tensor(np.arange(-W//2, W-W//2), dtype=torch.float32).unsqueeze(0).unsqueeze(1).repeat(batch_size, H, 1).reshape(batch_size, -1).unsqueeze(2).repeat(1, 1, 1).to(self.device)
        pos_embed_h = torch.tensor(np.arange(H-1 - H//2, -1 - H//2, -1), dtype=torch.float32).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, W).reshape(batch_size, -1).unsqueeze(2).repeat(1, 1, 1).to(self.device)
        
        # Concatenate pos embedding with the img embedding
        img_embedding = torch.cat((img_embedding, pos_embed_w, pos_embed_h), dim=2)
        img_embedding = self.img_embed_merge_pos_embed(img_embedding)
        img_embed_query, img_embed_key, img_embed_value = self._embed_to_qkv_(img_embedding, self.img_embed_to_qkv)

        return img_embed_query, img_embed_key, img_embed_value

    def _joints_pathway_(self, joints):
        joints_embedding = self.joint_encoder(joints).unsqueeze(1)
        joints_embed_query, joints_embed_key, joints_embed_value = self._embed_to_qkv_(joints_embedding, self.joint_embed_to_qkv)
        return joints_embed_query, joints_embed_key, joints_embed_value

    def _task_id_pathway_(self, target_id):
        task_embedding = self.task_id_encoder(target_id).unsqueeze(1)
        task_embed_query, task_embed_key, task_embed_value = self._embed_to_qkv_(task_embedding, self.task_embed_to_qkv)
        return task_embed_query, task_embed_key, task_embed_value

    def _requests_pathway_(self, batch_size):
        requests_qs = []
        requests_ks = []
        requests_vs = []
        for i in range(len(self.requests)):
            requests_qs.append(self.requests_to_queries[i](self.requests[i]).unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1))
            requests_ks.append(self.requests_to_keys[i](self.requests[i]).unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1))
            requests_vs.append(self.requests_to_values[i](self.requests[i]).unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1))
        requests_qs = torch.cat(requests_qs, dim=1)
        requests_ks = torch.cat(requests_ks, dim=1)
        requests_vs = torch.cat(requests_vs, dim=1)
        return requests_qs, requests_ks, requests_vs

    def _status_embed_to_qkv_(self, status_embed):
        last_subjective_part_query, last_subjective_part_key, last_subjective_part_value = self._embed_to_qkv_(status_embed, self.status_embed_to_qkv)
        return last_subjective_part_query, last_subjective_part_key, last_subjective_part_value

    def _update_cortex_status_(self, last_state_embed, perception_query, perception_key, perception_value):
        last_subjective_part = last_state_embed[:, :4, :]
        last_subjective_part_query, last_subjective_part_key, last_subjective_part_value = self._status_embed_to_qkv_(last_subjective_part)
        # last_subjective_part_query, last_subjective_part_key, last_subjective_part_value = last_subjective_part, last_subjective_part, last_subjective_part
        cortex_query = torch.cat((last_subjective_part_query, perception_query), dim=1)
        cortex_key = torch.cat((last_subjective_part_key, perception_value), dim=1)
        cortex_value = torch.cat((last_subjective_part_value, perception_value), dim=1)
        return cortex_query, cortex_key, cortex_value

    def _get_segment_embed_(self):
        segment_ids = [0, 1, 2, 3] + 256 * [4] + [5, 6]

    def forward(self, img, joints, target_id, attn_mask=None):

        # Image Pathway
        img_embed_query, img_embed_key, img_embed_value = self._img_pathway_(img)

        # Joints Pathway
        joints_embed_query, joints_embed_key, joints_embed_value = self._joints_pathway_(joints)

        # Task ID Pathway
        task_embed_query, task_embed_key, task_embed_value = self._task_id_pathway_(target_id)

        # Concatenate All Perception Pathways
        perception_query = torch.cat((img_embed_query, joints_embed_query, task_embed_query), dim=1)
        perception_key = torch.cat((img_embed_key, joints_embed_key, task_embed_key), dim=1)
        perception_value = torch.cat((img_embed_value, joints_embed_value, task_embed_value), dim=1)

        # Requests
        requests_qs, requests_ks, requests_vs = self._requests_pathway_(batch_size=img.shape[0])

        # Concatenate All Subjective and Objective Parts
        cortex_query = torch.cat((requests_qs, perception_query), dim=1)
        cortex_key = torch.cat((requests_ks, perception_key), dim=1)
        cortex_value = torch.cat((requests_vs, perception_value), dim=1)

        # Attention itself
        state_embedding, attn_map = self.attn(cortex_query, cortex_key, cortex_value, need_weights=True, attn_mask=attn_mask)
        cortex_query2, cortex_key2, cortex_value2 = self._update_cortex_status_(state_embedding, perception_query, perception_key, perception_value)
        state_embedding2, attn_map2 = self.attn2(cortex_query2, cortex_key2, cortex_value2, need_weights=True, attn_mask=attn_mask)
        # state_embedding2, attn_map2 = self.attn2(cortex_query, cortex_key, cortex_value, need_weights=True, attn_mask=attn_mask)

        # Post-attn operations. Predict the results from the state embedding
        target_position_pred = self.embed_to_target_position(state_embedding2[:,0,:])
        joints_pred = self.joints_predictor(state_embedding2[:,3,:])
        action_pred = self.controller(state_embedding2[:,1,:])

        if self.add_displacement:
            displacement_pred = self.embed_to_displacement(state_embedding2[:, 2, :])
            return action_pred, target_position_pred, displacement_pred, attn_map, attn_map2, joints_pred
        return action_pred, target_position_pred, attn_map, joints_pred
