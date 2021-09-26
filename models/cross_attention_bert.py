import transformers
import torch
import torch.nn as nn


# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')


class CrossAttentionBERT(nn.Module):
    def __init__(self, embedding_size, device=torch.device('cuda')):
        super(CrossAttentionBERT, self).__init__()
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

    def forward(self, img_embedding, joint_embedding, task_embedding):
        inputs_embeds = torch.stack([img_embedding, joint_embedding, task_embedding], dim=1)
        output_embeds = self.cross_attention_bert(
            token_type_ids = torch.tensor([0, 1, 2]).to(self.device),
            position_ids = torch.tensor([0, 1, 2]).to(self.device),
            inputs_embeds = inputs_embeds
        )

        return output_embeds


if __name__ == '__main__':
    attn = CrossAttentionBERT(128)
    batch_size = 4
    embed_size = 128
    inputs_embeds = torch.rand(batch_size, embed_size)
    output_embeds = attn(inputs_embeds, inputs_embeds, inputs_embeds)
    print(len(output_embeds))
    print(output_embeds[0].shape, output_embeds[1].shape)