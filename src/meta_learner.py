import math
from typing import Optional

import clip
import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from utils import set_device


class MetaLearner(nn.Module):
    def __init__(self, prefix_length, seq_len, clip_model_type, new_words=False):
        super(MetaLearner, self).__init__()
        self.device = set_device()
        self.prefix_length = prefix_length
        self.seq_len = seq_len

        self.clip_model_type = clip_model_type
        self.clip, _ = clip.load(clip_model_type, device=self.device, jit=False)
        self.mapper_dim = self.clip.visual.output_dim

        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]

        self.mapper_net = AttentionMapper(dim_clip=self.mapper_dim, dim_gpt_embedding=self.gpt_embedding_size,
                                          prefix_length=self.prefix_length)

        # Freeze CLIP weights
        for param in self.clip.parameters():
            param.requires_grad = False

        # Freeze LM weights
        for param in self.gpt.parameters():
            param.requires_grad = False

        self.text_generator = TextSampler(self.gpt, self.gpt_tokenizer, self.seq_len, self.prefix_length)
        self.is_multi_gpu = True if torch.cuda.device_count() > 1 else False
        self.enable_multi_gpu()

        if new_words:
            self.gpt.resize_token_embeddings(len(self.gpt_tokenizer))
            self.reinit_word_matrix()

    def forward(self, image, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None, fast_weights=None,
                labels: Optional[torch.Tensor] = None, get_pred_tokens=True):
        batch_size = 4
        clip_prefix = self.clip.encode_image(image)  # encoding image
        tokens_embed = self.get_gpt_embeddings(tokens)

        proj_clip = self.mapper_net(clip_prefix, fast_weights)
        embedding_cat = torch.cat((proj_clip, tokens_embed), dim=1)

        if labels is not None:
            dummy_token = self.get_dummy_token(batch_size=batch_size, dummy_token_len=self.prefix_length)
            labels = torch.cat((dummy_token, tokens), dim=1)

        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        out_logits = out.logits[:, self.prefix_length-1:-1]

        if get_pred_tokens:
            gen_tokens = self.generate_text(prefix_embed=embedding_cat)
            return out_logits, gen_tokens

        return out_logits

    def get_dummy_token(self, batch_size: int, dummy_token_len: int) -> torch.Tensor:
        return torch.zeros(batch_size, dummy_token_len, dtype=torch.int64, device=self.device)

    def get_gpt_embeddings(self, tokens):
        return self.gpt.transformer.wte(tokens) if not self.is_multi_gpu \
            else self.gpt.module.transformer.wte(tokens)

    def reinit_word_matrix(self):
        params = self.gpt.state_dict()
        embeddings = params['transformer.wte.weight']
        pre_expansion_embeddings = embeddings[:-3, :]
        mu = torch.mean(pre_expansion_embeddings, dim=0)
        n = pre_expansion_embeddings.size()[0]
        sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
        dist = torch.distributions.multivariate_normal.MultivariateNormal(
            mu, covariance_matrix=1e-5 * sigma)
        new_embeddings = torch.stack(tuple((dist.sample() for _ in range(3))), dim=0)
        embeddings[-3:, :] = new_embeddings
        params['transformer.wte.weight'][-3:, :] = new_embeddings
        self.gpt.load_state_dict(params)

    def generate_text(self, prefix_embed=None):
        gen_tokens_list = []
        for i in range(prefix_embed.shape[0]):
            gen_tokens = self.text_generator.generate(embed=prefix_embed[i, :self.prefix_length])
            gen_tokens_list.append(gen_tokens)
        gen_tokens_ = torch.stack(gen_tokens_list, dim=0).squeeze(1)
        return gen_tokens_

    def enable_multi_gpu(self):
        if torch.cuda.device_count() > 1:
            gpus_num = torch.cuda.device_count()
            print("Training on {} GPUs! ".format(gpus_num))
            self.clip.encode_image = nn.DataParallel(self.clip.encode_image)
            self.mapper_net = nn.DataParallel(self.mapper_net)
            self.gpt = nn.DataParallel(self.gpt)

            print("The used GPUs are: {}".format(self.gpt.device_ids))


class AttentionMapper(nn.Module):
    def __init__(self, dim_clip, dim_gpt_embedding, prefix_length):
        super(AttentionMapper, self).__init__()
        self.dim_V = dim_clip
        self.num_heads = 8
        self.prefix_length = prefix_length
        self.config = [
            # ( name of param ), [out_size, in_size],
            ('parameter', [prefix_length, dim_clip]),
            # ('linear', [dim_clip, dim_gpt_embedding])
            ('fc_q_linear', [dim_gpt_embedding, dim_clip]),
            ('fc_k_linear', [dim_gpt_embedding, dim_clip]),
            ('fc_v_linear', [dim_gpt_embedding, dim_clip]),
            ('fc_o_linear', [dim_gpt_embedding, dim_gpt_embedding]),
            ('layer_norm_1', [dim_gpt_embedding]),
            ('layer_norm_2', [dim_gpt_embedding])
        ]

        self.vars = nn.ParameterList()
        for i, (name, param) in enumerate(self.config):
            if 'linear' in name:
                w = nn.Parameter(torch.ones(*param))
                b = nn.Parameter(torch.zeros(param[0]))
                torch.nn.init.xavier_uniform_(w)
                self.vars.append(w)
                self.vars.append(b)
            elif 'parameter' in name:  # the visual prefix
                param_learn = nn.Parameter(torch.randn(*param), requires_grad=True)
                self.vars.append(param_learn)
            elif 'layer_norm' in name:
                layer_norm_w = nn.Parameter(torch.ones(*param))
                layer_norm_b = nn.Parameter(torch.zeros(param[0]))
                self.vars.append(layer_norm_w)
                self.vars.append(layer_norm_b)

    def forward(self, clip_x, fast_weights=None):
        clip_x = clip_x.float().unsqueeze(1)
        batch_size, clip_len = clip_x.shape[:2]

        prefix = fast_weights[0].unsqueeze(0).expand(batch_size, *fast_weights[0].shape)  # I
        x_prefix = torch.cat((prefix, clip_x), dim=1)
        # O = F.linear(x_prefix, weight=fast_weights[1], bias=fast_weights[2])

        Q = F.linear(x_prefix, weight=fast_weights[1], bias=fast_weights[2])
        K = F.linear(x_prefix, weight=fast_weights[3], bias=fast_weights[4])
        V = F.linear(x_prefix, weight=fast_weights[5], bias=fast_weights[6])

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = F.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = F.dropout(O, p=0.5)
        O = F.layer_norm(O, normalized_shape=[O.shape[-1]], weight=fast_weights[9], bias=fast_weights[10]) \
            if 'layer_norm_1' in [c[0] for c in self.config] else O

        O = O + F.leaky_relu(F.linear(O, weight=fast_weights[7], bias=fast_weights[8]))
        O = F.dropout(O, p=0.5)
        O = F.layer_norm(O, normalized_shape=[O.shape[-1]], weight=fast_weights[11],  bias=fast_weights[12]) \
            if 'layer_norm_2' in [c[0] for c in self.config] else O
        O = O[:, :self.prefix_length]

        return O


class TextSampler:
    def __init__(self, gpt, gpt_tokenizer, seq_len, prefix_len):
        self.gpt = gpt
        self.tokenizer = gpt_tokenizer
        self.seq_len = seq_len
        self.prefix_len = prefix_len

    def generate(self,
                 tokens=None,
                 prompt=None,
                 embed=None,
                 entry_count=1,
                 top_p=0.8,
                 temperature=1.0):
        """
        Adapted from:
        https://github.com/rmokady/CLIP_prefix_caption/blob/1ad805a844a62ab2e5480479aa021bccf0d4d12a/predict.py
        """
        entry_length = self.seq_len
        self.gpt.eval()
        filter_value = -float("Inf")
        device = next(self.gpt.parameters()).device

        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed.unsqueeze(0)
            else:
                if tokens is None:
                    tokens = torch.tensor(self.tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)
                generated = self.gpt.transformer.wte(tokens)

            for i in range(entry_length):
                outputs = self.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p  # nucleus sampling
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(1)
                next_token_embed = self.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)

        return tokens


