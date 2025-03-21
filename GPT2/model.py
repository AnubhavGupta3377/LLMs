"""model.py"""

'''
References:

Andrej Karpathy's video: https://www.youtube.com/watch?v=l8pRSuU81PU&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=10&ab_channel=AndrejKarpathy
'''

import math
import torch
from torch.nn import functional as F
from utils import GPTConfig

class MaskedMultiheadSelfAttentionBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0

        # QKV attention matrices concatenated together
        self.c_attn = torch.nn.Linear(config.n_embed, 3 * config.n_embed, bias=True)
        # Output projection matrix
        self.c_proj = torch.nn.Linear(config.n_embed, config.n_embed, bias=True)
        self.n_head = config.n_head
        self.n_embed = config.n_embed

    def forward(self, x):
        device = x.device
        batch_size, seq_len, embed_dim = x.size()

        # Calculate query, key and value vectors for the batches
        # Returns the matrix of size (batch_size, seq_len, 3 * embed_dim)
        qkv = self.c_attn(x)
        # Split query, key, value vectors
        q, k, v = qkv.split(self.n_embed, dim=2)
        q = q.view(batch_size, seq_len, self.n_head, embed_dim//self.n_head).transpose(1, 2) #(batch_size, n_head, seq_len, embed_dim/n_head)
        k = k.view(batch_size, seq_len, self.n_head, embed_dim//self.n_head).transpose(1, 2) #(batch_size, n_head, seq_len, embed_dim/n_head)
        v = v.view(batch_size, seq_len, self.n_head, embed_dim//self.n_head).transpose(1, 2) #(batch_size, n_head, seq_len, embed_dim/n_head)
        
        # # Implementation of attention
        # mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(device)
        # mask = mask.bool().view(1, 1, seq_len, seq_len)
        # attention_scores = (q @ k.transpose(2,3)) / (math.sqrt(q.size(-1))) # (batch_size, n_head, seq_len, seq_len)
        # attention_scores = attention_scores.masked_fill(mask, float('-inf'))
        # attention_scores = F.softmax(attention_scores, dim=-1)
        # output = attention_scores @ v # (batch_size, n_head, seq_len, embed_dim/n_head)

        # Flash attention
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        output = output.transpose(1, 2) # (batch_size, seq_len, n_head, embed_dim/n_head)
        output = output.reshape(batch_size, seq_len, self.n_embed)
        output = self.c_proj(output)
        return output


class FFNBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = torch.nn.Linear(config.n_embed, 4 * config.n_embed, bias=True)
        self.gelu = torch.nn.GELU(approximate='tanh')
        self.c_proj = torch.nn.Linear(4 * config.n_embed, config.n_embed, bias=True)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


'''
Transformer block, consisting of
    - Add & Norm
    - Masked multihead self-attention
    - Add & Norm
    - FFN
'''
class TransformerBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(config.n_embed)
        self.attn = MaskedMultiheadSelfAttentionBlock(config)
        self.ln_2 = torch.nn.LayerNorm(config.n_embed)
        self.mlp = FFNBlock(config)

    def forward(self, x):
        # x => (B, embed_dim)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
        self.transformer = torch.nn.ModuleDict(dict(
            wte = torch.nn.Embedding(config.vocab_size, config.n_embed),
            wpe = torch.nn.Embedding(config.block_size, config.n_embed),
            h = torch.nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f = torch.nn.LayerNorm(config.n_embed)
        ))
        self.lm_head = torch.nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # Weight sharing between token embeddings and final linear layer
        self.lm_head.weight = self.transformer.wte.weight

        # Parameter initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)): # GPT2 has std=0.1 for Embedding modules
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, torch.nn.LayerNorm): # Same as pytorch default
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, targets=None):
        device = idx.device
        batch_size, seq_len = idx.size()
        assert seq_len <= self.config.block_size

        positions = torch.arange(0, seq_len, dtype=torch.long, device=device)
        pos_embed = self.transformer.wpe(positions) # (batch_size, seq_len)
        tok_embed = self.transformer.wte(idx) # (batch_size, seq_len, n_embed)
        x = tok_embed + pos_embed
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (batch_size, seq_len, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """ Loads pretrained GPT-2 model weights from huggingface """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"loading weights from pretrained gpt: {model_type}")

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embed=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embed=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embed=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embed=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model