from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from transformers import GPT2LMHeadModel
import tiktoken 
import inspect

@dataclass
class GPTconfig:
    block_size: int=1024
    vocab_size: int=50257
    n_layer:int= 12
    n_head: int= 12
    n_embd: int=768

class CasualSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn=nn.Linear(config.n_embd,3*config.n_embd)
        self.c_proj=nn.Linear(config.n_embd, config.n_embd)
        # self.attn_dropout = nn.Dropout(config.dropout)
        # self.resid_dropout = nn.Dropout(config.dropout)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd
    
    def forward(self,x):
        B, T, C = x.size() 

        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # # att = self.attn_dropout(att)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y= F.scaled_dot_product_attention( q, k, v, is_causal=True )

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        # y = self.resid_dropout(self.c_proj(y))
        return y




class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc=nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu=nn.GELU(approximate='tanh')
        self.c_proj=nn.Linear(4*config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self,x):
        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.ln_1=nn.LayerNorm(config.n_embd)
        self.attn=CasualSelfAttention(config)
        self.ln_2=nn.LayerNorm(config.n_embd)
        self.mlp=MLP(config)

    def forward(self,x):
        x=x + self.attn(self.ln_1(x))
        x=x + self.mlp(self.ln_2(x))
        return x
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config=config
        self.transformer=nn.ModuleDict (dict(
            wpe=nn.Embedding(config.block_size,config.n_embd),
            wte=nn.Embedding(config.vocab_size,config.n_embd),
            # drop = nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        )
        )
        self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)
        
        # weight sharing scheme
        self.transformer.wte.weight=self.lm_head.weight 

        # weight initializatrion
        self.apply(self._initweights)

        n_params=sum([p.numel() for p in self.transformer.parameters()])

        print(f"The number of parameters inside this transformer is {n_params}")

        print("The number of parameters : %.2fM"% (n_params/1e6))

    def _initweights(self,module):
        if(isinstance(module,nn.Linear)):
            std=0.02
            if(hasattr(module,'NANOGPT_SCALE_INIT')):
                std*=(2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif(isinstance(module,nn.Embedding)):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)

    def forward(self,idx,targets=None):
        b,T=idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos=torch.arange(0,T,dtype=torch.long, device=idx.device).unsqueeze(0)
        pos_emb=self.transformer.wpe(pos) # pos (1,t) -----> (B,t)
        tok_emb=self.transformer.wte(idx) #(B,t,C)
        x = tok_emb + pos_emb
        # x=self.transformer.drop(pos_emb+tok_emb)
        for block in self.transformer.h:
            x = block(x)
        x=self.transformer.ln_f(x)
        logits=self.lm_head(x)
        
        loss=None
        if(targets is not None):
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))        
        return logits,loss
    
    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        # create a from-scratch initialized minGPT model
        config = GPTconfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()

        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('attn.masked_bias')] # ignore these
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys)

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimisers(self,weight_decay,learning_rate,device):

        param_dict={pn: p for pn,p in self.named_parameters()}
        
        param_dict={pn: p for pn,p in  param_dict.items() if p.requires_grad }

        decay_params=[p for n,p in param_dict.items() if p.dim()>= 2 ]

        non_decay_params=[p for n,p in param_dict.items() if p.dim() < 2 ]
        optim_groups=[
                {'params':decay_params,'weight_decay':weight_decay},
                {'params':non_decay_params,weight_decay:0.0}
        ]
        num_decay_params=sum(p.numel() for p in decay_params )
        non_num_decay_params=sum(p.numel() for p in non_decay_params )
        print(f"num decayed parameter tensors= {num_decay_params}")
        print(f"num non-decayed parameter tensors= {non_num_decay_params}")

        fused_available= 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused=fused_available and 'cuda' in device
        print(f"using fused adamW: {use_fused}")

        optimizer=torch.optim.AdamW(optim_groups,lr=learning_rate, betas=(0.9,0.95),eps= 1e-8,fused=use_fused)
        
        return optimizer