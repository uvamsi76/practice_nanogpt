from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from transformers import GPT2LMHeadModel
import tiktoken 
import numpy as np
import os
import torch.distributed as dist

def compute_val_loss(step,model,val_loader,device,ddp):
    val_loss_accum=0
    if step % 100 ==0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum=0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x,y=x.to(device),y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits,loss=model(x,y)
                loss=loss/val_loss_steps
                val_loss_accum+=loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum,op=dist.ReduceOp.AVG)
    return val_loss_accum


def generate(model,num_return_sequences,device,max_length,ddp_rank,input_text="Hello I'm a language model,"):
    model.eval()
    enc=tiktoken.get_encoding('gpt2')
    tokens=enc.encode(input_text)
    tokens=torch.tensor(tokens,dtype=torch.long)
    tokens=tokens.unsqueeze(0).repeat(num_return_sequences,1)
    xgen=tokens.to(device)
    sample_rng= torch.Generator(device=device)
    sample_rng.manual_seed(42+ddp_rank)

    while xgen.size(1) < max_length:
        with torch.no_grad():
            logits,loss=model(xgen)
            
            logits=logits[:,-1,:]
            
            probs=F.softmax(logits,dim=-1)
            
            topk_probs,topk_ind=torch.topk(probs,50,dim=-1)
            
            ix=torch.multinomial(topk_probs,1,generator=sample_rng)
            
            xcol=torch.gather(topk_ind,-1,ix)
            
            x=torch.cat((x,xcol),dim=1)
        
    for i in range(num_return_sequences):
        tokens=x[i,:max_length].tolist()
        decoded=enc.decode(tokens)
        print(f'------->>> Rank {ddp_rank} Sample {i} : \n |||---->> {decoded}' )


def load_tokens(file_name):
    npt=np.load(file_name)
    npt = npt.astype(np.int32)
    ptt=torch.tensor(npt,dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self,B,T,process_rank,num_processes, split,master_process):
        self.B=B
        self.T=T
        self.process_rank=process_rank
        self.num_processes=num_processes

        assert split in {'train','val'}

        data_root='edu_fineweb10B'
        shards=os.listdir(data_root)
        shards=[s for s in shards if split in s]
        shards=sorted(shards)
        shards= [os.path.join(data_root,s) for s in shards]
        self.shards=shards

        assert len(shards) > 0, f"no shards found for split {split}"

        if master_process:
            print(f"found {len(shards)} for split {split}")
        
        self.reset()

        # self.current_shard = 0

        # self.tokens=load_tokens(self.shards[self.current_shard])
        # # with open('input.txt','r') as f:
        # #     text=f.read()
        # # enc=tiktoken.get_encoding('gpt2')
        # # tokens=enc.encode(text)
        # # self.tokens=torch.tensor(tokens)

        # # print(f"loaded {len(self.tokens)} tokens")
        # # print(f"1 epoch = {len(self.tokens)//(B*T)} Batches")

        # self.current_position = B*T*self.process_rank


    def reset(self):
        self.current_shard=0
        self.tokens=load_tokens(self.shards[self.current_shard])
        self.current_position = self.B* self.T* self.process_rank


    def next_batch(self):
        B,T = self.B,self.T
        buf=self.tokens[self.current_position:self.current_position+B*T+1]
        x=buf[:-1].view(B,T)
        y=buf[1:].view(B,T)
        self.current_position+= B*T*self.num_processes
        if(self.current_position + (B*T*self.num_processes + 1) >len(self.tokens)):
            
            self.current_shard=(self.current_shard+1)%(len(self.shards))
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B*T*self.process_rank
        return x,y

max_lr=6e-4
min_lr=max_lr * 0.1
warmup_steps=715
def get_lr(step,max_steps=50):
    if(step < warmup_steps,):
        return max_lr * (step+1)/ warmup_steps
    if( step > max_steps):
        return min_lr
    decay_ratio=(step - warmup_steps) / (max_steps - warmup_steps)

    coeff=0.5 * (1.0 + math.cos(math.pi*decay_ratio))

    return min_lr + coeff * (max_lr -min_lr)