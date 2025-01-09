from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from transformers import GPT2LMHeadModel
import tiktoken 
import time
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import os 

from model import GPTconfig,GPT
from utils import generate,DataLoaderLite,get_lr


# script to run using ddp : torchrun --standalone --nproc_per_node=2 train.py 


# device='cpu'
# if(torch.cuda.is_available()):
#     device='cuda'

ddp= int(os.environ.get('RANK',-1)) != -1

if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank= int(os.environ['RANK'])
    ddp_local_rank=int(os.environ['LOCAL_RANK'])
    ddp_world_size=int(os.environ['WORLD_SIZE'])
    
    device=f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    
    master_process= ddp_rank==0

else:
    ddp_rank=0
    ddp_local_rank=0
    ddp_world_size=1
    master_process=True
    device='cpu'
    
    if(torch.cuda.is_available()):
        device='cuda'
device_type = "cuda" if device.startswith("cuda") else "cpu"
torch.manual_seed(1337)
if( torch.cuda.is_available()):
    torch.cuda.manual_seed(1337)

num_return_sequences=5
max_length=32

total_batch_size=524288

B=32 #H100 is handling this well up to 64 a100 upto 32
# B=16
T=1024

grad_accum_steps=total_batch_size// (B*T*ddp_world_size)

if(master_process):
    print(f"total desired batch size:{total_batch_size}")

    print(f"calculated grad accum step:{grad_accum_steps}")


print(f"I am GPU {ddp_rank}")

train_loader=DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train', master_process=master_process)  

val_loader=DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val', master_process=master_process)

torch.set_float32_matmul_precision('high') # to enable computations in fp32 i think


# model=GPT.from_pretrained('gpt2')
config = GPTconfig(vocab_size=50304)
model = GPT(config)
model.to(device)
use_compile=False
if(use_compile):
    model=torch.compile(model)
if ddp:
    model=DDP(model,device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model 

max_steps= 19073
optimizer = raw_model.configure_optimisers(weight_decay=0.1,learning_rate=6e-4,device=device)

for step in range(max_steps):

    t0=time.time()

    if step % 100 ==0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum=0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x,y=x.to(device),y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits,loss=model(x,y)
                loss=loss/val_loss_steps
                val_loss_accum+=loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum,op=dist.ReduceOp.AVG)
        
        if master_process:
            print(f"validation loss is : {val_loss_accum.item():.4f}")

    if(step>0 and step%100==0 and False):
        generate(model,num_return_sequences,device,max_length,ddp_rank,input_text="Hello I'm a language model,")

    loss_accum=0.0
    for micro_step in range(grad_accum_steps):
        x,y=train_loader.next_batch()
        x=x.to(device)
        y=y.to(device)
        optimizer.zero_grad()
        if ddp:
            model.require_backward_grad_sync= (micro_step == grad_accum_steps - 1 ) # this we are doing to synchronize the gradients in the last step and in all other steps it will be jus adding up gradients in backward pass i believe

        with torch.autocast(device_type=device_type,dtype=torch.bfloat16): #to convert the tensor value to bfloat16
            logits,loss=model(x,y)
        
        loss=loss / grad_accum_steps
        loss_accum+=loss.detach()
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum,op=dist.ReduceOp.AVG)

    norm=torch.nn.utils.clip_grad_norm_(model.parameters() , 1.0 )  # we are basically doing gradient clipping incase if model learns some outliers i would say

    lr=get_lr(step)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    
    torch.cuda.synchronize()
    
    t1=time.time()

    dt = (t1-t0) 
    
    tokens_per_sec=(train_loader.B*train_loader.T*grad_accum_steps*ddp_world_size)/(dt)
    
    if master_process:
        print(f"step: {step:.5d} --> loss: {loss_accum.item():.6f}-->Norm: {norm:.4f} ----> LR: {lr:.4e} ---> dt: {dt * 1000:.2f}ms ----> tokens/sec:{tokens_per_sec:.2f}")

if ddp:
    destroy_process_group()


#--------------------------------------------------------#-----------------------------------------------#

# sampling from the logits to predict next token basically generating logic 

# generate(model,num_return_sequences,device,max_length)
