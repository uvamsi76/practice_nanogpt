from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from transformers import GPT2LMHeadModel
import tiktoken 
import time

from model import GPTconfig,GPT
from utils import generate,DataLoaderLite,get_lr

device='cpu'
if(torch.cuda.is_available()):
    device='cuda'


max_steps=50

print(f"we have the device : {device}")

torch.set_float32_matmul_precision('high')

torch.manual_seed(1337)
if( torch.cuda.is_available()):
    torch.cuda.manual_seed(1337)

num_return_sequences=5
max_length=60
# model=GPT.from_pretrained('gpt2')
config = GPTconfig(vocab_size=50304)
model = GPT(config)
print('didnt crash')
model.eval()
model.to(device)
# model=torch.compile(model)

total_batch_size=524288
# B=64 H100 is handling this well
B=16
T=1024
grad_accum_steps=total_batch_size// (B*T)

print(f"total desired batch size:{total_batch_size}")
print(f"calculated grad accum step:{grad_accum_steps}")

train_loader=DataLoaderLite(B=B, T=T)
# train_loader=DataLoaderLite(B=2, T=512)
# logits,loss=model(x,y)
# print(loss)
# optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4, betas=(0.9,0.95),eps=1e-8)
optimizer = model.configure_optimisers(weight_decay=0.1,learning_rate=6e-4,device=device)

for step in range(max_steps):
    t0=time.time()
    loss_accum=0.0
    for micro_step in range(grad_accum_steps):
        x,y=train_loader.next_batch()
        x=x.to(device)
        y=y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device,dtype=torch.bfloat16):
            logits,loss=model(x,y)
        
        loss=loss / grad_accum_steps
        loss_accum+=loss.detach()

        loss.backward()


    norm=torch.nn.utils.clip_grad_norm_(model.parameters() , 1.0 )

    lr=get_lr(step)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    
    torch.cuda.synchronize()
    
    t1=time.time()
    dt = (t1-t0) * 1000
    tokens_per_sec=(train_loader.B*train_loader.T*grad_accum_steps)/(t1-t0)
    print(f"step: {step} ------> loss: {loss_accum.item():.6f}----------->Norm: {norm:.4f} -----> LR: {lr:.4e}   -------> dt: {dt:.2f}ms--------> tokens/sec:{tokens_per_sec:.2f}")



import sys; sys.exit(0)


#--------------------------------------------------------#-----------------------------------------------#

# sampling from the logits to predict next token basically generating logic 
generate(model,num_return_sequences,device,max_length)


# enc=tiktoken.get_encoding('gpt2')

# with open('input.txt','r') as f:
#     data=f.read()
# text=data[:1000]
# tokens=enc.encode(text)
# B=4
# T=32
# buf=torch.tensor(tokens[:B*T + 1])
# buf=buf.to(device)
# x=buf[:-1].view(B,T)
# y=buf[1:].view(B,T)
