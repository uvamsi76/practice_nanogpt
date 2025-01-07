from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from transformers import GPT2LMHeadModel
import tiktoken 
import time

from model import GPTconfig,GPT
from utils import generate,DataLoaderLite

device='cpu'
if(torch.cuda.is_available()):
    device='cuda'


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
model=torch.compile(model)

# train_loader=DataLoaderLite(B=16, T=1024)
train_loader=DataLoaderLite(B=2, T=512)
# logits,loss=model(x,y)
# print(loss)
optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4, betas=(0.9,0.95),eps=1e-8)

for i in range(50):
    t0=time.time()

    x,y=train_loader.next_batch()
    x=x.to(device)
    y=y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device,dtype=torch.bfloat16):
        logits,loss=model(x,y)
    loss.backward()

    norm=torch.nn.utils.clip_grad_norm_(model.parameters() , 1.0 )

    optimizer.step()
    
    torch.cuda.synchronize()
    
    t1=time.time()
    dt = (t1-t0) * 1000
    tokens_per_sec=(train_loader.B*train_loader.B)/(t1-t0)
    print(f"step: {i} ------> loss: {loss.item()}----------->Norm: {norm:.4f} ------------> dt: {dt:.2f}ms--------> tokens/sec:{tokens_per_sec:.2f}")



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
