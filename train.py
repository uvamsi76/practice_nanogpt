from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from transformers import GPT2LMHeadModel
import tiktoken 

from model import GPTconfig,GPT
from utils import generate

device='cpu'
# if(torch.cuda.is_available()):
#     device='cuda'

print(f"we have the device : {device}")


num_return_sequences=5
max_length=60
# model=GPT.from_pretrained('gpt2')
config = GPTconfig()
model = GPT(config)
print('didnt crash')
model.eval()
model.to(device)

enc=tiktoken.get_encoding('gpt2')

with open('input.txt','r') as f:
    data=f.read()
text=data[:1000]
tokens=enc.encode(text)
B=4
T=32
buf=torch.tensor(tokens[:B*T + 1])
buf=buf.to(device)
x=buf[:-1].view(B,T)
y=buf[1:].view(B,T)


# logits,loss=model(x,y)
# print(loss)
optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4)

for i in range(50):
    optimizer.zero_grad()
    logits,loss=model(x,y)
    loss.backward()
    optimizer.step()

    print(f"step: {i} ------> loss: {loss.item()}")



import sys; sys.exit(0)


#--------------------------------------------------------#-----------------------------------------------#

# sampling from the logits to predict next token basically generating logic 
generate(model,num_return_sequences,device,max_length)
