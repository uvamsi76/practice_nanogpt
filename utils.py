from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from transformers import GPT2LMHeadModel
import tiktoken 

def generate(model,num_return_sequences,device,max_length,input_text="Hello I'm a language model,"):
    enc=tiktoken.get_encoding('gpt2')
    tokens=enc.encode(input_text)
    tokens=torch.tensor(tokens,dtype=torch.long)
    tokens=tokens.unsqueeze(0).repeat(num_return_sequences,1)
    x=tokens.to(device)

    while x.size(1) < max_length:
        with torch.no_grad():
            logits,loss=model(x)
            logits=logits[:,-1,:]
            probs=F.softmax(logits,dim=-1)
            topk_probs,topk_ind=torch.topk(probs,50,dim=-1)
            ix=torch.multinomial(topk_probs,1)
            xcol=torch.gather(topk_ind,-1,ix)
            x=torch.cat((x,xcol),dim=1)
        
    for i in range(num_return_sequences):
        tokens=x[i,:max_length].tolist()
        decoded=enc.decode(tokens)
        print('>' , decoded)

class DataLoaderLite:
    def __init__(self,B,T):
        self.B=B
        self.T=T

        with open('input.txt','r') as f:
            text=f.read()
        enc=tiktoken.get_encoding('gpt2')
        tokens=enc.encode(text)
        self.tokens=torch.tensor(tokens)

        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens)//(B*T)} Batches")

        self.current_position=0

    def next_batch(self):
        B,T = self.B,self.T
        buf=self.tokens[self.current_position:self.current_position+B*T+1]
        x=buf[:-1].view(B,T)
        y=buf[1:].view(B,T)
        self.current_position+= B*T
        if(self.current_position + B*T >len(self.tokens)):
            self.current_position=0
        return x,y