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