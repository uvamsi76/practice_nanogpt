
from datasets import load_dataset

custom_storage_path='~/jl_fs/.cache/huggingface'

remote_name='sample-10BT'

fw=load_dataset("HuggingFaceFW/fineweb-edu",name=remote_name, split="train" , streaming=True)


for i in fw:
    print(i)