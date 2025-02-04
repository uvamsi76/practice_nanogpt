## About this repo

This is the repo which is Implementation of gpt 124M parm model using pytorch. I used this to learn more on the training process of an LLM by Implementing gpt-2 and training it from scratch which was part of karpathy's zero to hero series on youtube. The video I followed was named ( [Let's reproduce GPT-2 (124M)](https://youtu.be/l8pRSuU81PU?si=Z8Mn0UZim_laAJWP) ). For all who are curious on the my learning journey I went , I have added the git commits incrementally based on what i leanrned on each iteration. 

For all who are interested I have added the trained model weights in this Hugging face repository ( [Model](https://huggingface.co/uvamsi76/gpt-2_repro/) )

you can hop into the ipynb and load the model may be in a way similar to `play-wid-model.ipynb`  and generate some responces using generate function from utils 

` generate(model,num_return_sequences= 5,device='cuda',max_length= 40,ddp_rank=0,input_text="How are") `
```  
model is the model that you will load XD

num_return_sequences is to generate how many "sentences" from given input text, 

device is either ` cuda `  If you want to use GPU or keep it to ` cpu ` It works as well

max_length is the maximum length of each sentence, 

input_text is input text 

```

## Model settings, training process and other info

I have trained it for 1 epoch which is (10B tokens of fineweb edu dataset's sample10B ) , 19073 steps. I have followed the exact hyperparameters mentioned in the karpathy's nanogpt which inturn is from gpt-3 paper. I have used 3 A100s of 40GB of vram each and trained the model for almost 6 hours to complete 1 epoch .I have acheived val loss of 4.2 from 10.9 in 1 epoch. I have also used hellswag eval (which this model is performing terribly) which is giving me approx 25% . I have rented GPUs from [jarvis labs](https://jarvislabs.ai/) which costed me around 2700 approx ( 4500 approx if i include the time i spent on learning distributed model training with multiple gpus :/  ). I also have saved the model's state_dict for every 50 steps ( which is too much of data (179GB) i feel). I did this to visualize the transformer's learning process especially the attention layer's process. It will take time for me to visualize those and understand them in depth.

## Summary of Flow of things that I did from the beginning

0. For someone who has watched karpathy's video this part will feel like a rewind.

1. Started off by checking out huggingface's gpt2 model and loaded it with the pretrained weights that are released by OpenAI and performed some text generations and played with model a bit. 

2. Initial Idea was to replicate the same way as huggingface implemented it and train it so that we will be able to load it in the hf model and compare. (But this was not acheived though)

3. Had written GPT module using pytorch's nn.module. Had written this similar to hf implementation of GPT2 by taking config which contains hyperparams and given the same name to model parameters.

4. Implemented the simple training loop of 50 steps and were training on the tiny shakesphere dataset. And also added some code to see how much time it is taking for each step and how many tokens are being processed each step.

5. Did some minute changes to the naive model that has been written by tying weights of token embeddings and the lmhead (final layer of transformer), adding scaling factor to residual pathways .

6. Started off with GPU gymnastics to improve performance :
    
    1. Converted the tensors which will by default use float32/64 to use torch.tf32. TF32 is basically mantissa cutdown float number. We will loose some precision but this will improve speed of computation by a lot as there will be less bits to process.And we can take this as we are calculating some kind of scientific data where mantissa matters alot.(Think of mantissa as the part after . in a float number ex: .5234 in 12.5234 in simple words )
        * To give a perspective this step improved time taken per step from 1100ms -----> 400ms
    2. Converted from tf32 to bfloat16. This is similar but we cut down more mantissa. This will give us some not so precise but close results with a bit more faster compute speed.
        * After this step 1step jumped from 400ms to 340ms   
    3. Added torch.compile. Up until now our python interpreter does all the computation sequentially line by line. This causes many roundtrips from memory to GPU which can be optimised as we know what will be the next processes to perform,. The same thing is acheived using torch.compile(model). This will compile the whole model to run GPU efficiently with minimal round trips
        * 1 step jumped from 340ms ---> 150ms

    4. Replaced masked attention that we implemented with Flash Attention instead. Flash Attention was the algorithmic improved implementation of attention mechanism as It was proposed keeping parallel computing and GPUs in mind.
        * 1 step jumped from 150ms ---> 107ms
    5. Did some number Gymnastics by converting the numbers we can to closest 2 powers as this takes less time to process inside gpus.This is the reason why now our model is different from Hugging face. We changed the vocab_size as well from 50257 to 524288 (2**19).
        * 1 step jumped from 107ms ---> 100ms
7. After some number gymnastics we added some hyper params according to GPT-3 paper like adding beta and eps to adamw optimiser. We also introduced gradient clipping as this will normalise and prevent training from any extreme outliers in our data. 

8. Added cosine learning rate decay to match with GPT-3 implementation. And also implemented fused adamw optimisation which is GPU friendly method of performing AdamW.

9. Also Added L2 regularization in the form of weight decay. And also changed the dataset from tiny shakesphere to fineweb-edu sample 10BT, which has 10B tokens approximately. We have downloaded it, divided into 100 shards and loaded into the training process. (Long story short)

10. GPT-3 Implemented 0.5M tokens per step of training. we cannot do that with these minute GPU's all at once, so we implemented gradiant accumulation instead. Long story short Instead of performing optimization for 0.5M batches parallelly we do some mini_batch parallelly say 32 and we iterate some 'x' amount of times so that all the 0.5M tokens gets processed in a loop. In this process all the gradients will be accumulated. we perform normalization and perform optimization step now on accumulated gradient. 

11. Now Comes the parallellization process. We did everything that we did up until now in a single GPU. But lets say you want to parallelly train on multiple GPUs at once. This can be acheived by using pytorch's DDP (Distributed Data Parallel). 
    1. But where will we use multiple GPUs is the valid question to have. 
    2. Basically we are doing gradient accumulation any way. What if we do gradient accumulation in multiple GPUs and do optimization by combining the gradients from all gpus. 
    3. This is what essentially we are doing. we perform forward pass and backward pass in the individual gpus inside the grad accum loop. Once we are about to get out of the loop we are synchronising gradients along all the GPUs and performing optimization. 
    4. DDP allows us to all of this. But instead of us running using python we use torchrun

12. Have used Hellaswag eval to evaluate our language modelling capability of our model.

13. After Checking if everything is in place, wanted to start the legit pre-training of the mdoel. I wanted to do that using 8 H100 GPUs that were available in jarvislabs. But at the end performed traing in 3 A100s (As all other A100s are busy :/ )

14. I could not perform in H100 even though they were free because for some reason jarvis labs has no feature to mount additional persistant storage to H100s (I assume they are running H100s seperately from the cluster as they are providing VM service with H100). And I wanted to store the statedicts in a persistand storage (which is of size 180GB approx ). and also wanted to download and process fine-web dataset once and store shards (20GB) in a persistant way.

## How did I use GPUs ? :

1. I checked out some instances from jarvislabs which lets us ssh into the machine. 
2. I sshed into machines , cloned the repo there , played with the model and learned new things and logged off and deleted the instance i created. 
3. I have also checked out multiple GPU machines to see their performance.
4. The powerful machine jarvislabs have is H100 
    * To give a perspective The optimal time to run a single step for A100 was 100ms .  H100 completed same process with the same setting within 50ms.
    * It is almost 2x faster. If I have trained the model using 3 H100s instead of 3 A100s it would have completed training in 3 hrs instead of taking 6hrs per epoch. Of cource the price is also high for H100s. 
    * If you dont mind the longer time to train I feel the cost will be almost same for training with a H100 and training with a A100. with an extra benifit of being able to attch persistant storage to A100s.
    * I have also tested some other GPUs like A6000. to give a perspective The setting i trained with right now took 3s per Gpu for A100 and 7s per Gpu for A6000.
5. To summarize the way I did things was 
    1. To checkout not so powerful GPU but many cpus instance to download dataset and saved shards in the persistant storage mount that I attached to instance
    2. Used The same persistant storage and attched it to Test multiple GPUs and Finally train on a GPU I have Fixed (3 x A100)
    3. Saved the statedict after every 50 steps and stored them in persistand storage. and finally once training was over stored the trained model weights in the different dir of the same persisted storage. Incase anyone wants to play around with the model and the weights here you go [Weights](https://huggingface.co/uvamsi76/gpt-2_repro/).

## My Thoughts, Resolutions and Plans maybe:

1. I am not super impressed by the model I got and also I wont blame it as it was just trained on one epoch. This is mostly like a learning project for me rather than a production grade product. Even though I wanted to run it for more epochs ( I truly want to ), The cost was too much for me to run it in a single month I feel. I will take this learn more from the saved dict, If I feel like I have to train more or I dont know , I will burn my cash again.

2. Having said that I will not recommend doing this If your only goal is to make a language model, cause there are much better opensource models with better trained weights that you can download and run, and Build RAG arch on top of them. But If you are a crazy guy like me who wants to experiment , visualize the process and wanting to mess architecture and play with it Go for it it is super fun.

3. I am also wondering what all other usecases this can be trained on. I mean It would be cool to see it as a forecasting model or some other thing . I have some ideas that I want to test with transformers and this " LLM " GPT architecture.

4. I am also introduced to the beautiful world of hugging face and I just can feel I am going to use it extensively. It would be super cool If I get a full time job to work on these.
