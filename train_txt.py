"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import argparse
import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from utils import *
from model import GPTConfig, GPT


# -----------------------------------------------------------------------------
# Configuration via argparse

def get_args():
    parser = argparse.ArgumentParser(description='Train a GPT-2 model on text data')

    # I/O
    parser.add_argument('--out_dir', type=str, default='out', help='output directory')
    parser.add_argument('--eval_interval', type=int, default=2000, help='eval interval')
    parser.add_argument('--log_interval', type=int, default=1, help='log interval')
    parser.add_argument('--eval_iters', type=int, default=200, help='eval iters')
    parser.add_argument('--eval_only', action='store_true', help='exit right after the first eval')
    parser.add_argument('--always_save_checkpoint', action='store_true', help='always save a checkpoint after each eval')
    parser.add_argument('--no-always_save_checkpoint', action='store_false', dest='always_save_checkpoint')
    parser.set_defaults(always_save_checkpoint=True)
    parser.add_argument('--init_from', type=str, default='scratch', help='scratch or resume or gpt2*')

    # Data
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--train_data_path', type=str, help='training data file name')
    parser.add_argument('--val_data_path', type=str, help='validation data file name')
    parser.add_argument('--batch_size', type=int, default=768, help='batch size')
    parser.add_argument('--block_size', type=int, default=256, help='block size')

    parser.add_argument('--data_shuffle', action='store_true', help='shuffle data')
    parser.add_argument('--no-data_shuffle', action='store_false', dest='data_shuffle')
    parser.set_defaults(data_shuffle=True)

    parser.add_argument('--operator', type=str, default='+', help='operator for arithmetic tasks')
    parser.add_argument('--data_format', type=str, default='plain', help='data format')
    parser.add_argument('--simple', action='store_true', help='use simple formatting')
    parser.add_argument('--reverse_c', action='store_true', help='reverse result')
    parser.add_argument('--data_type', type=str, default='text', help='data type')
    parser.add_argument('--vocabulary', type=str, default='custom_input_data', help='vocabulary type')
    parser.add_argument('--tokenizer', type=str, default='char', help='tokenizer type')

    # Model
    parser.add_argument('--n_layer', type=int, default=6, help='number of layers')
    parser.add_argument('--n_head', type=int, default=6, help='number of heads')
    parser.add_argument('--n_embd', type=int, default=384, help='embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--bias', action='store_true', default=False, help='use bias inside LayerNorm and Linear layers')

    # Optimizer
    parser.add_argument('--learning_rate', type=float, default=6e-4, help='max learning rate')
    parser.add_argument('--max_iters', type=int, default=50000, help='total number of training iterations')
    parser.add_argument('--weight_decay', type=float, default=1e-1, help='weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1')
    parser.add_argument('--beta2', type=float, default=0.95, help='beta2')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='gradient clipping')

    # Learning Rate Decay
    parser.add_argument('--decay_lr', action='store_true', help='whether to decay the learning rate')
    parser.add_argument('--no-decay_lr', action='store_false', dest='decay_lr')
    parser.set_defaults(decay_lr=True)

    parser.add_argument('--warmup_iters', type=int, default=2000, help='warmup iterations')
    parser.add_argument('--lr_decay_iters', type=int, default=50000, help='lr decay iterations')
    parser.add_argument('--min_lr', type=float, default=6e-5, help='minimum learning rate')

    # System
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--compile', action='store_true', help='use PyTorch 2.0 to compile the model')
    parser.add_argument('--no-compile', action='store_false', dest='compile')
    parser.set_defaults(compile=True)
    
    parser.add_argument('--gradient_accumulation_steps', type=int, default=40, help='gradient accumulation steps')
    parser.add_argument('--ddp', action='store_true', help='use ddp')

    args = parser.parse_args()
    return args

args = get_args()
globals().update(vars(args))

# Derived variables
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
config = vars(args)
# -----------------------------------------------------------------------------







# various inits, derived attributes, I/O setup
# if not ddp, we are running on a single gpu, and one process
master_process = True
seed_offset = 0
tokens_per_iter =  batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)






# poor man's data loader
data_dir = os.path.join('data', dataset)


# check for data_format
if dataset is None:
    raise ValueError("dataset argument must be provided")
if train_data_path is None:
    raise ValueError("train_data_path argument must be provided")
if val_data_path is None:
    raise ValueError("val_data_path argument must be provided")


if data_type == 'text':
    if ('reverse' in data_format and not reverse_c) or (reverse_c and 'reverse' not in data_format):
        raise ValueError('reverse_c must be True for data_format == "reverse"')
meta_path_specified = False

data_dir = os.path.join('data', dataset)
train_data_path = os.path.join(data_dir, train_data_path)
val_data = os.path.join(data_dir, val_data_path)
train_data_list = get_data_list(train_data_path, operator=operator)
val_data_list = get_data_list(filename=val_data, operator=operator) # get_data_list(val_data, operator='+')
train_data_str = generate_data_str(train_data_list, operator=operator, format=data_format, train=True, shuffle=data_shuffle, add_space=False, simple=simple, random_A=False, random_C=False)
val_data_str = generate_data_str(val_data_list, operator=operator, format=data_format, train=True, shuffle=data_shuffle, add_space=False, simple=simple, random_A=False, random_C=False)
meta, meta_path, data_encoder, data_decoder = create_meta_file(vocabulary=vocabulary, input_data_str=train_data_str, tokenizer=tokenizer)
meta_vocab_size = meta['vocab_size']
train_data = data_encoder(train_data_str)
val_data = data_encoder(val_data_str)


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])   

    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")



# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)




elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0





# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)




if master_process:
    with open(os.path.join(out_dir, 'log.txt'), 'a') as f:
        f.write(f"training run started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")




# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        with open(os.path.join(out_dir, 'log.txt'), 'a') as f:
            f.write(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}\n")
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        with open(os.path.join(out_dir, 'log.txt'), 'a') as f:
            f.write(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%\n")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break