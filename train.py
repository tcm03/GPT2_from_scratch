import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Callable
from gpt_dataset import *
from utils import *
from model import *
from inference import *

torch.manual_seed(123)


with open('the-verdict.txt', 'r', encoding = 'utf-8') as f:
    text_data = f.read()

tokenizer = tiktoken.get_encoding('gpt2')
text_ids = tokenizer.encode(text_data)

GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 256,  # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "mha_drop_rate": 0.1,    # Dropout rate for multihead attention block
    "trf_drop_rate": 0.1,    # Dropout rate for each transformer block (after MHA and FFN)
    "gpt_drop_rate": 0.1,    # Dropout rate after the embedding layers
    "qkv_bias": False        # Query-Key-Value bias
}



EOT_TOKEN = text_to_token_ids("<|endoftext|>", tiktoken.get_encoding('gpt2'))[0, 0]

train_ratio = 0.9
split_ix = int(train_ratio * len(text_data))
train_data = text_data[:split_ix]
val_data = text_data[split_ix:]

batch_size = 2
max_length = GPT_CONFIG_124M['context_length']
stride = GPT_CONFIG_124M['context_length']
train_loader = create_dataloader_v1(train_data, batch_size, max_length, stride)
val_loader = create_dataloader_v1(val_data, batch_size, max_length, stride)

def train_simple(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    num_epochs: int,
    start_context: str,
    tokenizer: tiktoken.core.Encoding,
    eval_step = 10,
):
    global_step = 0
    train_losses = []
    eval_losses = []
    tokens_seen = 0
    for t in range(num_epochs):
        print(f"\nEpoch {t}...")
        total_loss = 0
        model.train()
        for i, (input_batch, target_batch) in enumerate(train_loader):
            global_step += 1
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += int(input_batch.numel())
            if global_step % eval_step == 0:
                train_loss, eval_loss = evaluate(model, train_loader, val_loader, device)
                train_losses.append(train_loss)
                eval_losses.append(eval_loss)
                print(f"Epoch {t}, global step {global_step}, train_loss: {train_loss}, eval_loss: {eval_loss}")
            generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, eval_losses, tokens_seen




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPTModel(GPT_CONFIG_124M).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0004, weight_decay = 0.1)
train_simple(train_loader, val_loader, model, optimizer, device, 10, "Every effort moves you", tokenizer, eval_step=10)


## TEST AND SAVE
generate_and_print_sample(model, tokenizer, device, "Every effort moves you", temperature = 1.4, top_k = 25)
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    },
    "model_optimizer.pth"
)

## TEST MODEL LOADING
model_opt_sd = torch.load("model_optimizer.pth", map_location = device)
load_model = GPTModel(GPT_CONFIG_124M)
load_model.load_state_dict(model_opt_sd["model_state_dict"])
load_model.to(device)
load_optimizer = torch.optim.AdamW(load_model.parameters(), lr = 5e-4, weight_decay = 0.1)
load_optimizer.load_state_dict(model_opt_sd["optimizer_state_dict"])

train_simple(
    train_loader = train_loader,
    val_loader = val_loader,
    model = load_model,
    optimizer = load_optimizer,
    device = device,
    num_epochs = 1,
    start_context = "Every effort moves you",
    tokenizer = tokenizer,
    eval_step = 10
)

generate_and_print_sample(load_model, tokenizer, device, "Every effort moves you", temperature = 1.4, top_k = 25)