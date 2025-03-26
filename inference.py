import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Callable
from gpt_dataset import *
from utils import *
from model import *



def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: nn.Module,
    device: str,
) -> torch.Tensor:
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    assert isinstance(logits, torch.Tensor) and logits.ndim == 3, "expected tensor of shape: (batch size, sequence length, vocab_size)"
    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(
    data_loader: DataLoader,
    model: nn.Module,
    device: str,
    num_batches: Optional[int] = None
):
    if len(data_loader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    total_loss = 0
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i == num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches

def evaluate(
    model: torch.nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device
):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        eval_loss = calc_loss_loader(eval_loader, model, device)
    model.train()
    return train_loss, eval_loss

def generate_text_simple(
    model: torch.nn.Module,
    start_ids: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    **kwargs
):
    model.eval()
    with torch.no_grad():
        for i in range(max_new_tokens):
            start_ids = start_ids[..., -context_size:]
            logits = model(start_ids)
            last_logits = logits[..., -1, :]
            last_tokens = last_logits.argmax(dim = -1, keepdims = True)
            start_ids = torch.cat([start_ids, last_tokens], dim = -1)
    return start_ids

def generate_text(
    model: torch.nn.Module,
    start_ids: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    **kwargs
) -> torch.Tensor:
    temperature = kwargs.get("temperature", 1.0)
    top_k = kwargs.get("top_k", None)
    model.eval()
    with torch.no_grad():
        for i in range(max_new_tokens):
            start_ids = start_ids[..., -context_size:]
            logits = model(start_ids)
            last_logits = logits[..., -1, :]
            if top_k is not None:
                values, _ = torch.topk(last_logits, k = top_k, dim = -1)
                last_logits = torch.where(
                    condition = last_logits < values[..., -1],
                    input = torch.tensor(float('-inf')).to(last_logits.device),
                    other = last_logits
                )
            if temperature > 0.:
                last_logits = last_logits / temperature
            probs = torch.softmax(last_logits, dim = -1)
            sampled_tokens = torch.multinomial(probs, num_samples = 1)
            start_ids = torch.cat([start_ids, sampled_tokens], dim = -1)
    return start_ids

def generate_and_print_sample(
    model: torch.nn.Module,
    tokenizer: tiktoken.core.Encoding,
    device: torch.device,
    start_context: str,
    **kwargs
):
    model.eval()
    context_size = model.pos_embed.weight.shape[0]
    start_ids = text_to_token_ids(start_context, tokenizer).to(device)
    gen_ids = generate_text(model, start_ids, 50, context_size, **kwargs)
    decoded_text = token_ids_to_text(gen_ids, tokenizer)
    decoded_text = decoded_text.replace("\n", " ")
    print(f"Output text: {decoded_text}")
    model.train()