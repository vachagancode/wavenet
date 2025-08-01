import torch

def greedy_decode(logits):
    return torch.argmax(logits, dim=-1, keepdim=True).permute(0, 2, 1)
