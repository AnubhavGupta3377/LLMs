from dataclasses import dataclass
import tiktoken
import torch
from torch.utils.data import Dataset


@dataclass
class GPTConfig:
    block_size: int = 256      # max seq length
    vocab_size: int = 50257     # vocabulary
    n_layer: int = 12          # number of transformer layers
    n_embed: int = 768          # embedding dimension
    n_head: int = 12             # number of attention heads


@dataclass
class TrainConfig:
    adam_epsilon: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    warmup_steps: int = 10 # Number of warmup steps for lr scheduler
    max_lr: float = 6e-4 # Maximum learning rate
    num_epochs: int = 5 # Number of training steps
    batch_size: int = 16
    effective_batch_size: int = 2**15 # 32k (For gradient accumulation)


class ShakespeareDataset(Dataset):
    def __init__(self, batch_size, block_size, accelerator=None):
        self.batch_size = batch_size
        self.block_size = block_size

        with open("dataset/input.txt", "r") as f:
            data = f.read()
        
        encoder = tiktoken.get_encoding("gpt2")
        tokens = encoder.encode(data)
        self.tokens = torch.tensor(tokens)
        num_samples = len(tokens) // block_size
        extra_tokens = len(tokens) - num_samples * block_size
        self.inputs = self.tokens[:-extra_tokens].view(num_samples, self.block_size)
        self.targets = self.tokens[1:-extra_tokens+1].view(num_samples, self.block_size)
        
        print_func = print if accelerator is None else accelerator.print
        print_func(f"Loaded {len(self.tokens)} tokens")
        print_func(f"Number of batches per epoch: {len(self.tokens) // (batch_size * block_size)}")
        print_func(f"Number of samples (with block_size = {block_size}): {num_samples}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
