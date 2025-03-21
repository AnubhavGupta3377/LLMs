from dataclasses import dataclass
import tiktoken
import torch


@dataclass
class GPTConfig:
    block_size: int = 1024      # max seq length
    vocab_size: int = 50257     # vocabulary
    n_layer: int = 12          # number of transformer layers
    n_embed: int = 768          # embedding dimension
    n_head: int = 12             # number of attention heads


@dataclass
class TrainConfig:
    adam_epsilon: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    warmup_steps: int = 10
    max_steps: int = 1000
    max_lr: float = 6e-4
    training_steps: int = 1000


class DataLoader:
    """ Load data from a file """
    def __init__(self, batch_size, block_size):
        self.batch_size = batch_size
        self.block_size = block_size

        with open("dataset/input.txt", "r") as f:
            data = f.read()
        encoder = tiktoken.get_encoding("gpt2")
        tokens = encoder.encode(data)
        self.tokens = torch.tensor(tokens)
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"Number of batches per epoch: {len(self.tokens) // (batch_size * block_size)}")

        self.batch_start = 0

    def get_next_batch(self):
        """ Get next batch """
        if self.batch_start + self.batch_size * self.block_size >= len(self.tokens):
            self.batch_start = 0

        buffer = self.tokens[self.batch_start:self.batch_start + self.batch_size * self.block_size + 1]
        
        inputs = buffer[:-1].view(self.batch_size, self.block_size)
        targets = buffer[1:].view(self.batch_size, self.block_size)

        self.batch_start += self.batch_size * self.block_size
        return inputs, targets
