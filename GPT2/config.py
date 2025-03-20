from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024      # max seq length
    vocab_size: int = 50257     # vocabulary
    n_layer: int = 12          # number of transformer layers
    n_embed: int = 768          # embedding dimension
    n_head: int = 12             # number of attention heads
