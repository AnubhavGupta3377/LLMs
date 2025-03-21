import torch
import time
import torch.optim as optim
import math
from utils import GPTConfig, TrainConfig, DataLoader
from model import GPT


def get_lr(warmup_steps, max_steps, max_lr, min_lr, current_step):
    if current_step < warmup_steps:
        return max_lr * float(current_step + 1) / warmup_steps
    if current_step >= max_steps:
        return min_lr

    decay_ratio = (current_step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
    

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print(f"Using device: {device}")

# Random seed for reproducibility
torch.manual_seed(3377)
torch.cuda.manual_seed(3377)

## Use TF32 precision for matrix multiplications
# torch.set_float32_matmul_precision("high")

train_loader = DataLoader(batch_size=16, block_size=256)
config = GPTConfig(vocab_size=50304) # Make vocab size a "nice" number by adding dummy vocab tokens
model = GPT(config)
model.to(device)
# model = torch.compile(model) # Always compile model for faster training time

train_config = TrainConfig()
optimizer = torch.optim.AdamW(model.parameters(), betas=(train_config.adam_beta1, train_config.adam_beta2), eps=train_config.adam_epsilon)
# Training loop
for step in range(train_config.training_steps):
    start_time = time.time()
    x, y = train_loader.get_next_batch()
    x, y = x.to(device), y.to(device)

    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    learning_rate = get_lr(warmup_steps=train_config.warmup_steps, max_steps=train_config.max_steps, max_lr=train_config.max_lr, min_lr=train_config.max_lr * 0.1, current_step=step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    optimizer.step()
    
    torch.cuda.synchronize()
    end_time = time.time()
    runtime = (end_time - start_time) * 1000
    print(f"Step {step} | Loss: {loss.item():.6f} | LR: {learning_rate:.6f} | Grad Norm: {norm:.4f} | Time: {runtime:.2f} ms | tok/sec: {x.size(0) * x.size(1) / runtime:.2f}")
