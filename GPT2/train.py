import torch
import time
import math
from accelerate import Accelerator
from accelerate.utils import set_seed
from utils import GPTConfig, TrainConfig, ShakespeareDataset
from torch.utils.data import DataLoader
from model import GPT
from transformers import get_cosine_schedule_with_warmup


# Random seed for reproducibility
seed = 3377
set_seed(seed)

# Initialize accelerator
accelerator = Accelerator(mixed_precision="fp16", step_scheduler_with_optimizer=False)
device = accelerator.device
print(f"Using device: {device}")

config = GPTConfig(vocab_size=50304) # Make vocab size a "nice" number by adding dummy vocab tokens
model = GPT(config)
# model = torch.compile(model) # Always compile model for faster training time (at least without accelerate)

train_config = TrainConfig()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_config.max_lr,
    betas=(train_config.adam_beta1, train_config.adam_beta2),
    eps=train_config.adam_epsilon,
    weight_decay=0.1
    )

train_dataset = ShakespeareDataset(block_size=config.block_size, accelerator=accelerator)
train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size)
grad_accum_steps = train_config.effective_batch_size // (train_config.batch_size * config.block_size * accelerator.num_processes)
num_steps = train_config.num_epochs * len(train_loader) // (grad_accum_steps * accelerator.num_processes)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=train_config.warmup_steps, num_training_steps=num_steps * 1.5)

model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)

accelerator.print("Starting training")
accelerator.print(f"Epochs: {train_config.num_epochs}")
accelerator.print(f"Total training steps: {num_steps}")
accelerator.print(f"Total batch size: {train_config.effective_batch_size}")
accelerator.print(f"Gradient accumulation steps: {grad_accum_steps}")

# Training loop
for epoch in range(train_config.num_epochs):
    start_time = time.time()
    optimizer.zero_grad()
    step_loss = 0.0

    for idx, batch in enumerate(train_loader):
        x, y = batch
        step_count = epoch * len(train_loader) + idx + 1
        with accelerator.autocast():
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        step_loss += loss.detach().item()
        accelerator.backward(loss)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping

        if (idx + 1) % grad_accum_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
            torch.cuda.synchronize()
            end_time = time.time()
            runtime = (end_time - start_time) * 1000
            final_loss = step_loss

            # Gather metrics from all processes
            gathered_loss = accelerator.gather(torch.tensor(final_loss).to(device)).mean().item()
            gathered_step_time = accelerator.gather(torch.tensor(runtime).to(device)).mean().item()
            step_loss = 0.0
            start_time = time.time()
            accelerator.print(f"Step {step_count // grad_accum_steps} | Loss: {gathered_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f} | Grad Norm: {norm:.4f} | Time: {gathered_step_time:.2f} ms | tok/sec: {train_config.effective_batch_size * 1000 / gathered_step_time:.2f}")
    
    accelerator.save_state(f"training/Checkpoints/epoch_{epoch}")

if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()
