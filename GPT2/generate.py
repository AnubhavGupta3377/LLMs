import torch
from model import GPT
import tiktoken
from torch.nn import functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from utils import GPTConfig


def load_model(checkpoint_path=None):
    config = GPTConfig(vocab_size=50304) # Make vocab size a "nice" number by adding dummy vocab tokens
    model = GPT(config)
    model = accelerator.prepare(model)
    if checkpoint_path is not None:
        accelerator.load_state(checkpoint_path)
    return model

accelerator = Accelerator(mixed_precision="fp16")
device = accelerator.device
seed = 3377
set_seed(seed)

model = load_model()
model.eval()
print("Loaded model")

encoder = tiktoken.get_encoding("gpt2")
text = "Hello, I'm a language model,"
tokens = encoder.encode(text) # Get list of token Ids
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(5, 1) # Repeat the tensor 5 times

x = tokens.to(device)

# Generate text
max_length = 30
while x.size(1) < max_length:
    with torch.no_grad():
        logits, _ = model(x)
        logits = logits[:, -1, :] # (batch_size, n_vocab)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1) # (batch_size, 50)
        indices = torch.multinomial(topk_probs, num_samples=1) # (batch_size, 1)
        xcol = torch.gather(topk_indices, -1, indices)
        x = torch.cat((x, xcol), dim=1)

# Print generated text
for i in range(len(x)):
    tokens = x[i].tolist()
    decoded_text = encoder.decode(tokens)
    print("> ", decoded_text)
