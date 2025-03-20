import torch
from model import GPT
import tiktoken
from torch.nn import functional as F

model = GPT.from_pretrained("gpt2")
model.eval()
print("Loaded")

encoder = tiktoken.get_encoding("gpt2")
text = "Hello, I'm a language model,"
tokens = encoder.encode(text) # Get list of token Ids
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(5, 1) # Repeat the tensor 5 times
device = "cpu"
x = tokens.to(device)

# Generate text
random_seed = 42
max_length = 30
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)[:, -1, :] # (batch_size, n_vocab)
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
