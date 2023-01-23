import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
batch_size = 32
block_size = 8  # maximum context length for prediction
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

# --------------------
print('Running on hardware:', device)

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# unique characters from text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# mapping of characters to index
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s): return [stoi[c] for c in s]
def decode(l): return "".join([itos[i] for i in l])


# train and test split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading


def get_batch(split):
    # generate batch of data
    data = train_data if split == 'train' else val_data

    ix = torch.randint(len(data)-block_size, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):

        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are (B,T) tensor of integers
        logits = self.token_embedding_table(idx)  # B,T,C

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape

            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in current context

        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self(idx)

            # focus only on last timestemp
            logits = logits[:, -1, :]  # becomes B,C

            # convert to probability
            probs = F.softmax(logits, dim=-1)  # B,C

            # sample from distribution

            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)

            # append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


model = BigramLanguageModel(vocab_size)
m = model.to(device)

# optimizer
# small models work with higher lr
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while eval loss on train and valid set
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample batch
    xb, yb = get_batch('train')

    # evaluate loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
