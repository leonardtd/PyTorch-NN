import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
batch_size = 64
block_size = 256  # maximum context length for prediction
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2
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


class Head(nn.Module):
    """one head self-attention"""

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)

        # computer attention scores ("affinities")
        wei = q @ k.transpose(1, 2) * C**-0.5  # (B,T,C)@(B,C,T) --> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B,T,T)
        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        wei = self.dropout(wei)  # PREVENT SOME NODES FROM COMUNICATION

        # perform weighted aggregation of values
        v = self.value(x)

        out = wei @ v

        return out


class MultiHeadAttention(nn.Module):
    """ multiple head self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non linearity """

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),  # projection layer
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed//n_head  # i.e. 4 heads of 8 dimensional self-attention
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)

        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # residual conections
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))  # pre-norm formula, PER TOKEN
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):

        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        self.blocks = nn.Sequential(
            *[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)  # final layer norm

        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # B,T,C
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device))  # T, C

        x = tok_emb + pos_emb  # broadcast
        x = self.blocks(x)  # apply N heads of self-attention (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab size)

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

            # crop idx to the last block_size tokens (for pos embedding)
            idx_cond = idx[:, -block_size:]

            # get predictions
            logits, loss = self(idx_cond)

            # focus only on last timestemp
            logits = logits[:, -1, :]  # becomes B,C

            # convert to probability
            probs = F.softmax(logits, dim=-1)  # B,C

            # sample from distribution

            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)

            # append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


model = BigramLanguageModel()
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
