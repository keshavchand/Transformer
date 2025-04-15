import math
import torch
import torch.nn as nn
from torch.nn import functional as F

with open("dataset.txt") as f:
    dataset = f.read()

vocab = ''.join(sorted(set(dataset)))
vocabSize = len(vocab)
n_embed = 32
    
chToInt = {ch : i  for i, ch in enumerate(vocab)}
intToCh = {i  : ch for i, ch in enumerate(vocab)}
print('Dataset: ', ''.join(sorted(chToInt)))
print('VocabSize: ', vocabSize)

encoder = lambda s: [chToInt[si] for si in s]
decoder = lambda s: ''.join(intToCh[si] for si in s)

dataset = torch.tensor([0] + encoder(dataset) + [0], dtype=torch.int64)

# Train test split
n           = int(len(dataset) * 0.9)
train_data  = dataset[n:]
val_data    = dataset[:n]
blockSize   = 8  # Context length of the model
batchSize   = 32 # How many will we train at once
device      = 'cuda' if torch.cuda.is_available() else 'cpu'

@torch.no_grad
def estimate_loss(model, iterCount = 100):
    out = {}
    model.eval()

    inputs = ['train', 'test']
    for i in inputs:
        splits = torch.zeros((iterCount), dtype=torch.float32)
        for k in range(iterCount):
            xb, yb = get_batch(i)
            _, loss = model.forward(xb, yb)
            splits[k] = loss.item()
            
        out[i] = splits.mean()
    model.train()
    return out

@torch.compile
def scaled_dot_product_attention(Q, K, V):
    wei = Q @ K.transpose(-2, -1)
    wei = wei.tril()
    wei = wei.masked_fill(wei == 0, float('-inf'))
    wei = wei / torch.sqrt(Q.shape[-1])
    wei = F.softmax(wei, dim = -1)
    return wei @ V

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - blockSize, (batchSize, ))
    x = torch.stack([data[i + 0 : i + blockSize + 0] for i in ix])
    y = torch.stack([data[i + 1 : i + blockSize + 1] for i in ix])
    return x.to(device), y.to(device)

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.seq = nn.Sequential(
                nn.Linear(n_embed, 4 * n_embed),
                nn.ReLU(),
                nn.Linear(4 * n_embed, n_embed),
            )

    def forward(self, ix):
        return self.seq(ix)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.K = nn.Linear(n_embed, head_size, device=device, bias=False);
        self.Q = nn.Linear(n_embed, head_size, device=device, bias=False);
        self.V = nn.Linear(n_embed, head_size, device=device, bias=False);
        self.register_buffer('tril', torch.tril(torch.ones((blockSize, blockSize), device=device)))

    def forward(self, ix):
        B, T, C = ix.shape
        Q = self.Q(ix) # B, T, C
        K = self.K(ix) # B, T, C
        V = self.V(ix) # B, T, C

        wei = (Q @ K.transpose(-2, -1)) * C ** 0.5 # Scaling
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # causal
        wei = F.softmax(wei, dim = -1) # B, T, T
        wei = wei @ V # B, T, C
        return wei

class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, head_count):
        super().__init__()
        self.heads = [Head(head_size) for n in range(head_count)]
        self.projection = nn.Linear(n_embed, n_embed, device=device)

    def forward(self, ix):
        # ix (B, T, n_embed)
        out = torch.cat(tuple(head(ix) for head in self.heads), dim = -1)
        out = self.projection(out)
        return out


class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.mha = MultiHeadAttention(n_embed // n_head, n_head)
        self.ffn = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, ix):
        ix = ix + self.mha(self.ln1(ix))
        ix = ix + self.ffn(self.ln2(ix))
        return ix


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenEmbeddingTable = nn.Embedding(vocabSize, n_embed) # Embedding table
        self.positionEmbeddingTable = nn.Embedding(blockSize, n_embed) # Position Embedding table
        # self.sa_head = Head(n_embed)
        # self.sa_heads = MultiHeadAttention(n_embed // 4, 4)
        # self.net = FeedForward(n_embed)
        self.blocks = nn.Sequential(
                Block(n_embed, 4),
                Block(n_embed, 4),
                Block(n_embed, 4),
                Block(n_embed, 4),
            )
        self.lm_head = nn.Linear(n_embed, vocabSize)

    def forward(self, idx, target=None):
        B, T = idx.shape
        # Idx is in shape (B, T)
        token_embeddings =  self.tokenEmbeddingTable(idx) # logits is in (B,T, C = n_embed)
        token_embeddings += self.positionEmbeddingTable(torch.arange(T, device=idx.device))
        token_embeddings =  self.blocks(token_embeddings)
        logits = self.lm_head(token_embeddings) # (B, T, C = vocabSize)
        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, -1)
            target = target.view(B * T)
            loss = F.cross_entropy(logits, target)
        return logits, loss

    def generate(self, idx, newMaxSize):
        for i in range(newMaxSize):
            new_idx = idx[:, -blockSize:]
            logits, loss = self(new_idx) # Logits is (B, T, C)
            newToken = logits[:, -1, :] # (B, C)
            prob = F.softmax(newToken, dim = -1) # (B, C)
            idx_next = torch.multinomial(prob, num_samples = 1)
            idx = torch.cat((idx, idx_next), dim = 1) # (B, T + 1)

        return idx

model = GPTLanguageModel().to(device)
# logits, loss = model.forward(xb, yb)
# print(logits, loss)
# 
# idx = torch.zeros((1, 1), dtype=torch.long)
# batch1 = model.generate(idx, newMaxSize=100)[0].tolist()
# print(decoder(batch1))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for step in range(3000):
    xb, yb = get_batch('train')

    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        loss = estimate_loss(model)
        print(f"Train Loss: {loss['train']}, Validation Loss: {loss['test']}")

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
for step in range(3000):
    xb, yb = get_batch('train')

    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        loss = estimate_loss(model)
        print(f"Train Loss: {loss['train']}, Validation Loss: {loss['test']}")

generatedData = model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), newMaxSize=1000)[0].tolist()
print(decoder(generatedData))
