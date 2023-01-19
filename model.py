
import dataclasses
import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclasses.dataclass
class Args:
    block_size: int = 0
    n_embd: int = 512
    n_head: int = 8
    n_layer: int = 6
    dropout: float = 0.1
    head_size: int = 64
    vocab_size: int = 0
    device: str = "cpu"
    

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, args , head_size):
        super().__init__()
        # linear layers for the query, key, and value
        self.key = nn.Linear(args.n_embd, head_size, bias=False)
        self.query = nn.Linear(args.n_embd, head_size, bias=False)
        self.value = nn.Linear(args.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(args.block_size, args.block_size)))

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self,args , head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(args, head_size) for _ in range(args.n_head)])
        self.proj = nn.Linear(args.n_embd, args.n_embd)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self,args):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(args.n_embd, 4 * args.n_embd),
            nn.ReLU(),
            nn.Linear(4 * args.n_embd,args.n_embd),
            nn.Dropout(args.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self,args):
        # args.n_embd: embedding dimension, args.n_head: the number of heads we'd like
        super().__init__()
        head_size = args.n_embd // args.n_head
        self.sa = MultiHeadAttention(args, head_size)
        self.ffwd = FeedFoward(args)
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class LanguageModel(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        print(self.args)
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(args.vocab_size, args.n_embd)
        self.position_embedding_table = nn.Embedding(args.block_size, args.n_embd)
        self.blocks = nn.Sequential(*[Block(args) for _ in range(args.n_layer)])
        self.ln_f = nn.LayerNorm(args.n_embd) # final layer norm
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.args.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
        
        return logits

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last args.block_size tokens
            idx_cond = idx[:, -args.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    

if __name__ == '__main__':
    # create namespace args and add placeholder values for arguments
    import argparse
    args = argparse.Namespace()
    args.vocab_size = 1000
    args.block_size = 128
    args.n_embd = 512
    args.n_head = 8
    args.n_layer = 6
    args.dropout = 0.1
    args.device = 'cuda'

    # create a model
    model = BigramLanguageModel(args)
    model.to(args.device)