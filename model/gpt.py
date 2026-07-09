"""A small char-level GPT in the nanoGPT lineage.

The whole point of this file is to be *read*. Each tensor operation carries a
shape comment, because in a transformer shape confusion is the #1 source of
bugs and of misunderstanding. The pieces, top to bottom:

    token + position embeddings  ->  turn ids into vectors that know their order
    Block (xN):                      pre-norm residual around attention + MLP
      CausalSelfAttention            each position looks only *backward*
      MLP                            per-position feed-forward
    final LayerNorm + linear head    vectors -> a score per vocab character

There is no training loop, tokenizer, or sampling here — just the module and a
plain forward pass (see the sibling issues). Run `python -m model.gpt` for a
quick shape sanity check.

Further reading (the whole architecture, explained well):
  - Vaswani et al., "Attention Is All You Need" — the transformer paper:
    https://arxiv.org/abs/1706.03762
  - Karpathy, "Let's build GPT: from scratch, in code, spelled out" (video):
    https://www.youtube.com/watch?v=kCc8FmEb1nY
  - Karpathy, nanoGPT — the reference this file is a trimmed, commented copy of:
    https://github.com/karpathy/nanoGPT
  - Alammar, "The Illustrated Transformer" — the best visual intro:
    https://jalammar.github.io/illustrated-transformer/
  - Alammar, "The Illustrated GPT-2" — decoder-only + masked self-attention:
    https://jalammar.github.io/illustrated-gpt2/
  - Rush et al., "The Annotated Transformer" — paper as runnable code:
    https://nlp.seas.harvard.edu/annotated-transformer/
Each component below also links the specific resource that explains it.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """Hyperparameters for the model. Defaults are deliberately tiny — CPU-only,
    megabytes not gigabytes (see CLAUDE.md invariants).

    Char-level modelling (one token = one character) is the simplest tokenizer
    and keeps the vocab tiny; Karpathy's char-RNN post motivates it well:
    https://karpathy.github.io/2015/05/21/rnn-effectiveness/
    """

    vocab_size: int = 65  # number of distinct characters (a char-level vocab is small)
    block_size: int = 128  # maximum context length in tokens
    n_layer: int = 4  # number of transformer blocks
    n_head: int = 4  # attention heads per block
    n_embd: int = 128  # embedding / residual-stream width
    dropout: float = 0.0  # 0.0 keeps the forward pass deterministic by default


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where position t may attend only to 0..t.

    The causal mask is what makes this a *language* model: when predicting the
    next character at position t, the model must not peek at positions > t.

    Concepts and where they're explained:
      - Scaled dot-product + multi-head attention: "Attention Is All You Need"
        §3.2 (https://arxiv.org/abs/1706.03762).
      - The query/key/value intuition, drawn out: "The Illustrated Transformer"
        (https://jalammar.github.io/illustrated-transformer/).
      - Masked (causal) self-attention specifically: "The Illustrated GPT-2"
        (https://jalammar.github.io/illustrated-gpt2/).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # One projection produces query, key, and value together, then we split.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Projection applied to the concatenated head outputs.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Lower-triangular matrix of ones; a 0 marks a (query, key) pair that is
        # in the future and must be masked out. Stored as a buffer (not a
        # parameter) so it moves with the model but isn't trained.
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        # Shape (1, 1, block_size, block_size) to broadcast over batch and heads.
        mask = mask.view(1, 1, config.block_size, config.block_size)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # (batch, seq_len, n_embd)
        head_dim = C // self.n_head

        # Project to q, k, v — each (B, T, C) — then split heads.
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim): heads become a batch dimension.
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # Scaled dot-product attention scores: (B, n_head, T, T).
        # Dividing by sqrt(head_dim) keeps the logits from growing with head
        # width, which would push softmax into tiny-gradient regions — see
        # "Attention Is All You Need" §3.2.1 (https://arxiv.org/abs/1706.03762).
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        # Mask the upper triangle (the future) with -inf so softmax zeros it:
        # e^(-inf) = 0, so position t gets exactly zero weight on positions > t.
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)  # (B, n_head, T, T), rows sum to 1
        att = self.attn_dropout(att)

        y = att @ v  # (B, n_head, T, head_dim): weighted sum over the past
        # Reassemble heads back into the residual stream: (B, T, C).
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Per-position feed-forward network: widen 4x, GELU, project back.

    The position-wise feed-forward layer is described in "Attention Is All You
    Need" §3.3 (https://arxiv.org/abs/1706.03762); the 4x width is the standard
    ratio there. GELU is the smooth activation from Hendrycks & Gimpel,
    "Gaussian Error Linear Units" (https://arxiv.org/abs/1606.08415).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)  # (B, T, 4 * n_embd)
        x = self.gelu(x)
        x = self.c_proj(x)  # (B, T, n_embd)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """One transformer block: pre-norm residual around attention, then MLP.

    Two ideas here:
      - Residual connections (the `x + ...`) let gradients flow through deep
        stacks — from He et al., "Deep Residual Learning"
        (https://arxiv.org/abs/1512.03385).
      - LayerNorm placed *before* each sub-layer ("pre-norm") trains more
        stably than the original post-norm; see Xiong et al., "On Layer
        Normalization in the Transformer Architecture"
        (https://arxiv.org/abs/2002.04745). LayerNorm itself: Ba et al.
        (https://arxiv.org/abs/1607.06450).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LayerNorm *before* each sub-layer, and add the input back (residual).
        x = x + self.attn(self.ln_1(x))  # (B, T, n_embd)
        x = x + self.mlp(self.ln_2(x))  # (B, T, n_embd)
        return x


class GPT(nn.Module):
    """A minimal char-level GPT: embeddings -> blocks -> linear head over vocab."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token ids -> vectors; positions 0..block_size-1 -> vectors.
        # nn.Embedding is a lookup table (row i = vector for id i):
        # https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        # These are *learned* positional embeddings (GPT-2 style). The original
        # transformer instead used fixed sinusoids ("Attention Is All You Need"
        # §3.5, https://arxiv.org/abs/1706.03762); learned is simpler and what
        # nanoGPT uses.
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)  # final norm before the head
        # Head maps each position's vector to a logit per vocab character.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Map a batch of token ids to next-token logits.

        idx:    (batch, seq_len) int tensor of token ids
        returns (batch, seq_len, vocab_size) logits — the raw scores per
        character at every position.
        """
        B, T = idx.size()  # (batch, seq_len)
        assert T <= self.config.block_size, (
            f"sequence length {T} exceeds block size {self.config.block_size}"
        )

        pos = torch.arange(T, device=idx.device)  # (T,) positions 0..T-1
        tok_emb = self.token_embedding(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding(pos)  # (T, n_embd)
        x = self.drop(tok_emb + pos_emb)  # (B, T, n_embd) — pos broadcasts over batch

        for block in self.blocks:
            x = block(x)  # (B, T, n_embd)
        x = self.ln_f(x)  # (B, T, n_embd)

        # One logit per vocab character, at every position. We stop at raw
        # logits: apply softmax over the last dim to get the model's next-char
        # distribution (done by the caller / puzzle code, not here). How logits
        # become a prediction: "The Illustrated GPT-2"
        # (https://jalammar.github.io/illustrated-gpt2/).
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits


if __name__ == "__main__":
    # Quick sanity check: a random batch in, correctly-shaped logits out.
    config = GPTConfig()
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (2, config.block_size))  # (2, block_size)
    logits = model(idx)
    print("input  :", tuple(idx.shape))
    print("logits :", tuple(logits.shape))  # expect (2, block_size, vocab_size)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params : {n_params / 1e6:.2f}M")
