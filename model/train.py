"""Train a char-level GPT on a single corpus.

The whole loop, spelled out:

    read text -> build tokenizer -> encode to ids -> split train/val
    for each step: sample a batch, predict next char, cross-entropy loss,
                   AdamW step; every so often, print train/val loss
    save a checkpoint (weights + config + vocab), then print a text sample.

Everything runs on CPU with small defaults (see CLAUDE.md invariants). Usage:

    python -m model.train --corpus corpora/tinyshakespeare/input.txt

Lower --max-iters for a quicker run; the loss starts dropping within the first
few hundred steps.
"""

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from model.gpt import GPT, GPTConfig
from model.tokenizer import CharTokenizer


@dataclass
class TrainConfig:
    """Everything the training run needs. Defaults are CPU-friendly."""

    corpus: Path
    out_dir: Path | None = None  # defaults to <corpus dir>/checkpoints
    # model shape (mirrors GPTConfig defaults)
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 128
    dropout: float = 0.0
    # optimization
    batch_size: int = 32
    max_iters: int = 2000
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    # evaluation / logging
    eval_interval: int = 250
    eval_iters: int = 100
    val_fraction: float = 0.1
    # sampling at the end
    sample_tokens: int = 300
    # misc
    seed: int = 1337
    device: str = "cpu"


def load_ids(
    text: str, tokenizer: CharTokenizer, val_fraction: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode the whole corpus once and split off a validation tail."""
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)  # (len,)
    n_val = int(len(ids) * val_fraction)
    if n_val == 0:  # tiny corpora: reuse the whole thing for both splits
        return ids, ids
    return ids[:-n_val], ids[-n_val:]


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random (x, y) batch where y is x shifted one char to the right.

    x[b, t] is the context character and y[b, t] is the *next* character the
    model should predict — the supervision signal for language modelling.
    """
    if len(data) <= block_size:
        raise ValueError(
            f"split of {len(data)} tokens is too short for block_size {block_size}"
        )
    ix = torch.randint(len(data) - block_size, (batch_size,))  # (batch,) start offsets
    x = torch.stack([data[i : i + block_size] for i in ix])  # (batch, block_size)
    y = torch.stack(
        [data[i + 1 : i + block_size + 1] for i in ix]
    )  # (batch, block_size)
    return x.to(device), y.to(device)


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Flatten (batch, seq, vocab) logits and (batch, seq) targets for CE loss."""
    B, T, vocab_size = logits.shape
    return F.cross_entropy(logits.view(B * T, vocab_size), targets.view(B * T))


@torch.no_grad()
def estimate_loss(
    model: GPT, splits: dict[str, torch.Tensor], cfg: TrainConfig
) -> dict[str, float]:
    """Average loss over a few random batches per split (smooths the noise)."""
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_batch(data, cfg.block_size, cfg.batch_size, cfg.device)
            losses[k] = cross_entropy_loss(model(x), y).item()
        out[name] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def generate(
    model: GPT,
    tokenizer: CharTokenizer,
    cfg: TrainConfig,
    prompt: str,
    max_new_tokens: int,
) -> str:
    """Autoregressively sample text from the model, one character at a time."""
    model.eval()
    idx = torch.tensor(
        [tokenizer.encode(prompt)], dtype=torch.long, device=cfg.device
    )  # (1, len)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -cfg.block_size :]  # crop to the last block_size tokens
        logits = model(idx_cond)[
            :, -1, :
        ]  # (1, vocab) — only the last position matters
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(
            probs, num_samples=1
        )  # (1, 1) sample from the distribution
        idx = torch.cat([idx, next_id], dim=1)
    model.train()
    return tokenizer.decode(idx[0].tolist())


def save_checkpoint(
    model: GPT, model_config: GPTConfig, tokenizer: CharTokenizer, cfg: TrainConfig
) -> Path:
    """Write weights + config + vocab so serving can rebuild the exact model."""
    out_dir = (
        Path(cfg.out_dir) if cfg.out_dir else Path(cfg.corpus).parent / "checkpoints"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ckpt.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(model_config),
            "vocab": tokenizer.stoi,
        },
        path,
    )
    return path


def train(cfg: TrainConfig) -> dict:
    """Run the full training loop and return a small results dict."""
    torch.manual_seed(cfg.seed)

    text = Path(cfg.corpus).read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    train_data, val_data = load_ids(text, tokenizer, cfg.val_fraction)
    splits = {"train": train_data, "val": val_data}
    print(
        f"corpus: {len(text):,} chars | vocab: {tokenizer.vocab_size} | "
        f"train/val tokens: {len(train_data):,}/{len(val_data):,}"
    )

    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
    )
    model = GPT(model_config).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.2f}M parameters on {cfg.device}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    history: list[tuple[int, dict[str, float]]] = []
    start = time.time()
    for it in range(cfg.max_iters):
        if it % cfg.eval_interval == 0 or it == cfg.max_iters - 1:
            losses = estimate_loss(model, splits, cfg)
            history.append((it, losses))
            print(
                f"iter {it:5d} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | {time.time() - start:.0f}s"
            )

        x, y = get_batch(train_data, cfg.block_size, cfg.batch_size, cfg.device)
        loss = cross_entropy_loss(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    checkpoint = save_checkpoint(model, model_config, tokenizer, cfg)
    print(f"\nsaved checkpoint -> {checkpoint}")

    seed_char = "\n" if "\n" in tokenizer.stoi else text[0]
    sample = generate(
        model, tokenizer, cfg, prompt=seed_char, max_new_tokens=cfg.sample_tokens
    )
    print("\n--- sample ---\n" + sample + "\n--------------")

    return {"history": history, "checkpoint": checkpoint, "sample": sample}


def _build_parser() -> argparse.ArgumentParser:
    defaults = TrainConfig(corpus=Path("placeholder"))  # for default values only
    p = argparse.ArgumentParser(description="Train a char-level GPT on one corpus.")
    p.add_argument(
        "--corpus", type=Path, required=True, help="path to the corpus text file"
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="checkpoint dir (default: <corpus>/../checkpoints)",
    )
    p.add_argument("--n-layer", type=int, default=defaults.n_layer)
    p.add_argument("--n-head", type=int, default=defaults.n_head)
    p.add_argument("--n-embd", type=int, default=defaults.n_embd)
    p.add_argument("--block-size", type=int, default=defaults.block_size)
    p.add_argument("--dropout", type=float, default=defaults.dropout)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--max-iters", type=int, default=defaults.max_iters)
    p.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    p.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    p.add_argument("--eval-interval", type=int, default=defaults.eval_interval)
    p.add_argument("--eval-iters", type=int, default=defaults.eval_iters)
    p.add_argument("--val-fraction", type=float, default=defaults.val_fraction)
    p.add_argument("--sample-tokens", type=int, default=defaults.sample_tokens)
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--device", type=str, default=defaults.device)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    cfg = TrainConfig(**vars(args))
    train(cfg)


if __name__ == "__main__":
    main()
