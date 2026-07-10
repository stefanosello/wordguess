"""Tests for the training script (model/train.py).

Everything here uses a tiny model and a tiny synthetic corpus so it stays fast
on CPU — never the real config or the real corpus.
"""

import torch

from model.gpt import GPT, GPTConfig
from model.tokenizer import CharTokenizer
from model.train import (
    TrainConfig,
    cross_entropy_loss,
    generate,
    get_batch,
    load_ids,
    train,
)

# A highly repetitive corpus so a tiny model can obviously reduce its loss.
TINY_TEXT = "hello world. " * 400


def _tiny_cfg(tmp_path, **overrides) -> TrainConfig:
    corpus = tmp_path / "input.txt"
    corpus.write_text(TINY_TEXT, encoding="utf-8")
    cfg = TrainConfig(
        corpus=corpus,
        out_dir=tmp_path / "checkpoints",
        n_layer=1,
        n_head=2,
        n_embd=16,
        block_size=8,
        batch_size=16,
        max_iters=120,
        learning_rate=1e-2,
        eval_interval=40,
        eval_iters=5,
        sample_tokens=16,
        seed=0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_get_batch_shapes_and_shift():
    data = torch.arange(100)
    x, y = get_batch(data, block_size=8, batch_size=4, device="cpu")
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    # y is x shifted one position to the right (next-char targets).
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_get_batch_rejects_too_short_split():
    data = torch.arange(5)
    try:
        get_batch(data, block_size=8, batch_size=2, device="cpu")
    except ValueError:
        return
    raise AssertionError("expected ValueError for a split shorter than block_size")


def test_load_ids_splits_train_and_val():
    tok = CharTokenizer.from_text(TINY_TEXT)
    train_data, val_data = load_ids(TINY_TEXT, tok, val_fraction=0.1)
    assert len(train_data) + len(val_data) == len(TINY_TEXT)
    assert len(val_data) == int(len(TINY_TEXT) * 0.1)


def test_cross_entropy_loss_matches_manual():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 5)  # (batch, seq, vocab)
    targets = torch.randint(0, 5, (2, 3))
    expected = torch.nn.functional.cross_entropy(logits.view(6, 5), targets.view(6))
    assert torch.allclose(cross_entropy_loss(logits, targets), expected)


def test_training_reduces_loss_and_writes_checkpoint(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    result = train(cfg)

    first_val = result["history"][0][1]["val"]
    last_val = result["history"][-1][1]["val"]
    # The whole point of training: validation loss goes down.
    assert last_val < first_val

    # Checkpoint exists and carries weights, config, and vocab.
    ckpt_path = result["checkpoint"]
    assert ckpt_path.exists()
    ckpt = torch.load(ckpt_path, weights_only=False)
    assert set(ckpt) == {"model_state", "config", "vocab"}
    assert ckpt["config"]["vocab_size"] == len(ckpt["vocab"])


def test_checkpoint_reloads_and_generates_in_vocab(tmp_path):
    # Rebuild the model straight from the saved checkpoint (the serving path),
    # then sample and confirm every generated character is decodable.
    cfg = _tiny_cfg(tmp_path, max_iters=1)  # cheap: we only need a loadable model
    result = train(cfg)

    ckpt = torch.load(result["checkpoint"], weights_only=False)
    tokenizer = CharTokenizer(ckpt["vocab"])
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model_state"])  # must match exactly

    sample = generate(model, tokenizer, cfg, prompt="h", max_new_tokens=20)
    assert isinstance(sample, str)
    assert all(ch in tokenizer.stoi for ch in sample)
