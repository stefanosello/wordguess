"""Tests for the char-level GPT (model/gpt.py).

These pin the two contracts that matter for the game: the forward pass returns
per-position logits over the vocabulary, and attention is causal — position t
never sees the future.
"""

import torch

from model.gpt import GPT, GPTConfig

# A deliberately tiny config so the tests run on CPU in well under a second.
TINY = GPTConfig(vocab_size=11, block_size=8, n_layer=2, n_head=2, n_embd=16)


def test_forward_returns_logits_shaped_batch_time_vocab():
    model = GPT(TINY)
    idx = torch.randint(0, TINY.vocab_size, (3, TINY.block_size))  # (batch, seq_len)

    logits = model(idx)

    assert logits.shape == (3, TINY.block_size, TINY.vocab_size)


def test_causal_masking_hides_future_tokens():
    # Behavioral check: changing the LAST token must not move the logits at any
    # earlier position, because those positions can't attend to the future.
    torch.manual_seed(0)
    model = GPT(TINY)
    model.eval()

    idx = torch.randint(0, TINY.vocab_size, (1, TINY.block_size))
    with torch.no_grad():
        logits = model(idx)

        # Perturb only the last token.
        idx_changed = idx.clone()
        idx_changed[0, -1] = (idx[0, -1] + 1) % TINY.vocab_size
        logits_changed = model(idx_changed)

    # Every position before the last is untouched...
    assert torch.allclose(logits[:, :-1], logits_changed[:, :-1], atol=1e-6)
    # ...and the last position *did* change, so masking isn't just zeroing everything.
    assert not torch.allclose(logits[:, -1], logits_changed[:, -1], atol=1e-6)


def test_default_config_forward_runs_on_cpu():
    # The acceptance criterion: instantiate with defaults and run a forward pass.
    config = GPTConfig()
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (1, config.block_size))

    logits = model(idx)

    assert logits.shape == (1, config.block_size, config.vocab_size)


def test_sequence_longer_than_block_size_is_rejected():
    # Fail loudly rather than silently corrupting: too-long context raises.
    model = GPT(TINY)
    idx = torch.randint(0, TINY.vocab_size, (1, TINY.block_size + 1))

    try:
        model(idx)
    except AssertionError:
        return
    raise AssertionError("expected an AssertionError for seq_len > block_size")
