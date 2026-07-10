# Tiny Shakespeare

`input.txt` is the "Tiny Shakespeare" dataset: a concatenation of the works of
William Shakespeare (~1.1 MB of plain text), long popularized by Andrej
Karpathy's char-rnn as a small char-level language-modelling benchmark.

- **Public domain.** Shakespeare's works are in the public domain, so the text
  is safe to commit (see CLAUDE.md invariant #5).
- **Source.** https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

Trained checkpoints for this corpus are written to `checkpoints/` here, which is
gitignored — weights are never committed.
