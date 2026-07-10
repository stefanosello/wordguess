# wordguess

A tiny web game where you try to out-predict a small language model.

Pick a book the model was trained on, read an incomplete phrase, and guess which
next word the model thinks is most likely — chosen from a handful of candidate
words the model itself produced. Your score reflects how much probability mass
your pick actually carried: the model's 2nd choice hurts less than a wild guess.

The point of the project is **learning** — building real intuition for how an LLM
assigns probability one token at a time, and training and serving a small model
end to end. Readability is favored over cleverness everywhere.

## How the game works

Next-word puzzles via **greedy rollout**:

1. Take a real slice of the chosen corpus as the prompt.
2. At the branch point, look at the model's next-token distribution and take the
   top-K first tokens.
3. For each first token, greedily roll forward token-by-token until a word
   boundary, multiplying token probabilities along the way. Each rollout yields
   one candidate word plus its cumulative probability.
4. The highest-probability candidate is the correct answer; the rest are
   plausible-but-wrong distractors drawn from the model's own distribution.
5. You guess; the reveal shows the correct word and the full probability spread.

The **MVP** ships a simpler single-token version first (correct = argmax next
char, distractors = ranks 2..K) to prove the plumbing, then M2 upgrades to the
full next-word rollout.

## Architecture

```
Browser  ──POST /puzzle──▶  FastAPI  ──▶  one small char-level GPT per corpus
  guess   ──POST /guess───▶            (nanoGPT-style, resident in memory)
```

- **model/**  — char-level GPT definition, tokenizer, training script
- **server/** — FastAPI app: puzzle generation, rollout, scoring, in-memory cache
- **web/**    — single-page frontend (vanilla JS, no build step)
- **corpora/** — raw training texts + trained checkpoints (weights gitignored)
- **scripts/** — setup + tooling, and the issue backlog generator

Stack: Python, PyTorch, FastAPI, uvicorn. Model weights are **not** committed —
they are trained locally from the corpora.

## Dev setup

Everything runs on CPU — no GPU required. From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # torch (CPU build), fastapi, uvicorn
```

Run the server:

```bash
uvicorn server.app:app --reload
```

Then check it's alive:

```bash
curl localhost:8000/health   # -> {"status":"ok"}
```

Run the tests and linter:

```bash
pip install -r requirements-dev.txt   # pytest, httpx, ruff (dev deps)
pytest
ruff check .                          # lint; add --fix to auto-fix
```

## Train a model

Train a char-level GPT on a corpus. The loop reads the text, builds the
tokenizer, and learns to predict the next character; it prints train/val loss
as it goes and a short text sample at the end.

```bash
python -m model.train --corpus corpora/tinyshakespeare/input.txt
```

On a laptop CPU this takes a few minutes with the defaults; pass
`--max-iters 500` for a quicker taste, or `--help` to see all knobs. The
checkpoint (weights + config + vocab) is written to
`corpora/<name>/checkpoints/ckpt.pt`, which is **gitignored** — weights are
trained locally, never committed.

## API contract (target)

- `POST /puzzle` → `{corpus?}` → `{puzzle_id, prompt, choices[]}`
  (choices shuffled; the correct index is never leaked)
- `POST /guess`  → `{puzzle_id, choice_idx}` → `{correct_idx, probs[], your_prob, score?}`
- `GET /corpora` → available corpus names + short descriptions (M3)
- `GET /health`  → `{status: "ok"}`

## Backlog

The backlog lives on GitHub and is the single source of truth:

- **Issues** — <https://github.com/stefanosello/wordguess/issues> (grouped into
  milestones M1–M4)
- **Project board** — <https://github.com/users/stefanosello/projects/3>

Browse or edit work there; there is no generated backlog file to keep in sync.
