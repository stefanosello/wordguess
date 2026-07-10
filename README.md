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

## Run with Docker

Docker gives you the exact same CPU-only environment on any machine — handy if
your system Python is old or you'd rather not manage a venv. You only need
Docker installed.

Serve the API:

```bash
docker compose up --build          # http://localhost:8000
curl localhost:8000/health         # -> {"status":"ok"}
```

Train a model (once the training script is available — see `model/train.py`).
The checkpoint lands in `./corpora` on your host, because that directory is
mounted into the container:

```bash
docker compose run --rm app \
  python -m model.train --corpus corpora/tinyshakespeare/input.txt
```

The image installs the CPU build of PyTorch, so it stays around ~1 GB rather
than pulling the multi-gigabyte CUDA wheel.

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
