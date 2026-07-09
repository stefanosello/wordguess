"""A minimal char-level tokenizer.

The model only ever sees integers, so something has to translate between text
and ids. Char-level keeps that mapping trivial to inspect: the vocabulary is
just the set of distinct characters in the corpus, sorted, numbered 0..N-1.

    encode("hi") -> [45, 46]      decode([45, 46]) -> "hi"

`stoi` (string->int) and `itos` (int->string) are the whole tokenizer. They are
saved next to a checkpoint as JSON so training and serving always agree on the
mapping — a mismatch would silently scramble every prediction.

Run `python -m model.tokenizer` for a round-trip demo.
"""

import json
from pathlib import Path


class CharTokenizer:
    """Maps characters to ids and back, built from a corpus's unique characters."""

    def __init__(self, stoi: dict[str, int]):
        # stoi: character -> id. itos is just its inverse, cached for decoding.
        self.stoi = stoi
        self.itos = {i: ch for ch, i in stoi.items()}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Build a vocabulary from the distinct characters in `text`.

        Characters are sorted so the mapping is deterministic: the same corpus
        always yields the same ids, whoever builds it.
        """
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        return cls(stoi)

    @property
    def vocab_size(self) -> int:
        """Number of distinct characters — this is the model's output width."""
        return len(self.stoi)

    def encode(self, s: str) -> list[int]:
        """Turn a string into a list of token ids.

        Raises KeyError (loudly) on any character not in the vocabulary, rather
        than guessing — an unknown character means the wrong corpus/tokenizer.
        """
        ids = []
        for ch in s:
            if ch not in self.stoi:
                raise KeyError(f"character {ch!r} is not in the vocabulary")
            ids.append(self.stoi[ch])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Turn a list of token ids back into a string.

        Raises KeyError (loudly) on any id outside [0, vocab_size); silently
        dropping or clamping it would corrupt the output invisibly.
        """
        chars = []
        for i in ids:
            if i not in self.itos:
                raise KeyError(f"token id {i} is out of range [0, {self.vocab_size})")
            chars.append(self.itos[i])
        return "".join(chars)

    def save(self, path: str | Path) -> None:
        """Write the vocabulary to `path` as JSON (a character->id object)."""
        with open(path, "w", encoding="utf-8") as f:
            # ensure_ascii=False keeps non-ASCII characters human-readable.
            json.dump(self.stoi, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        """Load a vocabulary previously written by `save`."""
        with open(path, encoding="utf-8") as f:
            stoi = json.load(f)
        return cls(stoi)


if __name__ == "__main__":
    sample = "hello, world!"
    tok = CharTokenizer.from_text(sample)
    ids = tok.encode(sample)
    print("vocab_size:", tok.vocab_size)
    print("encode    :", ids)
    print("decode    :", tok.decode(ids))
    assert tok.decode(tok.encode(sample)) == sample  # round-trip holds
