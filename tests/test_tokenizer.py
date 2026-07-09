"""Tests for the char-level tokenizer (model/tokenizer.py)."""

import pytest

from model.tokenizer import CharTokenizer

CORPUS = "the quick brown fox jumps over the lazy dog.\nAND SOME CAPS 123!"


def test_encode_decode_round_trips_exactly():
    tok = CharTokenizer.from_text(CORPUS)
    # Any string drawn from the corpus must survive the round-trip untouched.
    assert tok.decode(tok.encode(CORPUS)) == CORPUS


def test_vocab_size_is_number_of_unique_chars():
    tok = CharTokenizer.from_text(CORPUS)
    assert tok.vocab_size == len(set(CORPUS))


def test_ids_are_contiguous_from_zero():
    tok = CharTokenizer.from_text(CORPUS)
    assert sorted(tok.stoi.values()) == list(range(tok.vocab_size))


def test_save_and_load_preserves_mapping(tmp_path):
    tok = CharTokenizer.from_text(CORPUS)
    path = tmp_path / "vocab.json"
    tok.save(path)

    loaded = CharTokenizer.load(path)

    assert loaded.stoi == tok.stoi
    assert loaded.vocab_size == tok.vocab_size
    # A loaded tokenizer must encode identically to the original.
    assert loaded.encode(CORPUS) == tok.encode(CORPUS)


def test_decode_out_of_range_id_raises():
    tok = CharTokenizer.from_text(CORPUS)
    with pytest.raises(KeyError):
        tok.decode([tok.vocab_size])  # one past the largest valid id


def test_encode_unknown_character_raises():
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(KeyError):
        tok.encode("z")  # 'z' was never in the corpus
