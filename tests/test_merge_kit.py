"""Offline tests. rank_concat is checked against the identity it promises;
the GGUF reader is checked against bytes we build in-test, including metadata
it must skip to reach the tensor directory."""

import struct

import numpy as np
import pytest

from merge_kit.lora_merge import rank_concat, delta
from merge_kit.gguf import read_header, read_tensor_names, has_speculative_heads, GgufError


# --- rank_concat ---

def test_rank_concat_matches_sum_of_effects():
    rng = np.random.default_rng(0)
    in_dim, out_dim, r1, r2 = 8, 5, 3, 4
    A1, B1 = rng.standard_normal((r1, in_dim)), rng.standard_normal((out_dim, r1))
    A2, B2 = rng.standard_normal((r2, in_dim)), rng.standard_normal((out_dim, r2))

    A, B = rank_concat(A1, B1, A2, B2)
    assert A.shape == (r1 + r2, in_dim)
    assert B.shape == (out_dim, r1 + r2)
    # The whole point: merged delta == sum of the two separate deltas.
    assert np.allclose(delta(A, B), delta(A1, B1) + delta(A2, B2))


def test_rank_concat_rejects_mismatched_shapes():
    A1, B1 = np.zeros((3, 8)), np.zeros((5, 3))
    with pytest.raises(ValueError):
        rank_concat(A1, B1, np.zeros((4, 9)), np.zeros((5, 4)))   # in dim differs
    with pytest.raises(ValueError):
        rank_concat(A1, B1, np.zeros((4, 8)), np.zeros((6, 4)))   # out dim differs
    with pytest.raises(ValueError):
        rank_concat(A1, np.zeros((5, 99)), np.zeros((4, 8)), np.zeros((5, 4)))  # rank mismatch


# --- gguf reader ---

def _s(text: str) -> bytes:
    b = text.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _tensor_info(name: str) -> bytes:
    # name, n_dims=1, dims=[4096], type=0, offset=0
    return _s(name) + struct.pack("<I", 1) + struct.pack("<Q", 4096) + struct.pack("<I", 0) + struct.pack("<Q", 0)


def _make_gguf(tensor_names, metadata=b"", kv_count=0) -> bytes:
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensor_names)) + struct.pack("<Q", kv_count)
    body = b"".join(_tensor_info(n) for n in tensor_names)
    return header + metadata + body


def test_read_tensor_names_no_metadata(tmp_path):
    p = tmp_path / "m.gguf"
    p.write_bytes(_make_gguf(["blk.0.attn_q.weight", "blk.0.attn_k.weight", "output.weight"]))
    names = read_tensor_names(p)
    assert names == ["blk.0.attn_q.weight", "blk.0.attn_k.weight", "output.weight"]


def test_reader_skips_metadata_to_reach_tensors(tmp_path):
    # One string KV, one u32 KV, one array-of-string KV. The reader must skip
    # all three correctly or it won't find the tensor names.
    meta = (
        _s("general.name") + struct.pack("<I", 8) + _s("demo-model") +
        _s("block_count") + struct.pack("<I", 4) + struct.pack("<I", 64) +
        _s("tokenizer.tokens") + struct.pack("<I", 9)
        + struct.pack("<I", 8) + struct.pack("<Q", 2) + _s("a") + _s("b")
    )
    p = tmp_path / "m.gguf"
    p.write_bytes(_make_gguf(["blk.0.attn_q.weight", "blk.64.nextn.embed_tokens.weight"], metadata=meta, kv_count=3))
    head = read_header(p)
    assert head["tensor_count"] == 2
    assert "blk.64.nextn.embed_tokens.weight" in head["tensor_names"]


def test_has_speculative_heads():
    present, n = has_speculative_heads(["blk.0.attn_q.weight", "blk.64.nextn.embed.weight", "blk.64.nextn.norm.weight"])
    assert present is True and n == 2

    present, n = has_speculative_heads(["blk.0.attn_q.weight", "output.weight"])
    assert present is False and n == 0

    present, n = has_speculative_heads(["model.mtp.head.weight"])
    assert present is True and n == 1


def test_bad_magic_raises(tmp_path):
    p = tmp_path / "bad.gguf"
    p.write_bytes(b"NOPE" + b"\x00" * 32)
    with pytest.raises(GgufError):
        read_header(p)
