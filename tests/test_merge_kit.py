"""Offline tests. The GGUF tests build a small synthetic file so the reader is
exercised for real instead of being taken on trust."""

import struct

import numpy as np
import pytest

from merge_kit.lora_merge import rank_concat, delta
from merge_kit.gguf import (read_header, read_tensor_names, has_speculative_heads,
                            GgufError)


# --- the merge identity: this is the whole point of the repo ---

def test_rank_concat_equals_sum_of_deltas():
    rng = np.random.default_rng(0)
    in_dim, out_dim, r = 8, 5, 3
    A1, B1 = rng.normal(size=(r, in_dim)), rng.normal(size=(out_dim, r))
    A2, B2 = rng.normal(size=(r, in_dim)), rng.normal(size=(out_dim, r))

    A, B = rank_concat(A1, B1, A2, B2)

    assert A.shape == (2 * r, in_dim)
    assert B.shape == (out_dim, 2 * r)
    np.testing.assert_allclose(delta(A, B), delta(A1, B1) + delta(A2, B2), atol=1e-12)


def test_rank_concat_handles_different_ranks():
    rng = np.random.default_rng(1)
    A1, B1 = rng.normal(size=(2, 6)), rng.normal(size=(4, 2))
    A2, B2 = rng.normal(size=(5, 6)), rng.normal(size=(4, 5))
    A, B = rank_concat(A1, B1, A2, B2)
    assert A.shape == (7, 6) and B.shape == (4, 7)
    np.testing.assert_allclose(delta(A, B), delta(A1, B1) + delta(A2, B2), atol=1e-12)


def test_adding_adapters_is_not_the_same_thing():
    """Guards the claim in the README: elementwise addition gives a different model."""
    rng = np.random.default_rng(2)
    A1, B1 = rng.normal(size=(3, 6)), rng.normal(size=(4, 3))
    A2, B2 = rng.normal(size=(3, 6)), rng.normal(size=(4, 3))

    correct = delta(A1, B1) + delta(A2, B2)
    naive = delta(A1 + A2, B1 + B2)
    assert not np.allclose(correct, naive)


@pytest.mark.parametrize("bad", ["in", "out", "rank1", "rank2", "dims"])
def test_rank_concat_rejects_mismatched_shapes(bad):
    A1, B1 = np.zeros((3, 6)), np.zeros((4, 3))
    A2, B2 = np.zeros((3, 6)), np.zeros((4, 3))
    if bad == "in":
        A2 = np.zeros((3, 7))
    elif bad == "out":
        B2 = np.zeros((5, 3))
    elif bad == "rank1":
        B1 = np.zeros((4, 2))
    elif bad == "rank2":
        B2 = np.zeros((4, 2))
    elif bad == "dims":
        A1 = np.zeros(3)
    with pytest.raises(ValueError):
        rank_concat(A1, B1, A2, B2)


# --- GGUF reader, against a synthetic file ---

def _write_gguf(path, tensor_names, *, with_metadata=True):
    """Minimal valid GGUF: header, a couple of metadata entries, tensor directory."""
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))                    # version
        f.write(struct.pack("<Q", len(tensor_names)))    # tensor count
        kv = 2 if with_metadata else 0
        f.write(struct.pack("<Q", kv))                   # metadata count

        def wstr(s):
            b = s.encode("utf-8")
            f.write(struct.pack("<Q", len(b)))
            f.write(b)

        if with_metadata:
            # a string value
            wstr("general.architecture")
            f.write(struct.pack("<I", 8))                # STRING
            wstr("qwen")
            # an array of u32, to exercise the array skip path
            wstr("some.array")
            f.write(struct.pack("<I", 9))                # ARRAY
            f.write(struct.pack("<I", 4))                # element type U32
            f.write(struct.pack("<Q", 3))                # count
            for v in (1, 2, 3):
                f.write(struct.pack("<I", v))

        for name in tensor_names:
            wstr(name)
            f.write(struct.pack("<I", 2))                # n_dims
            f.write(struct.pack("<Q", 16))
            f.write(struct.pack("<Q", 32))
            f.write(struct.pack("<I", 0))                # ggml type
            f.write(struct.pack("<Q", 0))                # offset


def test_reads_tensor_names(tmp_path):
    p = tmp_path / "m.gguf"
    _write_gguf(p, ["blk.0.attn_q.weight", "blk.0.attn_k.weight", "output.weight"])
    head = read_header(p)
    assert head["version"] == 3
    assert head["tensor_count"] == 3
    assert head["tensor_names"][0] == "blk.0.attn_q.weight"


def test_skips_metadata_without_it(tmp_path):
    """Same tensors, no metadata block: names must still parse."""
    p = tmp_path / "m.gguf"
    _write_gguf(p, ["a.weight"], with_metadata=False)
    assert read_tensor_names(p) == ["a.weight"]


def test_detects_speculative_heads(tmp_path):
    p = tmp_path / "spec.gguf"
    _write_gguf(p, ["blk.0.attn_q.weight", "blk.64.nextn.eh_proj.weight",
                    "blk.64.nextn.enorm.weight"])
    present, count = has_speculative_heads(read_tensor_names(p))
    assert present is True and count == 2


def test_detects_missing_speculative_heads(tmp_path):
    p = tmp_path / "plain.gguf"
    _write_gguf(p, ["blk.0.attn_q.weight", "output.weight"])
    present, count = has_speculative_heads(read_tensor_names(p))
    assert present is False and count == 0


def test_rejects_non_gguf(tmp_path):
    p = tmp_path / "not.gguf"
    p.write_bytes(b"XXXX" + b"\x00" * 32)
    with pytest.raises(GgufError):
        read_header(p)


def test_truncated_file_raises(tmp_path):
    p = tmp_path / "cut.gguf"
    p.write_bytes(b"GGUF" + struct.pack("<I", 3))      # header cut short
    with pytest.raises(GgufError):
        read_header(p)
