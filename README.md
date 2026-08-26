# merge-quantize-keep-mtp

Merge two LoRA adapters into one model, quantize it to GGUF, and verify the speculative-decoding heads survived the quantization.

Two things here that cost me time and aren't written down much:

1. Merging two adapters trained separately. The naive answer is to add them, and that's wrong when they were trained independently. Rank-concatenation is the right move and it's a one-liner once you see it.
2. Quantizing a model that has multi-token-prediction (MTP) heads for speculative decoding. Some quant levels silently drop those heads, and you don't find out until draft speculation does nothing at inference. This repo has a GGUF reader that checks, so you catch it at build time instead of wondering why your tokens/sec didn't move.

Everything is model-family agnostic. The MTP checks look for the tensor names Qwen3-family and DeepSeek-style speculative heads use; adjust the patterns for yours.

This came out of moving a local 27B setup from one Qwen generation to the next. Two adapters trained on the old base had to be carried onto the new one and folded into a single model, with the speculative-decoding heads kept intact through quantization. The steps are generic; only the motivation was mine.

## Why rank-concatenation, not addition

You have two adapters, `A1/B1` and `A2/B2`, each rank `r`, trained separately on the same base. You want one adapter that applies both.

Adding them (`A1+A2`, `B1+B2`) is wrong. It only works if the adapters share a subspace, which independently trained ones don't. What you actually want is the sum of their *effects*: `B1·A1·x + B2·A2·x`.

Concatenate along the rank dimension and you get exactly that, for free:

```
A = concat([A1, A2], axis=0)     # (2r, in)
B = concat([B1, B2], axis=1)     # (out, 2r)
B · A  ==  B1·A1 + B2·A2          # provably, no approximation
```

The merged adapter has rank `2r`. `merge_kit/lora_merge.py` does this and asserts the identity holds.

```python
from merge_kit.lora_merge import rank_concat
A, B = rank_concat(A1, B1, A2, B2)   # raises if shapes don't line up
```

## Keeping MTP heads through quantization

MTP heads (the `nextn` / `mtp` tensors) are what let a model draft several tokens ahead and verify them in one pass. They're small, and some quantization paths drop or zero them, so the model still loads and runs, just without the speedup you built it for.

The fix is not clever: check. After you quantize, read the GGUF and confirm the speculative tensors are still there.

```bash
merge-mtp verify model.Q5_K_M.gguf
```
```
model.Q5_K_M.gguf
  866 tensors
  speculative heads: FOUND (5 nextn tensors)
  OK
```

If they're gone:
```
  speculative heads: MISSING
  Q5 dropped the nextn tensors. Re-quantize at Q4_K_M, or keep nextn in higher
  precision. This model will load and run but draft speculation will do nothing.
```

`merge_kit/gguf.py` is a small dependency-free GGUF header reader. It parses the metadata and tensor list without loading weights, so `verify` runs in milliseconds on a 20GB file. You can also use it on its own to list what's in any GGUF:

```python
from merge_kit.gguf import read_tensor_names, has_speculative_heads
names = read_tensor_names("model.gguf")
print(has_speculative_heads(names))    # True / False
```

## The full pipeline

The steps, in order. Only the merge and the verify are in this repo as code; the assemble/convert/quantize steps use llama.cpp's own tools, and `examples/pipeline.md` has the exact commands.

1. **Merge** the two LoRA adapters with `rank_concat` (this repo).
2. **Weight-merge** the combined adapter into the base weights.
3. **Assemble** one checkpoint: merged text weights, plus any vision tensors, plus the MTP heads pulled from the speculative-capable release.
4. **Convert** to GGUF (`convert_hf_to_gguf.py` from llama.cpp).
5. **Quantize** (`llama-quantize` to Q5_K_M or whatever fits).
6. **Verify** the MTP heads survived with `merge-mtp verify` (this repo). If missing, drop to Q4_K_M.

## Install

```bash
pip install -e .
```

Only dependency is numpy, for the merge math. The GGUF reader is pure standard library.

## What this is not

- Not a full GGUF library. The reader parses the header (metadata + tensor list); it doesn't read tensor data. That's all `verify` needs.
- Not a trainer or a quantizer. It's the merge step and the check around llama.cpp's quantizer.
- The MTP tensor-name patterns are for Qwen3-family / DeepSeek-style heads. Other architectures name them differently; edit `SPECULATIVE_PATTERNS` in `gguf.py`.

## License

MIT.

From building [CastelOS](https://github.com/CastelDazur/castelos-public). Part of a small set of local-AI tooling alongside [qlora-single-gpu-playbook](https://github.com/CastelDazur/qlora-single-gpu-playbook).
