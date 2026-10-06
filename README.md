# merge-quantize-keep-mtp

Merge two LoRA adapters into one model, quantize it to GGUF, and verify the speculative-decoding (MTP) heads are actually in the file.

Two things here that cost me time and aren't written down much:

1. Merging two adapters trained separately. The naive answer is to add them, and that's wrong when they were trained independently. Rank-concatenation is the right move and it's a one-liner once you see it.
2. Making sure a GGUF of a model with multi-token-prediction (MTP) heads still has them. Many published quants of the same model ship without the heads, and a hand-assembled checkpoint can lose them before conversion. Either way the model loads and runs, and you only notice when draft speculation does nothing at inference. This repo has a GGUF reader that checks, so you catch it at build time instead of wondering why your tokens/sec didn't move.

Everything is model-family agnostic. The MTP checks look for the tensor names Qwen3-family and DeepSeek-style speculative heads use; adjust the patterns for yours.

## Where this came from

I run a 27B model locally as an always-on assistant, and I moved it from Qwen 3.6 to Qwen 3.8.

That upgrade is not a download. Two LoRA adapters had been trained separately against the old base, one for a domain and one for reasoning, and both had to end up inside the new model. Not loaded at runtime as adapters, baked into the weights, because the runtime slot takes one file. On top of that the new base ships multi-token-prediction heads for speculative decoding, and those come from a different release than the weights I was fine-tuning, so they had to be assembled in by hand. Then the whole thing gets quantized to fit 32GB of VRAM.

Each of those steps has a way to go quietly wrong. Adding the adapters instead of concatenating them gives you a model that is subtly worse and never tells you why. Picking or building a GGUF without checking can give you a model that loads, runs, and has no speculative heads at all. I nearly shipped one: most of the published Qwen 3.6 27B quants I checked at Q4/Q5/Q6 had no `nextn` tensors in the header, while the build with heads was +39% tokens/sec with MTP drafting on. The code here is the two checks that stop both mistakes.

An honest update: the +39% was on Qwen 3.6. On the current Qwen 3.8 build I did not see a worthwhile gain from MTP drafting for my workload, so my latest quant is built without the heads on purpose. The check is for when you do rely on them: it tells you whether the file you are about to run actually has them.

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

## Checking the MTP heads are in the GGUF

MTP heads (the `nextn` / `mtp` tensors) are what let a model draft several tokens ahead and verify them in one pass. They're small and optional, so a GGUF without them still loads and runs, just without the speedup. Published quants often leave them out, and if you assemble the checkpoint yourself they come from a separate release and are easy to miss.

The fix is not clever: check. Before you rely on a GGUF, after you download it or after you quantize your own, read the header and confirm the speculative tensors are there.

In my own pipeline `llama-quantize` to Q5_K_M kept all four `blk.64.nextn.*` tensors; the quantization step was never the problem. The check is cheap insurance against picking or building the wrong file.

```bash
merge-mtp verify model.Q5_K_M.gguf
```
```
model.Q5_K_M.gguf
  866 tensors
  speculative heads: FOUND (4 nextn/mtp tensors)
  OK
```

If they're gone:
```
  speculative heads: MISSING
  This GGUF has no nextn/mtp tensors. Use a quant that includes them, or check
  that the heads were included when the checkpoint was assembled and converted.
  The model will load and run, but draft speculation will do nothing.
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
6. **Verify** the MTP heads are in the GGUF with `merge-mtp verify` (this repo). If missing, go back to step 3: the heads did not make it into the assembled checkpoint.

## Install

```bash
pip install -e .
```

Only dependency is numpy, for the merge math. The GGUF reader is pure standard library.

## Scope

The GGUF reader parses the header, meaning metadata and the tensor directory. It never touches tensor data, which is why `verify` is instant on a 20GB file, and also why this is not a general GGUF library.

There is no trainer and no quantizer in here. Training is your business and quantizing is llama.cpp's; this fills the two gaps on either side of it.

The tensor-name patterns match Qwen3-family and DeepSeek-style speculative heads. If your architecture names them something else, `verify` will report MISSING on a perfectly good file. Edit `SPECULATIVE_PATTERNS` in `gguf.py` before you trust it.

## License

MIT.

From building [CastelOS](https://github.com/CastelDazur/castelos-public). Part of a small set of local-AI tooling alongside [qlora-single-gpu-playbook](https://github.com/CastelDazur/qlora-single-gpu-playbook).
