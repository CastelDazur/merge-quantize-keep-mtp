# The full pipeline, with commands

Two of these steps are code in this repo (merge, verify). The rest use
llama.cpp's own tools. Paths and model names are placeholders.

## 1. Merge the two adapters (this repo)

Load each adapter's `A`/`B` matrices per target layer and combine them:

```python
import numpy as np
from safetensors.numpy import load_file, save_file
from merge_kit.lora_merge import rank_concat

a1 = load_file("adapter_1/adapter_model.safetensors")
a2 = load_file("adapter_2/adapter_model.safetensors")

merged = {}
for key in a1:
    if key.endswith("lora_A.weight"):
        b_key = key.replace("lora_A", "lora_B")
        A, B = rank_concat(a1[key], a1[b_key], a2[key], a2[b_key])
        merged[key] = A
        merged[b_key] = B

save_file(merged, "adapter_merged/adapter_model.safetensors")
```

Note: if the two adapters came out of nested PEFT wrappers, their tensor keys
can have different prefixes (`base_model.model.model.*` vs
`base_model.model.base_model.model.model.*`). Normalize the keys before pairing
them, or the loop above won't match A to B.

## 2. Weight-merge the combined adapter into the base

Fold the merged adapter into the base weights so inference needs no adapter at
runtime. `peft`'s `merge_and_unload()` does this, or do it by hand per layer:
`W' = W + (alpha / r) * (B @ A)`.

## 3. Assemble one checkpoint

Combine, into a single set of shards:
- the merged text weights from step 2,
- vision tensors (if a multimodal base), copied from the base unchanged,
- the MTP heads (`nextn` / `mtp` tensors) pulled from the speculative-capable
  release of the model.

## 4. Convert to GGUF

```bash
python convert_hf_to_gguf.py ./assembled --outfile model.f16.gguf --outtype f16
```

The MTP heads land as `blk.<n>.nextn.*` in the GGUF.

## 5. Quantize

```bash
llama-quantize model.f16.gguf model.Q5_K_M.gguf Q5_K_M
```

## 6. Verify the heads are in the GGUF (this repo)

```bash
merge-mtp verify model.Q5_K_M.gguf
```

If it reports MISSING, the heads did not make it into the file. Check step 3
(were the MTP tensors included in the assembled checkpoint?) and the conversion,
then verify again. For a downloaded quant, pick a build that includes them.
Don't ship a "speculative" model without heads; it runs, it's just not faster.
