"""merge-kit: rank-concat LoRA merge + GGUF speculative-head verification."""

from merge_kit.lora_merge import rank_concat
from merge_kit.gguf import read_tensor_names, has_speculative_heads, SPECULATIVE_PATTERNS

__version__ = "0.1.0"
__all__ = ["rank_concat", "read_tensor_names", "has_speculative_heads", "SPECULATIVE_PATTERNS"]
