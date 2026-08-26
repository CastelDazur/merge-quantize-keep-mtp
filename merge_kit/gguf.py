"""A tiny, dependency-free GGUF header reader.

Just enough of the format to list tensor names and check for speculative-decoding
(MTP) heads after quantization. It parses the metadata and the tensor directory,
never the tensor data, so it runs in milliseconds on a 20GB file.

GGUF layout (v2/v3), little-endian:
    "GGUF"  version:u32  tensor_count:u64  metadata_kv_count:u64
    metadata_kv * count
    tensor_info * tensor_count
Each metadata value is typed; arrays are typed and counted. A tensor_info is
name:string  n_dims:u32  dims:u64[n_dims]  type:u32  offset:u64.
"""

from __future__ import annotations

import struct
from pathlib import Path


SPECULATIVE_PATTERNS = ("nextn", "mtp")

_MAGIC = b"GGUF"

# GGUF metadata value type ids.
_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL, _STRING, _ARRAY, _U64, _I64, _F64 = range(13)

# Fixed-size scalar type -> struct format.
_SCALAR = {
    _U8: "<B", _I8: "<b", _U16: "<H", _I16: "<h", _U32: "<I", _I32: "<i",
    _F32: "<f", _BOOL: "<?", _U64: "<Q", _I64: "<q", _F64: "<d",
}
_SCALAR_SIZE = {t: struct.calcsize(f) for t, f in _SCALAR.items()}


class GgufError(RuntimeError):
    pass


class _Reader:
    def __init__(self, f):
        self.f = f

    def _read(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise GgufError("unexpected end of file reading header")
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self._read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self._read(8))[0]

    def string(self) -> str:
        n = self.u64()
        return self._read(n).decode("utf-8", errors="replace")

    def skip_value(self, vtype: int) -> None:
        """Advance past one metadata value of the given type without keeping it."""
        if vtype in _SCALAR_SIZE:
            self._read(_SCALAR_SIZE[vtype])
        elif vtype == _STRING:
            self.string()
        elif vtype == _ARRAY:
            elem_type = self.u32()
            count = self.u64()
            for _ in range(count):
                self.skip_value(elem_type)
        else:
            raise GgufError(f"unknown metadata value type {vtype}")


def read_header(path: str | Path) -> dict:
    """Return {'version', 'tensor_count', 'tensor_names'} from a GGUF file."""
    with open(path, "rb") as f:
        r = _Reader(f)
        if r._read(4) != _MAGIC:
            raise GgufError("not a GGUF file (bad magic)")
        version = r.u32()
        tensor_count = r.u64()
        kv_count = r.u64()

        # Metadata: read key + type, skip the value. We only need to advance
        # past it to reach the tensor directory.
        for _ in range(kv_count):
            r.string()                 # key
            vtype = r.u32()
            r.skip_value(vtype)

        names = []
        for _ in range(tensor_count):
            name = r.string()
            n_dims = r.u32()
            for _ in range(n_dims):
                r.u64()                # dim
            r.u32()                    # ggml type
            r.u64()                    # offset
            names.append(name)

    return {"version": version, "tensor_count": tensor_count, "tensor_names": names}


def read_tensor_names(path: str | Path) -> list[str]:
    """Just the tensor names."""
    return read_header(path)["tensor_names"]


def has_speculative_heads(
    names: list[str],
    patterns: tuple[str, ...] = SPECULATIVE_PATTERNS,
) -> tuple[bool, int]:
    """(present, count) of tensors whose name contains a speculative-head pattern."""
    hits = [n for n in names if any(p in n.lower() for p in patterns)]
    return (len(hits) > 0, len(hits))
