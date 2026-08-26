"""Command line.

  merge-mtp verify model.Q5_K_M.gguf   # check speculative heads survived quant
  merge-mtp list   model.gguf          # tensor count + a sample of names
"""

from __future__ import annotations

import argparse
import sys

from merge_kit.gguf import read_header, has_speculative_heads, GgufError


def _cmd_verify(args: argparse.Namespace) -> int:
    try:
        head = read_header(args.gguf)
    except (GgufError, OSError) as exc:
        print(f"{args.gguf}\n  error: {exc}")
        return 2

    names = head["tensor_names"]
    present, count = has_speculative_heads(names)
    print(args.gguf)
    print(f"  {head['tensor_count']} tensors")
    if present:
        print(f"  speculative heads: FOUND ({count} nextn/mtp tensors)")
        print("  OK")
        return 0
    print("  speculative heads: MISSING")
    print("  This quant dropped the nextn/mtp tensors. Re-quantize at a lower")
    print("  level (e.g. Q4_K_M) or keep nextn in higher precision. The model")
    print("  will load and run, but draft speculation will do nothing.")
    return 1


def _cmd_list(args: argparse.Namespace) -> int:
    try:
        head = read_header(args.gguf)
    except (GgufError, OSError) as exc:
        print(f"{args.gguf}\n  error: {exc}")
        return 2
    names = head["tensor_names"]
    print(f"{args.gguf}\n  {head['tensor_count']} tensors (GGUF v{head['version']})")
    for n in names[: args.n]:
        print(f"    {n}")
    if len(names) > args.n:
        print(f"    ... and {len(names) - args.n} more")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="merge-mtp", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="check speculative (MTP) heads are present")
    v.add_argument("gguf")
    v.set_defaults(func=_cmd_verify)

    l = sub.add_parser("list", help="list tensor names")
    l.add_argument("gguf")
    l.add_argument("-n", type=int, default=20, help="how many names to show")
    l.set_defaults(func=_cmd_list)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
