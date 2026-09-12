"""Standalone eval: N random boards x {text grid, screenshot} through the 12B.

    python -m gemma_seat.eval --n 20 --seed 0 [--modality text|image|both] [--json out.json]

Per modality it prints: valid-word rate (valid / returned), mean valid word
length, words per call (returned and valid), latency per call, mean score,
and the invalid breakdown (not a word vs. not traceable on the board).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter

from .boards import grid_text, load_dictionary, packed_board, score_word, validate
from .client import GemmaSeatClient


def run(args: argparse.Namespace) -> dict:
    d = load_dictionary(args.wordlist)
    rng = random.Random(args.seed)
    boards = [packed_board(rng, d, min_words=args.min_words)[0] for _ in range(args.n)]
    modalities = {"both": ["text", "image"], "all": ["text", "image", "index"]}.get(
        args.modality, [args.modality]
    )
    client = GemmaSeatClient(
        base_url=args.base_url,
        model=args.model,
        timeout_s=args.timeout,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        thinking=args.thinking,
        caller="nano-vs-gemma-eval",
        thread=f"seed{args.seed}-n{args.n}-{args.modality}",
        tags={"where": "eval", "seed": args.seed, "n_boards": args.n},
    )
    suffix = "+think" if args.thinking else ""

    rows: list[dict] = []
    for i, board in enumerate(boards):
        for modality in modalities:
            r = client.words(board, modality)
            reasons = Counter()
            valid: list[str] = []
            for w in r.words:
                ok, why = validate(board, w, d)
                reasons[why] += 1
                if ok:
                    valid.append(w)
            row = {
                "board": board,
                "modality": modality + suffix,
                "thinking": args.thinking,
                "latency_s": round(r.latency_s, 3),
                "error": r.error,
                "timed_out": bool(r.error and "Timeout" in r.error) or r.latency_s >= args.timeout,
                "candidates": r.candidates,
                "returned": len(r.words),
                "valid": len(valid),
                "score": sum(score_word(w) for w in valid),
                "valid_words": valid,
                "invalid": {k: v for k, v in reasons.items() if k != "ok"},
                "raw_words": r.words,
            }
            rows.append(row)
            if not args.quiet:
                print(
                    f"[{i + 1:02d}/{args.n}] {row['modality']:10s} {board.upper()} "
                    f"{r.latency_s:5.1f}s cand={r.candidates:2d} ret={len(r.words):2d} valid={len(valid):2d} "
                    f"score={row['score']:5d} inv={row['invalid']}"
                    + (f" ERR={r.error}" if r.error else ""),
                    file=sys.stderr,
                )

    labels = [m + suffix for m in modalities]
    summary = {m: summarize([r for r in rows if r["modality"] == m]) for m in labels}
    return {
        "config": {
            "n": args.n,
            "seed": args.seed,
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "timeout_s": args.timeout,
            "thinking": "on" if args.thinking else "off",
            "ran_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "summary": summary,
        "rows": rows,
    }


def _p95(xs: list[float]) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]


def summarize(rows: list[dict]) -> dict:
    returned = sum(r["returned"] for r in rows)
    valid = sum(r["valid"] for r in rows)
    lens = [len(w) for r in rows for w in r["valid_words"]]
    inv = Counter()
    for r in rows:
        inv.update(r["invalid"])
    lat = [r["latency_s"] for r in rows]
    return {
        "calls": len(rows),
        "errors": sum(1 for r in rows if r["error"]),
        "timeouts": sum(1 for r in rows if r.get("timed_out")),
        "candidates_per_call": sum(r.get("candidates", r["returned"]) for r in rows) / len(rows) if rows else 0.0,
        "valid_word_rate": (valid / returned) if returned else 0.0,
        "mean_word_len": statistics.fmean(lens) if lens else 0.0,
        "words_per_call": returned / len(rows) if rows else 0.0,
        "valid_words_per_call": valid / len(rows) if rows else 0.0,
        "mean_score": statistics.fmean(r["score"] for r in rows) if rows else 0.0,
        "latency_mean_s": statistics.fmean(lat) if lat else 0.0,
        "latency_p50_s": statistics.median(lat) if lat else 0.0,
        "latency_p95_s": _p95(lat),
        "latency_max_s": max(lat) if lat else 0.0,
        "invalid_not_word": inv.get("not_word", 0),
        "invalid_not_on_board": inv.get("not_on_board", 0),
    }


TABLE_SPEC = [
    ("valid-word rate", "valid_word_rate", "{:.0%}"),
    ("mean word length (valid)", "mean_word_len", "{:.2f}"),
    ("words / call (model emitted)", "candidates_per_call", "{:.1f}"),
    ("words / call (after client filter)", "words_per_call", "{:.1f}"),
    ("valid words / call", "valid_words_per_call", "{:.1f}"),
    ("mean score / board", "mean_score", "{:.0f}"),
    ("latency / call, mean", "latency_mean_s", "{:.1f}s"),
    ("latency / call, p50", "latency_p50_s", "{:.1f}s"),
    ("latency / call, p95", "latency_p95_s", "{:.1f}s"),
    ("latency / call, max", "latency_max_s", "{:.1f}s"),
    ("invalid: not a word", "invalid_not_word", "{}"),
    ("invalid: not on board", "invalid_not_on_board", "{}"),
    ("timeouts (hit hard cap)", "timeouts", "{}"),
    ("errors", "errors", "{}"),
]


def table(result: dict) -> str:
    return merged_table([result])


def merged_table(results: list[dict]) -> str:
    """One table across several eval runs (same n/seed, different modes)."""
    cols: list[tuple[str, dict]] = []
    for res in results:
        for m, s in res["summary"].items():
            cols.append((m, s))
    head = "| metric | " + " | ".join(f"12B {m}" for m, _ in cols) + " |"
    sep = "|---|" + "---:|" * len(cols)
    lines = [head, sep]
    for label, key, fmt in TABLE_SPEC:
        cells = []
        for _, s in cols:
            v = s.get(key)
            cells.append(fmt.format(v) if v is not None else "-")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    for res in results:
        c = res["config"]
        lines.append(
            f"- {', '.join(res['summary'])}: n={c['n']} boards, seed={c['seed']}, `{c['model']}` via mlx-vlm, "
            f"thinking {c['thinking']}, T={c['temperature']}, max_tokens={c['max_tokens']}, "
            f"timeout {c['timeout_s']}s"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--modality", choices=["text", "image", "index", "both", "all"], default="both")
    p.add_argument("--thinking", action="store_true", help="enable_thinking=true (bump --max-tokens)")
    p.add_argument("--merge", nargs="*", default=None, metavar="JSON",
                   help="skip running; print one table merging these result JSON files")
    p.add_argument("--min-words", type=int, default=40, help="board packing threshold")
    p.add_argument("--wordlist", default=None)
    p.add_argument("--base-url", default=GemmaSeatClient.__init__.__defaults__[0])
    p.add_argument("--model", default=GemmaSeatClient.__init__.__defaults__[1])
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--json", default=None, help="write full results here")
    p.add_argument("--show-boards", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    if args.merge is not None:
        results = [json.load(open(f)) for f in args.merge]
        print(merged_table(results))
        return 0
    if args.wordlist is None:
        from .boards import DEFAULT_WORDLIST

        args.wordlist = DEFAULT_WORDLIST

    result = run(args)
    if args.show_boards:
        for r in result["rows"]:
            print(grid_text(r["board"]), r["modality"], r["valid_words"], "\n")
    print(table(result))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(result, fh, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
