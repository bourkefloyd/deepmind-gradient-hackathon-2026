#!/usr/bin/env python3
"""Quick check of data/levels.json: per-level table + a rough "casual human" score estimate.

    python scripts/levels_report.py                    # print the table
    python scripts/levels_report.py --md docs/levels.md   # also write the markdown page

Stats are recomputed with the game's solver (enable1 + top 10k of common-30k), so this also
validates hand-edited boards. Exit code 1 if any level misses the easy-target bar
(>= 40 common 3-4 letter words, >= 3 common 5+ letter words, all seed words findable).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from wordhunt.levels import COMMON_RANK, LEVELS_PATH, board_stats, load_levels  # noqa: E402
from wordhunt.solver import Solver  # noqa: E402

MIN_COMMON_34 = 40
MIN_COMMON_5 = 3


def rows(levels, solver: Solver) -> list[dict]:
    out = []
    for lvl in levels:
        st = board_stats(lvl.board, solver, seed_words=lvl.seed_words)
        present = st["seeds_present"]
        missing = [w for w in lvl.seed_words if w not in present]
        ok = st["common_3"] + st["common_4"] >= MIN_COMMON_34 and st["common_5plus"] >= MIN_COMMON_5 and not missing
        out.append({"n": lvl.n, "theme": lvl.theme, "board": lvl.board.upper(), **st, "missing": missing, "ok": ok})
    return out


def table(rs: list[dict], md: bool) -> str:
    head = ["#", "Theme", "Board", "Words", "Common", "3 / 4 / 5+", "Max", "Longest", "Theme words", "Casual ~"]
    body = []
    for r in rs:
        seeds = " ".join(w.upper() for w in r["seeds_present"])
        if r["missing"]:
            seeds += " (missing: " + " ".join(w.upper() for w in r["missing"]) + ")"
        body.append([str(r["n"]), r["theme"], r["board"], str(r["n_words"]), str(r["common_words"]),
                     f"{r['common_3']} / {r['common_4']} / {r['common_5plus']}", f"{r['max_score']:,}",
                     r["longest"].upper(), seeds, f"{r['human_target']:,}" + ("" if r["ok"] else " ⚠")])
    if md:
        lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
        lines += ["| " + " | ".join(row) + " |" for row in body]
        return "\n".join(lines)
    widths = [max(len(x) for x in col) for col in zip(head, *body)]
    fmt = "  ".join("{:<%d}" % w for w in widths)
    return "\n".join([fmt.format(*head)] + [fmt.format(*row) for row in body])


def markdown(rs: list[dict], path: str) -> str:
    n_ok = sum(1 for r in rs if r["ok"])
    return f"""# Themed levels

{len(rs)} pre-canned boards in `{os.path.relpath(path)}`, played in order (round 1 = level 1, each rematch
moves to the next, wrapping after the last). Each board is seeded with 3-5 everyday words for its
theme and then filled to maximise *common* word density, so a casual player scores 1,000-2,000 in a
75 s round without effort.

Regenerate this page with `python scripts/levels_report.py --md docs/levels.md`.

## Levels

{table(rs, md=True)}

**Columns.** *Words* = every valid enable1 word (3-8 letters) on the board; *Common* = those in the
top {COMMON_RANK:,} of `data/common-30k.txt`, split by length (*3 / 4 / 5+*); *Max* = score if every word
were found; *Theme words* = the seed words, all verified findable on the board; *Casual ~* = the
rough human estimate below. ⚠ marks a level under the bar (< {MIN_COMMON_34} common 3-4 letter words, < {MIN_COMMON_5}
common 5+ letter words, or a seed word that is not traceable). {n_ok}/{len(rs)} levels pass.

## Casual human estimate

A relaxed player finds roughly one word every 6 s, so ~12 words in 75 s, mostly 3-letter words
with a few 4s: 12 finds at a 3:1 mix is 9 x 100 + 3 x 400 = 2,100; all 3s is 1,200. The estimate
takes 30% of the board's common 3-4 letter words (capped at 12 finds), weights 4-letter words at a
third of the 3-letter rate, and rounds to the nearest 100. Common 5+ letter words (`5+` column) are
upside on top: one `PLATE` is +800.

## Curating

- Edit `data/levels.json`: change a `theme` title, reorder entries, swap in a hand-made 16-letter
  `board` (row-major, lowercase), or delete a level. `n` and `stats` are informational; the game
  reads `theme`, `board` and file order.
- Re-check with `python scripts/levels_report.py` (exit code 1 if a level misses the bar), then
  redeploy (`scripts/deploy.sh <label>`) — the JSON is copied into the image.
- Regenerate a single theme: `python scripts/make_levels.py --only Kitchen`; all themes with a new
  search seed: `python scripts/make_levels.py --seed 7`.
- Runtime knobs: `WH_LEVELS=data/levels.json` (default; set empty or `0` for the old packed/random
  boards), `WH_LEVEL_START=n` to start a room at level n.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", default=LEVELS_PATH)
    ap.add_argument("--md", default="", help="write the markdown page here (e.g. docs/levels.md)")
    a = ap.parse_args()

    levels = load_levels(a.levels)
    if not levels:
        print(f"no levels in {a.levels}", file=sys.stderr)
        return 2
    rs = rows(levels, Solver())
    print(table(rs, md=False))
    bad = [r for r in rs if not r["ok"]]
    print(f"\n{len(rs) - len(bad)}/{len(rs)} levels pass (>= {MIN_COMMON_34} common 3-4 letter words, "
          f">= {MIN_COMMON_5} common 5+ letter words, all theme words findable)")
    if a.md:
        with open(a.md, "w", encoding="utf-8") as f:
            f.write(markdown(rs, a.levels))
        print(f"wrote {a.md}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
