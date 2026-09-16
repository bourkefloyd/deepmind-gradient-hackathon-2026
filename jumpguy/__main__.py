"""python -m jumpguy <play|train|eval|collect|device> ..."""

from __future__ import annotations

import json
import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: python -m jumpguy {play,train,eval,collect,device} ...")
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "play":
        from .play import main as play_main

        return play_main(rest)
    if cmd == "train":
        from .train import main as train_main

        return train_main(rest)
    if cmd == "eval":
        from .eval import main as eval_main

        return eval_main(rest)
    if cmd == "collect":
        from .collect import main as collect_main

        return collect_main(rest)
    if cmd == "device":
        from .device import smoke_test

        info = smoke_test(rest[0] if rest else "auto")
        print(json.dumps(info, indent=2))
        return 0
    raise SystemExit(f"unknown command {cmd!r}")


if __name__ == "__main__":
    raise SystemExit(main())
