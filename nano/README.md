# nano/ - the hero seat

~11M-parameter transformer (depth 6) that plays Word Hunt tile by tile from the 16 letters and its current path.
No dictionary at inference; words live in the weights. Ported from actionfleet `sandbox/nanoagent`
(`docs/nanoagent-handoff.md`). Train recipe: `nano/data.py` -> `nano/train.py` (`make check-mps` first);
gate: `python -m nano.gate --model nano/checkpoints/d6_s0.pt` -> `nano/results/`.

## Plugging the seat into `wordhunt/seats/` (game worker)

1. `pip install torch numpy` (CPU wheels are enough; ~1 ms/action single-thread) and `COPY nano ./nano` in the Dockerfile so `nano/checkpoints/d6_s0.pt` ships.
2. `from nano.seat import NanoSeat; seat = NanoSeat("nano/checkpoints/d6_s0.pt", temperature=1.0, seed=<seat seed>)` once per seat (loads the checkpoint, ~45 MB fp32).
3. On round start: `seat.reset(board)`; ignore the solver `words` argument (the nano must not see it).
4. Per hand tick: `a = seat.next_action(board, path, found)` -> `("extend", tile)` / `("submit",)` / `("abort",)`; map to your `Action.extend(tile)` / `SUBMIT` / `ABORT`. `seat.step()` does the same but keeps its own path copy, for callers without a hand.
5. After the room judges a submit: `seat.on_result(word, ok, reason)` so it stops re-tracing accepted words. Two nanos with different seeds diverge (temperature sampling); `seat.last_info["value"]` is the escalation signal for the stretch Gemma hand-off.
