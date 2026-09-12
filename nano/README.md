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

## Live learning between rounds (`nano/learn.py`)

```python
from nano.learn import OnlineLearner
learner = OnlineLearner("nano/checkpoints/d6_s0.pt")          # ~10 s to load: solver + 400 replay boards + held-out set
res = learner.update(board, validated_words)                     # any seat's accepted words for that board
# {"kept": bool, "held_out_before": 7160, "held_out_after": 7145, "seconds": 9.5, "n_samples": ..., "loss_first": ..., "loss_last": ...}
seat = learner.seat                                              # NanoSeat on the current (kept) weights; or learner.save(path)
```

`update` runs 50 BC steps on CPU (~9-10 s total, measured in `nano/results/live_learning_curve.md`) on the round's words
converted to the same soft targets as `data.py`, mixed 1:1 with replay from the base distribution and blended with the
model's own predictions (hinted self-distillation), then replays a fixed 20-board held-out set and rolls back if the
score drops more than 3%. Call it during the 20 s rematch countdown, off the event loop (`asyncio.to_thread`). Pass
`replay="data/wh_200000.npz"` if that file exists; otherwise it generates 400 replay boards at construction.
