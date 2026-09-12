# nano/ - the hero seat

~11M-parameter transformer (depth 6) that plays Word Hunt tile by tile from the 16 letters and its current path.
No dictionary at inference; words live in the weights. Ported from actionfleet `sandbox/nanoagent`
(`docs/nanoagent-handoff.md`). Train recipe: `nano/data.py` -> `nano/train.py` (`make check-mps` first);
gate: `python -m nano.gate --model nano/checkpoints/d6_lambda.pt` -> `nano/results/`.

**Default checkpoint: `nano/checkpoints/d6_lambda.pt`** (trained on a Lambda A100, see "GPU training"). `d6_s0.pt` is the
Mac-trained first hero, kept as the fallback; same architecture, same interface, same 1.7-1.9 ms/action.

## Plugging the seat into `wordhunt/seats/` (game worker)

1. `pip install torch numpy` (CPU wheels are enough; ~1 ms/action single-thread) and `COPY nano ./nano` in the Dockerfile so `nano/checkpoints/*.pt` ships.
2. `from nano.seat import NanoSeat; seat = NanoSeat("nano/checkpoints/d6_lambda.pt", temperature=1.0, seed=<seat seed>)` once per seat (loads the checkpoint, ~45 MB fp32).
3. On round start: `seat.reset(board)`; ignore the solver `words` argument (the nano must not see it).
4. Per hand tick: `a = seat.next_action(board, path, found)` -> `("extend", tile)` / `("submit",)` / `("abort",)`; map to your `Action.extend(tile)` / `SUBMIT` / `ABORT`. `seat.step()` does the same but keeps its own path copy, for callers without a hand.
5. After the room judges a submit: `seat.on_result(word, ok, reason)` so it stops re-tracing accepted words. Two nanos with different seeds diverge (temperature sampling); `seat.last_info["value"]` is the escalation signal for the stretch Gemma hand-off.

## Live learning between rounds (`nano/learn.py`)

```python
from nano.learn import OnlineLearner
learner = OnlineLearner("nano/checkpoints/d6_lambda.pt")      # ~10 s to load: solver + 400 replay boards + held-out set
res = learner.update(board, validated_words)                     # any seat's accepted words for that board
# {"kept": bool, "held_out_before": 7160, "held_out_after": 7145, "seconds": 9.5, "n_samples": ..., "loss_first": ..., "loss_last": ...}
seat = learner.seat                                              # NanoSeat on the current (kept) weights; or learner.save(path)
```

`update` runs 50 BC steps on CPU (~9-10 s total, measured in `nano/results/live_learning_curve.md`) on the round's words
converted to the same soft targets as `data.py`, mixed 1:1 with replay from the base distribution and blended with the
model's own predictions (hinted self-distillation), then replays a fixed 20-board held-out set and rolls back if the
score drops more than 3%. Call it during the 20 s rematch countdown, off the event loop (`asyncio.to_thread`). Pass
`replay="data/wh_200000.npz"` if that file exists; otherwise it generates 400 replay boards at construction.

## Lab notebook (`nano/lab_notebook.ipynb`, rendered `nano/lab_notebook.html`)

One executable record of the model: architecture (token layout, heads, `[CLS]` attention), data and soft labels, the
recorded `d6_s0` training run plus a live CPU smoke run, inference latency, sample trajectories (traced paths, value head
along each episode, two seeds diverging), the gate and league numbers, and the live-learning curve with three live rounds.
It documents the Mac-trained `d6_s0.pt` (the fallback seat); the recorded artifacts it reads are
`nano/results/{train,gate}_d6_s0.json` and `live_learning_curve.md`. Needs `matplotlib nbformat nbconvert nbclient ipykernel`
in the venv (notebook-only, not in `requirements.txt`); ~90 s on CPU from the repo root:

```
.venv/bin/python -m jupyter nbconvert --execute --to html --output lab_notebook.html nano/lab_notebook.ipynb
```

## GPU training (Lambda, hackathon requirement)

Instance: Lambda Cloud `gpu_1x_a100_sxm4` (1x A100-SXM4-40GB, 30 vCPU, us-west-2, $1.99/h), Lambda Stack torch 2.7 +
CUDA. Launched 13:40 PT 2026-09-12 via the Lambda MCP tools (`start_instance`, SSH key registered through the REST
API). Driver script: `nano/lambda_train.sh up|log|pull ubuntu@<ip>` (ships `nano/` + word lists, generates data on
the box, launches the trainers with `nohup`).

Data on the box (28 workers): 1M boards -> 25.3M samples in 109 s (`--paths 6`, base recipe) and 1M boards -> 27.4M
samples in 115 s with `--len-bonus 1.6` (word weight x 1.6^(len-3): a 6-letter word counts 4.1x, the curriculum toward
longer words). Four trainers shared the GPU for one 33-minute budget each, bf16 autocast + `torch.compile`
(`--amp --compile`; the fp32 sgemm path was 7k samples/s on the A100 - slower than the Mac's MPS - TF32 made it 15k,
amp+compile 43k for one d6 alone; the four together ran at ~8.5k (d6) / ~5.4k (d8) samples/s each):

```
python3 -m nano.train --data data/wh_len16.npz --depth 6 --steps 400000 --budget-min 33 --batch-size 1024 \
    --val-every 2000 --amp --compile --out runs/c_d6_len16        # -> nano/checkpoints/d6_lambda.pt
```

| run | data | depth / params | steps x batch | val target_top1 | league score (20 boards) | mean len | ms/action CPU |
|---|---|---|---:|---:|---:|---:|---:|
| d6_s0 (Mac MPS, 30 min) | 200k boards | 6 / 11.0M | 17.3k x 256 | 0.767 | 11245 | 3.44 | 1.8 |
| a_d6_base (A100) | 1M boards | 6 / 11.0M | 16.4k x 1024 | 0.823 | 15180 | 3.56 | 1.7 |
| b_d8_base (A100) | 1M boards | 8 / 25.8M | 10.5k x 1024 | 0.817 | 14780 | 3.57 | 3.5 |
| **c_d6_len16 (A100) = `d6_lambda.pt`** | 1M boards, len-bonus 1.6 | 6 / 11.0M | 16.3k x 1024 | 0.800 | **16550** | **3.67** | 1.7 |
| d_d8_len16 (A100) | 1M boards, len-bonus 1.6 | 8 / 25.8M | 10.5k x 1024 | 0.784 | 16730 | 3.68 | 3.5 |

Reading: 5x the samples on the same 11M architecture is worth +35% league score (a); the length curriculum adds
another +9% and +0.11 mean word length (c); the wider depth-8 model buys nothing the d6 does not at 2x the CPU cost
(b, d), so it is not shipped. `d6_lambda` on the 50-board gate: **13586 vs random 952 (14.3x), valid 0.991, mean len
3.65, longest `linters`** (`nano/results/gate_d6_lambda.md`; d6_s0 was 9126 / 9.6x / 0.971 / 3.44). The default seat
switched to `d6_lambda.pt`; `d6_s0.pt` stays as fallback.

GPU-minutes and cost: 4 runs x 33 min = 132 GPU-run-minutes on one A100 wall-clock hour of training (plus ~10 min of a
discarded fp32 launch and data generation). Instance time 13:40 PT -> kept running for the Gemma vLLM seat after training;
training-phase cost ~1.1 h x $1.99 = **$2.2** of the $50 Lambda budget (cumulative spend is reported in the handoff).
