# Nano gate: `nano/checkpoints/d6_s0.pt` (PASS)

50 unseen boards (seed 20000, >= 15 words each), 600 actions per board (75 s at ~8 Hz), temperature 1.0, CPU inference. Pre-registered in PLAN.md section 3: score >= 4x random swiper AND valid-submit rate above the Gemma E4B seat (proxy: Gemma 4 12B (mlx-vlm 4-bit, text grid, n=20), `gemma_seat/results/eval_n20_seed0.md`).

| metric | nano | random swiper | Gemma proxy | gate |
|---|---:|---:|---:|---|
| score / board | **9126** | 952 | 1780 | 9.59x random (>= 4x) PASS |
| valid-submit rate | **0.971** | 0.048 | 0.20 | > 0.20 PASS |
| words found / board | 38.0 | 4.8 | 5.1 (per call) | |
| mean word length | 3.44 | 3.28 | 3.79 | |
| submits / board | 105.9 | 108.6 | | |
| aborts / board | 30.9 | 0.0 | | |
| longest word | `stokes` | `dialer` | | |
| ms / action (CPU, 1 thread) | 1.906 | | ~2000 per call | |
| params | 10.99M | 0 | 12B | |

Word-length histogram (nano, all boards): {3: 1139, 4: 676, 5: 83, 6: 4, 7: 0, 8: 0}. Longest: fonder, irones, season, stokes, acnes, acres, alans, areas, argue, atlas.

Training: depth 6, 17278 steps x batch 256 = 4423168 samples on 4955598 (30.0 min, 2457 samples/s, device mps); loss 2.611 -> 1.069; val {"loss": 1.073, "type": 0.203, "target": 0.823, "value": 0.095, "type_top1": 0.935, "target_top1": 0.767, "value_acc": 0.964}.

Rollout wall: 1.4 s/board for the nano. Stretch target (>= 60% of human median score) not measured: no human boards yet.

Example board `mtappdaanseitosa`: nano found ['data', 'east', 'pads', 'pass', 'seas', 'seat', 'sons', 'toes', 'tons', 'toss', 'aas', 'ads', 'ais', 'asp', 'dap'] (32/89); random found ['past'].
