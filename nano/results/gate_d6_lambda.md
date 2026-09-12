# Nano gate: `nano/checkpoints/d6_lambda.pt` (PASS)

50 unseen boards (seed 20000, >= 15 words each), 600 actions per board (75 s at ~8 Hz), temperature 1.0, CPU inference. Pre-registered in PLAN.md section 3: score >= 4x random swiper AND valid-submit rate above the Gemma E4B seat (proxy: Gemma 4 12B (mlx-vlm 4-bit, text grid, n=20), `gemma_seat/results/eval_n20_seed0.md`).

| metric | nano | random swiper | Gemma proxy | gate |
|---|---:|---:|---:|---|
| score / board | **13586** | 952 | 1780 | 14.27x random (>= 4x) PASS |
| valid-submit rate | **0.991** | 0.048 | 0.20 | > 0.20 PASS |
| words found / board | 42.5 | 4.8 | 5.1 (per call) | |
| mean word length | 3.65 | 3.28 | 3.79 | |
| submits / board | 112.4 | 108.6 | | |
| aborts / board | 20.1 | 0.0 | | |
| longest word | `linters` | `dialer` | | |
| ms / action (CPU, 1 thread) | 1.704 | | ~2000 per call | |
| params | 10.99M | 0 | 12B | |

Word-length histogram (nano, all boards): {3: 1013, 4: 854, 5: 206, 6: 46, 7: 4, 8: 0}. Longest: arsenal, linters, rentals, rockets, acting, became, bucked, career, carses, carter.

Training: depth 6, 16316 steps x batch 1024 = 16707584 samples on 26819862 (33.0 min, 8438 samples/s, device cuda); loss 2.675 -> 0.872; val {"loss": 0.888, "type": 0.135, "target": 0.731, "value": 0.044, "type_top1": 0.965, "target_top1": 0.798, "value_acc": 0.984}.

Rollout wall: 1.3 s/board for the nano. Stretch target (>= 60% of human median score) not measured: no human boards yet.

Example board `mtappdaanseitosa`: nano found ['passed', 'season', 'aeon', 'dais', 'data', 'east', 'nose', 'ossa', 'pads', 'pass', 'past', 'seat', 'tads', 'toes', 'toss'] (39/89); random found ['past'].
