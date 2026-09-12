# Gemma 4 12B (mlx-vlm 4-bit) Word Hunt eval, n=20 boards, seed 0: prompt x thinking A/B

Same 20 boards in every column. Default seat = **text, thinking off**.

| metric | 12B text | 12B image | 12B text+think | 12B index |
|---|---:|---:|---:|---:|
| valid-word rate | 20% | 22% | 0% | 100% |
| mean word length (valid) | 3.79 | 3.57 | 0.00 | 3.50 |
| words / call (model emitted) | - | - | 0.0 | 7.0 |
| words / call (after client filter) | 25.5 | 20.4 | 0.0 | 0.1 |
| valid words / call | 5.1 | 4.4 | 0.0 | 0.1 |
| mean score / board | 1780 | 1200 | 0 | 25 |
| latency / call, mean | 2.1s | 2.2s | 30.0s | 15.5s |
| latency / call, p50 | 1.8s | 1.6s | 30.0s | 19.3s |
| latency / call, p95 | - | - | 30.0s | 21.4s |
| latency / call, max | 5.4s | 10.3s | 30.0s | 22.9s |
| invalid: not a word | 44 | 32 | 0 | 0 |
| invalid: not on board | 364 | 287 | 0 | 0 |
| timeouts (hit hard cap) | - | - | 20 | 0 |
| errors | 0 | 0 | 0 | 0 |

- text, image: n=20 boards, seed=0, `mlx-community/gemma-4-12B-it-4bit` via mlx-vlm, thinking off, T=0.2, max_tokens=400, timeout 30.0s
- text+think: n=20 boards, seed=0, `mlx-community/gemma-4-12B-it-4bit` via mlx-vlm, thinking on, T=0.2, max_tokens=2048, timeout 30.0s
- index: n=20 boards, seed=0, `mlx-community/gemma-4-12B-it-4bit` via mlx-vlm, thinking off, T=0.2, max_tokens=400, timeout 30.0s

Notes:

- text+think: every call hit the 30 s hard cap with zero output tokens; an uncapped probe (max_tokens 4096) was still thinking at 180 s (~20 tok/s of tile-by-tile enumeration, no content).
- index (tile-path prompt, thinking off): the 12B reasons in-band instead of listing ('EATEN: 0-1-4-8-9 (Wait, E is 0...)'), so almost no line parses as a legal path; 2 legal words in 20 boards, 7-10x slower than text.
