# Gemma 4 12B (mlx-vlm 4-bit) Word Hunt eval, n=20, seed 0

| metric | 12B text | 12B image |
|---|---:|---:|
| valid-word rate | 20% | 22% |
| mean word length (valid) | 3.79 | 3.57 |
| words / call (returned) | 25.5 | 20.4 |
| valid words / call | 5.1 | 4.4 |
| mean score / board | 1780 | 1200 |
| latency / call, mean | 2.1s | 2.2s |
| latency / call, p50 | 1.8s | 1.6s |
| latency / call, max | 5.4s | 10.3s |
| invalid: not a word | 44 | 32 |
| invalid: not on board | 364 | 287 |
| errors / timeouts | 0 | 0 |

n=20 boards, seed=0, model `mlx-community/gemma-4-12B-it-4bit` via mlx-vlm, thinking off, T=0.2, max_tokens=400, timeout 30.0s
