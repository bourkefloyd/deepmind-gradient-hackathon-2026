# Speculative decoding on the 12B seat (2026-09-12, 20-minute cap)

Question: can speculative decoding make thinking-on affordable for the 75 s race?
Prior A/B (`eval_n20_seed0_ab.md`): thinking-on hit the 30 s cap on 20/20 boards
with zero output; uncapped it was still thinking at 180 s.

## Setup

- `mlx_lm.server --draft-model mlx-community/gemma-4-e4b-it-4bit`: **not possible**.
  mlx-lm 0.31.3 raises `Model type gemma4_unified not supported` for the 12B
  (encoder-free unified arch). Vocab does match (262144 both), so E4B would be a
  valid drafter under a runtime that loads the unified 12B.
- What works: `mlx_vlm.server 0.6.3 --draft-model mlx-community/gemma-4-12B-it-assistant-4bit`
  (Google's official Gemma 4 MTP drafter, 270 MB, auto-detected `--draft-kind mtp`),
  second server on port 8081, the plain 12B on 8080 untouched.

## Measured, same boards (seed 0), text-grid prompt, thinking off, max_tokens 400

| | plain 12B (8080) | 12B + MTP drafter (8081) |
|---|---:|---:|
| tok/s mean, 6 boards | 48.2 | 65.0 (first call 47.9 = warmup; steady 63-72) |
| speedup | 1.0x | **1.35x** mean, ~1.4x steady-state |
| valid words / call | 4.8 | 1.3 (output loops; repetition/presence penalties do not seem to apply on the MTP path) |

## Thinking on + drafter, 1 board, max_tokens 6000, 150 s cap

82.0 s, 6000 tokens at 73.2 tok/s, 11.6k chars of reasoning, **still no answer**
(`finish=length`). Thinking-on needs > 6000 tokens on a Word Hunt board; even at
the DeepMind dev's assumed 3x (~145 tok/s) that is > 40 s before the first word,
against a 75 s clock and a 1.35x measured speedup.

## Decision

Skip. Thinking-on does not improve validity within any race-shaped budget, and
the measured speedup is 1.35x, not 3x. Seat stays on text grid, thinking off,
plain 12B on 8080. MTP heads / drafter training not started (out of scope).
