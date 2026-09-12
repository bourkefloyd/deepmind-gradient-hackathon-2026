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

Support check (5 min): `mlx_vlm/speculative/drafters/gemma4_unified_assistant`
and `speculative/mtp.py` exist in mlx-vlm 0.6.3 (also `gemma4_assistant` for the
E4B/31B drafters); mlx-lm 0.31.3 has neither MTP nor the unified 12B. HF has
`mlx-community/gemma-4-12B-it-assistant-{4,5,6,8}bit,bf16` conversions of
`google/gemma-4-12B-it-assistant`. So: yes on the Mac, via mlx-vlm. The MTP path
verifies with deferred greedy decoding, which is why sampling penalties do not
apply to it.

## Measured, the 20 seed-0 eval boards, text-grid prompt, thinking off, max_tokens 400

| | plain 12B (8080) | 12B + MTP drafter (8081) |
|---|---:|---:|
| tok/s mean / median (20 boards) | 39.4 / 38.4 (min 24.8, max 55.4) | 58.3 / 59.5 (min 48.6, max 65.0) |
| speedup (tok/s) | 1.0x | **1.48x mean, 1.55x median** |
| latency median | 7.3 s (9/20 stopped early) | 6.7 s (0/20 stopped: all hit 400 tokens) |
| valid words / call | 7.2 | 1.5 (greedy MTP path loops `WORD\nWORD`; the repetition/presence penalties are not applied) |

Earlier 6-board pass: 48.2 -> 65.0 tok/s, 1.35x. Plain-12B tok/s is lower on the
20-board pass partly because short early-stopped answers (70-110 tokens) carry
the prefill in their tok/s.

## Thinking on + drafter, 1 board, max_tokens 6000, 150 s cap

82.0 s, 6000 tokens at 73.2 tok/s, 11.6k chars of reasoning, **still no answer**
(`finish=length`). Thinking-on needs > 6000 tokens on a Word Hunt board; even at
the DeepMind dev's assumed 3x (~145 tok/s) that is > 40 s before the first word,
against a 75 s clock and a 1.35x measured speedup.

## Decision

Skip. Thinking-on does not improve validity within any race-shaped budget, and
the measured MTP speedup is 1.5x, not 3x; on the thinking-off seat it also costs
validity (greedy loops). Seat stays on text grid, thinking off, plain 12B on
8080. MTP heads / drafter training not started (out of scope). vLLM
`--speculative-config '{"method":"mtp","model":"google/gemma-4-12B-it-assistant",...}'`
on a GPU box is the path if this is ever revisited.
