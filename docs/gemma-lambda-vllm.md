# Gemma 4 12B on Lambda (vLLM)

Hosted open model for the arena's Gemma seats, on the same Lambda A100 that trained the nano (`nano/README.md`, GPU training).

| | |
|---|---|
| Base URL | `http://129.146.67.197:8000/v1` (OpenAI-compatible) |
| Model id | `google/gemma-4-12B-it` (bf16, vLLM 0.29.0+cu129, `--max-model-len 4096 --reasoning-parser gemma4 --tool-call-parser gemma4 --enable-auto-tool-choice`) |
| Auth | Bearer key; GCP Secret Manager `gemma-lambda-api-key` (project `actionfleet-live`; `nanoagent-cloud@` SA has `secretAccessor`). Never in git or chat. |
| Instance | Lambda `gpu_1x_a100_sxm4`, us-west-2, id `3e9cacc31f214828af942dd70bf057c7`, $1.99/h, up since 13:40 PT 2026-09-12 |
| Firewall | Lambda account rules: tcp 22, icmp, **tcp 8000** (added via `PUT /api/v1/firewall-rules`) |

Health check (from anywhere):

```sh
KEY=$(gcloud secrets versions access latest --secret gemma-lambda-api-key --project actionfleet-live)
curl -m 5 http://129.146.67.197:8000/v1/models -H "Authorization: Bearer $KEY"      # -> {"data":[{"id":"google/gemma-4-12B-it",...}]}
curl -m 5 http://129.146.67.197:8000/health                                           # 200, no auth
```

Seat settings: `chat_template_kwargs: {"enable_thinking": false}`, `temperature 0.2`, `max_tokens 400`. `gemma_seat` reads the key
from `MLXVLM_API_KEY` and the endpoint from `--base-url` / `MLXVLM_BASE_URL`.

## Smoke and 5-board eval (from the Mac, 15:05 PT)

- Raw chat completion, 400 tokens, thinking off, T=0.2: **9.95 s** wall (~40 tok/s single stream, prompt 73 tokens).
- `python -m gemma_seat.eval --n 5 --seed 0 --modality text --base-url http://129.146.67.197:8000/v1 --model google/gemma-4-12B-it --timeout 60`
  (`docs/gemma-lambda-vllm-eval-n5.json`):

| metric | Lambda vLLM bf16 (n=5) | Mac mlx-vlm 4-bit (n=20, `gemma_seat/results/eval_n20_seed0.md`) |
|---|---:|---:|
| valid-word rate | 21% | 20% |
| mean word length (valid) | 3.65 | 3.79 |
| words / call (emitted) | 129 | 25.5 |
| valid words / call | 25.0 | 5.1 |
| mean score / board | 7560 | 1780 |
| latency / call mean / p50 | 7.7 s / 9.1 s | 2.1 s / 1.8 s |

Reading: same hallucination rate (about 4 in 5 emitted words are not on the board), but the bf16 model uses the whole 400-token
budget (129 words per call vs 26), so each call is ~4x slower and yields 5x the valid words. For the hand-cadence seat, cap
`max_tokens` at ~120 (about 2.5 s/call) or stream and start tracing the first words while the rest arrive. Speculative
decoding (MTP) was not enabled: the plain server came up first time and there was no slack before the 3:30 PT stop.

Ops: logs `~/vllm.log`, venv `~/vllm-env` (uv, cu129 wheels from `https://wheels.vllm.ai/0.29.0/cu129`; the PyPI wheel is CUDA 13 and
the box's driver 570 is CUDA 12.8), weights in `~/.cache/huggingface`. Restart:
`HF_TOKEN=$(cat ~/.cache/huggingface/token) nohup ~/vllm-env/bin/vllm serve google/gemma-4-12B-it --api-key $(cat ~/vllm_api_key) --port 8000 --host 0.0.0.0 --max-model-len 4096 --reasoning-parser gemma4 --tool-call-parser gemma4 --enable-auto-tool-choice --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image": 1}' > ~/vllm.log 2>&1 &`
