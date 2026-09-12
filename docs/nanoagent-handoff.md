# nanoagent handoff (trimmed for the Word Hunt nano)

Source: actionfleet `sandbox/nanoagent/records/HANDOFF.md` and `sandbox/nanoagent/README.md` as of 2026-09-12.
Only the parts that bear on `nano/model.py` / `nano/train.py` and the one-slide "prior work" story are kept; the
AndroidWorld farm ops, records 0000-0023 and the Toon Blast lane are left out.

## The model and the recipe (what `nano/` copies)

- **Architecture** (`nanoagent/model.py` -> `nano/model.py`): pre-norm transformer, one dial `depth`;
  width = 64 * depth, heads = depth (head dim 64), MLP 4x. depth 4 ~ 3M params, depth 6 ~ 10M, depth 8 ~ 25M.
  Heads read from `[CLS]`: `type_logits` (action type), `target_logits` = pointer over candidate tokens
  (`dot(query(cls), key(h[target]))`, illegal candidates masked to -inf), `value_logit` = P(success from here).
  The GUI version also had text-span heads and an optional frozen-image adapter; both are dropped here.
- **Loss** (`nanoagent/train.py` -> `nano/train.py`): soft cross-entropy against soft label distributions for
  type and pointer (rows whose target sums to 0 are skipped), BCE-with-logits on value (weight 0.5). Soft
  targets (L1 rung) beat one-hot BC (L0) - that was the point of the "hindsight soft targets" ladder.
- **Optimizer**: AdamW lr `3e-4 * (4/depth)**0.5` (smaller models take larger lr, nanochat-style), betas
  (0.9, 0.95), weight decay 0.1, grad-norm clip 1.0, cosine to 10% of lr. **fp32, no AMP** on CPU/MPS (bf16
  autocast only on CUDA; kept off so records stay comparable). Non-finite losses are counted and skipped, not
  crashed on.
- **Recipe that held as head** (record 0016 -> 0021): depth 4 (4.2M), fp32, 2500 steps, batch 32. Capacity
  depth 6 was tried (0018/0019) and did not move AndroidWorld success; Word Hunt is a different data regime
  (exact oracle, infinite boards), so depth 6 is the stage number here per PLAN.md.
- **Device guard** (`lab/device.py` -> `nano/device.py`, `make check-mps`): on Apple Silicon `auto` refuses to
  fall back to CPU. A whole night of gate runs once ran on CPU at ~11 samples/s instead of ~120-175 on MPS
  because a sandboxed agent shell blocked Metal; the logs looked normal. Preflight every long run; the first
  log line must read `Device: mps`. One MPS trainer at a time.
- **Policies** (`nanoagent/policy.py` -> `nano/policy.py`): `StudentPolicy` decodes independently per head
  (type argmax / temperature sample, then pointer restricted to legal targets), reports
  `confidence = p(type) * p(target)` and `value`. `EscalatingPolicy` wraps the student and hands a step to a
  teacher when value/confidence are low (or on "stuck"/"done" triggers in the GUI version); the teacher's
  cost is charged per escalated step. Runtime escalation was a *product* number, not a student improvement
  (0021: 14/30 at $0.29 vs student 9/30).

## Numbers for the slide (records 0024-0031, AndroidWorld)

| model | eval /120 | holdout /30 | note |
|---|---|---|---|
| nano student S8-plain, depth 4, 4.2M (head 0016-0023) | 23-26 | 8-9 | four-seed spread 18-26; five records of label-volume/label-form changes did not move it |
| teacher ceiling gemini-3.8-flash (0007) | 78 | - | |
| UI-Voyager 4B + LoRA `uivb_r1_s0` (0024, first swap) | 79 | 19 | step function was swapping the student for a small VLM |
| Gemma 4 E4B zero-shot (0025) | 2 | 0 | cannot ground on a phone screen |
| Gemma 4 12B-QAT zero-shot (0025 / 0026 at 560 vision tokens) | 1 -> 10 | 0 | |
| Gemma 4 31B zero-shot, 560 tokens, `[y, x]` pointing (0026) | 71 | 14 | pointing 90% pure / 80% in agent form |
| Gemma 4 31B + LoRA, both seeds (0027 / 0028) | 67-76 | 17-19 | inside seed noise; Unsloth QLoRA went backwards (66) |
| Gemma 4 31B, thinking on (0028 diagnostic) | 84 | - | +13 over its own thinking-off read, 1.8x wall |
| **Gemma 4 31B + element index + thinking (`g4b31e_think`, 0029/0031 head)** | **91** | **18** | sealed 19/24; zero-shot, no adapter |

Lessons that transfer to today:

1. **Interface beats fine-tuning.** 0027-0029 retired 0024-style LoRA on the 31B: three LoRA loops stayed
   inside seed noise, while two zero-shot interface changes (element index instead of coordinates, thinking
   on) moved 71 -> 91. Hence PLAN.md section 10: LoRA is a slide, not a build item.
2. **Vision budget matters more than model size for pointing.** `max_soft_tokens` 1120 halved E4B's pointing
   and made the 12B random; 560 is the number (0026). Ollama `gemma4:*-mlx` tags drop the vision tower
   (`docs/gemma4-local-vision-findings.md`).
3. **Thinking is an inference-budget knob** (+13 on eval, 1.8x wall). In a 75 s match it stays off.
4. **Pre-register the gate, two seeds, never weaken a gate after the numbers land.** Frozen suites are
   frozen; new targets get new suites.
5. **Diagnose before you fix**: every record opens with the presumed limiting factor and closes with whether
   it was. For the nano today the analogue is `nano/rollout.py` (words found / valid rate / score vs the
   random swiper) before any recipe change.

## Serving the current GUI head (for reference only)

`google/gemma-4-31B-it`, `--vlm-flavor gemma4e`, `VLM_CHAT_TEMPLATE_KWARGS='{"enable_thinking": true}'`,
560 vision tokens, T=0.3, `AW_RAW_XY=1`, no adapter. Next planned record there was 0032 AF-DAR (dense action
representation for a small student) - not part of today.
