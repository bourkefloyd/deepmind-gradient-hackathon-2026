# Running Gemma 4 locally for screen agents (and why mlx-vlm beats Ollama)

A short writeup of what we found benchmarking **Gemma 4** on Apple Silicon for a
vision agent that has to *look at a screenshot and tap the right thing*. If
you're on a MacBook Pro (M-series, including the M5), this is the TL;DR.

## TL;DR

- **Use Gemma 4 12B Unified through [mlx-vlm](https://github.com/Blaizzy/mlx-vlm), 4-bit.** It's the sweet spot: ~90% on our suite, ~2.3s per step, only ~6.4 GB of RAM.
- **Do NOT use Ollama's `gemma4:*-mlx` tags for vision.** They're converted **text-only** - the vision tower is dropped, so the model never actually sees the screenshot. It taps blind and loops. (It "works" only on pure-text tasks, which hides the problem.)
- The big 26B model is the quality ceiling (100%) but needs ~28 GB and a 64 GB Mac. For most laptops, **12B is the one to run.**

## The setup

The task is a real agent loop: the model gets a screenshot + an instruction and
must emit an action (tap coordinates, text, etc.). We score whether the tap
actually lands on the target, whether OCR is right, and a ladder of
text-reasoning tasks (tier1 = easy → tier3 = hard, multi-step).

Everything local is free at the margin (your electricity, your Mac). The only
cost numbers are the cloud baselines.

## Results (our suite, ~700 runs)

| Model | How it runs | Pass rate | Latency / step | RAM | Notes |
|---|---|---:|---:|---:|---|
| **Gemma 4 12B Unified** | **mlx-vlm, 4-bit** | **~90%** | **~2.3s** | **~6.4 GB** | **Recommended.** Perfect on grounding + easy/medium tasks |
| Gemma 4 12B Unified | mlx-vlm, QAT 4-bit | ~91% | ~3.0s | ~10 GB | No real lift over plain 4-bit, bigger + slower → skip |
| Gemma 4 26B-A4B (MoE) | mlx-vlm, 8-bit | 100% | ~3.8s | ~28 GB | Quality ceiling, needs a 64 GB Mac |
| Gemma 4 E4B | mlx-vlm, 8-bit | ~76% | ~1.8s | ~12 GB | Small + fast but **bad at grounding** (taps miss) |
| Gemma 4 E2B | mlx-vlm, 8-bit | ~76% | ~0.9s | ~8.5 GB | Fastest, same grounding weakness |
| **Gemma 4 12B** | **Ollama (`*-mlx`)** | **~52%** | ~6.3s | - | **Vision tower dropped - effectively blind** |
| Gemini 3.1 Flash-Lite | Cloud API | 100% | ~1.3s | - | Cheap cloud baseline (~$0.006 for the whole suite) |
| Gemini 3.5 Flash | Cloud API | ~90% | ~2.1s | - | Cloud baseline |

Where the 12B (mlx-vlm) wins and loses, by category:

- **Grounding (tap the right spot): 100%** ← this is the whole game for a screen agent
- Easy text tasks: 100% · Medium text tasks: 100%
- OCR: ~81% · Hardest multi-step tasks (tier3): ~75%

For comparison, the same 12B through **Ollama** scored **0% on grounding and 0%
on OCR** - it only passed the pure-text tasks, because it literally can't see
the image.

## Why mlx-vlm and not Ollama?

This is the headline finding, and it's easy to get burned by:

- Ollama's `gemma4:12b-mlx` (and the other `*-mlx` tags) are **text-only
  conversions**. The image encoder is stripped out during conversion, so the
  model silently ignores screenshots. You get confident-looking actions that tap
  nowhere and loop forever.
- **mlx-vlm** keeps the real vision tower intact and runs it on Apple's MLX
  (genuine Metal acceleration). That's why it can actually ground taps.

Also worth knowing: the *stock* community 4/8-bit Gemma 4 quants (mlx-community /
unsloth) can emit garbage because they quantize sensitive embedding layers. For
the 26B and E-series we use "PLE-safe" builds. The **12B Unified is different** -
it has no separate vision encoder and no PLE layers (image patches go straight
into the decoder), so a plain `mlx-community/gemma-4-12B-it-4bit` quantizes
cleanly. One less footgun.

## Recommendation for a MacBook Pro (M5)

1. **Run Gemma 4 12B Unified via mlx-vlm, 4-bit.** ~6.4 GB fits comfortably even
   on 16-24 GB Macs, single-digit seconds per action, ~90% quality, and crucially
   100% on actually hitting the right target on screen.
2. **Don't reach for Ollama for vision.** It's great for text models, but its
   Gemma 4 vision builds are blind. If you only ever test text prompts you won't
   notice until it's controlling a real screen.
3. **Skip QAT 12B** - no measurable quality gain over plain 4-bit, just bigger
   and slower.
4. **Only go to 26B** if you have a 64 GB+ Mac and want the last ~10% of
   reliability.
5. **Cloud sanity check:** Gemini 3.1 Flash-Lite was 100% for ~$0.006 across the
   whole suite. If you don't care about local/offline/privacy, that's the lazy
   baseline. The point of the local stack is doing this *on your own machine with
   no API*, and the 12B gets you ~90% of the way there for free.

### Quick start

```bash
make mlxvlm-up      # serves Gemma 4 on localhost:8080 (downloads weights first run)
make mlxvlm-status  # confirm it's up and which model is loaded
```

Set the model to the 4-bit 12B build:

```bash
export MLXVLM_MODEL="mlx-community/gemma-4-12B-it-4bit"
```

Then pick the **Gemma 4 12B Unified (mlx-vlm, vision, 4-bit)** option in the app.

---

*Numbers are from our internal agent benchmark (grounding + OCR + tiered
reasoning, ~700 runs, June 2026). Your mileage will vary with task and prompt,
but the mlx-vlm-vs-Ollama gap on vision is not subtle.*
