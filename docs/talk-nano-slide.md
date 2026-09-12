# The nano slide: three numbers to say on stage

Open on the nano's finger on the board, not on this slide (PLAN.md section 8). Then:

1. **11 million parameters, zero tokens, 2 ms a move.** A depth-6 transformer (`nano/model.py`) that sees 16 tiles and its own path, no dictionary at inference; the words live in the weights. Trained in 30 minutes on this laptop on 200k boards it solved for itself (`nano/train.py`, 17k steps, MPS). It decides in about 2 ms on CPU; the Gemma 12B seat needs about 2 s per call.

2. **Gate passed at 9.6x, 97% valid.** Pre-registered before training (PLAN.md section 3): on 50 unseen boards it had to score at least 4x a random swiper and beat the Gemma E4B seat on valid-word rate. It scored 9126 vs 952 (9.6x) with 97% of its submits being real words on the board; the Gemma 12B text seat is at 20% valid (it names words that are not on the board). Same 20-board league: nano 11245, 12B 1780, random 1180 (`docs/league.md`). Honest caveat: its words are short - mean 3.4 letters, longest `stokes`/`listen`; Gemma finds the longer ones when it is right.

3. **It learns the words it was just shown.** Between rounds every validated word from any seat - human or Gemma - goes into a 10-second CPU update (`nano/learn.py`: 50 BC steps, replay-mixed, held-out check, rollback if it drops). Over 10 simulated rounds its recall of the taught words on that board went from 35% to 46% while the held-out score stayed flat. Say exactly that: it learns the words it was just shown; it does not get better at Word Hunt in ten rounds. Gemma is the teacher on stage, at $0.

Close: *10M params, zero tokens, human pace; the 12B finds longer words at 1000x the compute.* Fund V line: frontier open model as teacher, tiny student that acts at reflex speed on the edge, measured against people (`docs/nanoagent-handoff.md` for the prior-work table).
