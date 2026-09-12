# Themed levels

15 pre-canned boards in `data/levels.json`, played in order (round 1 = level 1, each rematch
moves to the next, wrapping after the last). Each board is seeded with 3-5 everyday words for its
theme and then filled to maximise *common* word density, so a casual player scores 1,000-2,000 in a
75 s round without effort.

Regenerate this page with `python scripts/levels_report.py --md docs/levels.md`.

## Levels

| # | Theme | Board | Words | Common | 3 / 4 / 5+ | Max | Longest | Theme words | Casual ~ |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Kitchen | GGETTNASOLIRPSPT | 545 | 158 | 50 / 55 / 53 | 420,200 | STARTING | EGG POT PLATE | 1,900 |
| 2 | Animals | NTGNSAIELTRDMAES | 638 | 153 | 45 / 46 / 62 | 582,200 | MATERIAL | TIGER ANT RAT SEAL | 1,800 |
| 3 | Space | STOOUNEMNRASDELP | 492 | 141 | 49 / 44 / 48 | 359,000 | TONSURES | SUN MOON PLANET | 1,800 |
| 4 | Beach | IPECETAHSRETIDAS | 578 | 165 | 44 / 55 / 66 | 516,800 | TEACHERS | PIER HEAT SEA | 1,900 |
| 5 | Music | ISGTRNEATOOMSHPS | 400 | 129 | 38 / 47 / 44 | 261,400 | MOORINGS | SING TEMPO TONE HORN | 1,900 |
| 6 | Sports | SHOCDTSPEIAENLVR | 608 | 169 | 60 / 61 / 48 | 460,200 | OPERATED | SHOT DIVE NET | 1,800 |
| 7 | Garden | ESHPTESOLARTBEAN | 542 | 152 | 38 / 56 / 58 | 428,600 | STEALERS | HOSE TREE BEAN | 2,000 |
| 8 | Weather | PTRWSAESHEDNICSU | 528 | 148 | 49 / 48 / 51 | 416,700 | CHAPTERS | HEAT DEW ICE SUN | 1,800 |
| 9 | Colors | SRBPETALANICGLKP | 567 | 158 | 53 / 57 / 48 | 466,600 | TRACKING | BLACK TEAL PINK | 1,900 |
| 10 | Body | SLMREPAEINSTKSOE | 568 | 172 | 56 / 63 / 53 | 421,300 | SPINSTER | PALM NOSE TOE SKIN | 1,900 |
| 11 | School | CLSNKAETSSPREDIA | 490 | 145 | 41 / 52 / 52 | 430,700 | DISASTER | CLASS PEN DESK | 1,900 |
| 12 | City | ACSMFERALOTEPANS | 625 | 168 | 48 / 55 / 65 | 514,300 | SENATORS | CAFE LANE STORE | 1,900 |
| 13 | Farm | ILCTSAOSENRADETG | 565 | 141 | 39 / 52 / 50 | 467,800 | CONTRAST | CORN SILO GATE | 1,900 |
| 14 | Ocean | AEFLTREDISENALAT | 530 | 152 | 45 / 56 / 51 | 429,300 | ARSENATE | EEL REEF SAIL | 1,900 |
| 15 | Party | RSRTEDEBINALWTSP | 619 | 169 | 51 / 51 / 67 | 539,600 | RENTABLE | TREAT BAND WINE | 1,800 |

**Columns.** *Words* = every valid enable1 word (3-8 letters) on the board; *Common* = those in the
top 10,000 of `data/common-30k.txt`, split by length (*3 / 4 / 5+*); *Max* = score if every word
were found; *Theme words* = the seed words, all verified findable on the board; *Casual ~* = the
rough human estimate below. ⚠ marks a level under the bar (< 40 common 3-4 letter words, < 3
common 5+ letter words, or a seed word that is not traceable). 15/15 levels pass.

## Casual human estimate

A relaxed player finds roughly one word every 6 s, so ~12 words in 75 s, mostly 3-letter words
with a few 4s: 12 finds at a 3:1 mix is 9 x 100 + 3 x 400 = 2,100; all 3s is 1,200. The estimate
takes 30% of the board's common 3-4 letter words (capped at 12 finds), weights 4-letter words at a
third of the 3-letter rate, and rounds to the nearest 100. Common 5+ letter words (`5+` column) are
upside on top: one `PLATE` is +800.

## Curating

- Edit `data/levels.json`: change a `theme` title, reorder entries, swap in a hand-made 16-letter
  `board` (row-major, lowercase), or delete a level. `n` and `stats` are informational; the game
  reads `theme`, `board` and file order.
- Re-check with `python scripts/levels_report.py` (exit code 1 if a level misses the bar), then
  redeploy (`scripts/deploy.sh <label>`) — the JSON is copied into the image.
- Regenerate a single theme: `python scripts/make_levels.py --only Kitchen`; all themes with a new
  search seed: `python scripts/make_levels.py --seed 7`.
- Runtime knobs: `WH_LEVELS=data/levels.json` (default; set empty or `0` for the old packed/random
  boards), `WH_LEVEL_START=n` to start a room at level n.
