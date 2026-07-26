# Chip Fumbler

A heads-up Texas Hold'em poker agent whose strategy is evolved with a genetic algorithm.

Each agent's playstyle is encoded as a chromosome of 11 real-valued genes (aggressiveness, bluff
probability, tightness, and a couple of experimental traits). Populations of chromosomes play each
other in simulated matches, the winners get bred, and after enough generations you end up with a
strategy that reliably beats the random ones it started from.

Built for **ECE 470 (Artificial Intelligence)** at the University of Victoria, August 2024.

---

## Table of Contents

- [How It Works](#how-it-works)
- [The Chromosome](#the-chromosome)
- [Decision Making](#decision-making)
- [Hand Evaluation](#hand-evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Layout](#repository-layout)
- [Results](#results)
- [Known Limitations](#known-limitations)
- [Future Work](#future-work)
- [Authors](#authors)
- [References](#references)

---

## How It Works

```
                 ┌──────────────────────────┐
                 │  Random initial          │
                 │  population (N=20)       │
                 └────────────┬─────────────┘
                              ↓
                 ┌──────────────────────────┐
   ┌────────────→│  Current population      │
   │             └────────────┬─────────────┘
   │                          ↓
   │             ┌──────────────────────────┐
   │             │  Round-robin matches     │   every chromosome plays
   │             │  → fitness per agent     │   every other chromosome
   │             └────────────┬─────────────┘
   │                          ↓
   │             ┌──────────────────────────┐
   │             │  Selection → crossover   │
   └─────────────┤  → mutation (PyGAD)      │
                 └────────────┬─────────────┘
                              ↓  (after N generations)
                 ┌──────────────────────────┐
                 │  Best chromosome → JSON  │
                 └──────────────────────────┘
```

Fitness is **relative**, not absolute: a chromosome is only ever scored against the rest of its own
population. Two scoring methods are implemented in `simulation.py`:

| Method | Behaviour |
| --- | --- |
| `"wins"` | +1 for each match won, −1 for each lost. Rewards consistency. |
| `"chips per hand"` | `BUY_IN / num_hands`, signed by result. Rewards winning *fast* — i.e. efficient chip accumulation. |

### GA configuration

The PyGAD instance in `training.py` is set up as follows:

| Setting | Value |
| --- | --- |
| Parent selection | Steady-state selection (`sss`) |
| Elitism | Top 2 parents kept each generation |
| Crossover probability | 0.9 (single point) |
| Mutation | `random`, 10% of genes |
| Gene space | `[0.0, 1.0]`, step `0.001` |
| Genes per chromosome | 11 |

Round-robin evaluation is the expensive part — a 20-chromosome generation is 190 unique pairings.
`play_matches()` parallelises this with `joblib`, which took one generation from ~70 seconds down to
just over 10 on an 8-core i7-10700K.

---

## The Chromosome

All genes are floats in `[0, 1]`. See `chromosome.py`.

| Gene | Purpose |
| --- | --- |
| `aggressiveness_preflop` | Coefficient that scales up raise sizing, per street. Applied once the |
| `aggressiveness_flop` | agent has already decided to raise — either because EV is positive or |
| `aggressiveness_turn` | because it decided to bluff. |
| `aggressiveness_river` | |
| `bluff_probability_preflop` | Probability of bluffing on a hand the agent has evaluated as −EV, |
| `bluff_probability_flop` | per street. Checked *only* after EV comes back negative. |
| `bluff_probability_turn` | |
| `bluff_probability_river` | |
| `tightness_vs_looseness` | How selective the agent is about playing a hand. High tightness → only plays strong hands; low (loose) → plays almost anything. Feeds `perceived_win_percentage()`, which remaps raw hand strength onto a modified win estimate. |
| `dynamic_vs_static` | *Experimental.* Random mid-game strategy shifts. |
| `bet_size_variability` | *Experimental.* Widens the range of bet sizes. |

> **Note on the last two genes:** `dynamic_vs_static` and `bet_size_variability` were found to be
> unhelpful or actively counter-productive during testing. They are still part of the 11-gene
> chromosome and are still evolved, so they remain a source of noise. Dropping them to 9 genes is
> probably the single easiest improvement to make here.

A sample evolved chromosome (`best_chromosomes/*.json`):

```json
{
    "aggressiveness_preflop": 0.467,
    "aggressiveness_flop": 0.827,
    "aggressiveness_turn": 0.096,
    "aggressiveness_river": 0.791,
    "bluff_probability_preflop": 0.757,
    "bluff_probability_flop": 0.808,
    "bluff_probability_turn": 0.272,
    "bluff_probability_river": 0.408,
    "tightness_vs_looseness": 0.404,
    "dynamic_vs_static": 0.828,
    "bet_size_variability": 0.021
}
```

Chromosomes are implemented as an immutable, hashable `dataclass`, so every chromosome in a
generation is guaranteed unique and fixed for the duration of that generation.

---

## Decision Making

Every action the agent takes routes through a single expected-value calculation:

```
EV = (%Win × $WinAmount) − (%Loss × $LossAmount)
```

where `$WinAmount` is the current pot and `$LossAmount` is what the agent has already committed.
`%Win` comes from the hand evaluator, adjusted by tightness:

```
WinRateOdds = HandEvaluation × Tightness
```

The flow, per decision point:

1. Identify the current street (preflop / flop / turn / river).
2. Evaluate hand strength.
3. Convert to a perceived win percentage using `tightness_vs_looseness`.
4. Compute EV.
5. **EV > 0** → recalculate a raise amount boosted by that street's aggressiveness, and raise by the
   amount that maximises EV (call if the maximising raise is 0).
6. **EV ≤ 0** → roll against that street's bluff probability.
   - Bluffing → raise, sized by swapping tightness for bluff probability in the win-rate-odds
     equation.
   - Not bluffing → check if possible, otherwise fold.

---

## Hand Evaluation

**Preflop.** Uses a lookup table of published starting-hand win rates (from Rake the Rake), indexed
by hole cards and number of players, with suited and unsuited hands handled separately. Pocket aces
heads-up, for example, resolve to 84.9%.

**Postflop (flop / turn / river).** Delegates to the `evaluate` function in the `texasholdem`
library, which finds the strongest 5-card hand available from the hole cards plus the board. It
returns a rank from **1 (royal flush) to 7462 (worst possible hand)**, which is normalised into a win
percentage.

Heads-up play makes this reasonably reliable — the opponent's hole cards are the only genuinely
unknown quantity.

---

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/BenDNKennedy/chip-fumbler.git
cd chip-fumbler

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install pygad numpy texasholdem joblib tqdm
```

---

## Usage

### Train a new agent

```bash
python training.py
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `-l`, `--chromosome_length` | `11` | Number of genes |
| `-p`, `--population_size` | `20` | Chromosomes per generation |
| `-n`, `--num_generations` | `100` | Generations to run |
| `-m`, `--num_parents_mating` | `2` | Parents selected for breeding |

```bash
# larger population, more parents mating, shorter run
python training.py -p 32 -m 16 -n 50
```

Outputs:
- `genetic_algorithm_logs/ga_log_<timestamp>.txt` — per-generation population and fitness dumps
- `best_chromosomes/<timestamp>.json` — the fittest chromosome from the final generation

Mutation rate, crossover probability and scoring method are not exposed as CLI flags — change
`mutation_percent_genes` / `crossover_probability` in `run_genetic_algorithm()` and the
`scoring_method` argument in `fitness_function()` directly.

### Play against a trained agent

```bash
python user_vs_agents.py -f 20240731_232634
```

Loads `chromosomes/<name>.json` and opens the `texasholdem` text GUI for a heads-up match against
that agent. Hand histories are exported to `./pgns`.

### Run simulations directly

```python
from chromosome import Chromosome
from simulation import simulation_1v1, round_robin

a, b = Chromosome.random(), Chromosome.random()

winner, score = simulation_1v1(a, b, game_log=True, scoring_method="chips per hand")

scores = round_robin([Chromosome.random() for _ in range(8)], cycles=3)
```

Table settings live at the top of `simulation.py`: buy-in 500, big blind 50, small blind 25, 2
players. The high blind-to-buy-in ratio is deliberate — it keeps matches short enough to run
thousands of them.

---

## Repository Layout

```
chip-fumbler/
├── agent.py                  # Agent + RandomAgent; decision logic and EV calculation
├── chromosome.py             # Chromosome dataclass, .random(), .to_file(), .from_file()
├── simulation.py             # Match/tournament runners, scoring, parallel evaluation
├── training.py               # PyGAD setup, fitness function, training entry point
├── HandProbability.py        # Preflop starting-hand win-rate lookup
├── FullHandProbability.py    # Extended preflop probability table
├── user_vs_agents.py         # Play a human-vs-agent game in the text GUI
├── game_test.py              # Game simulation sanity checks
├── training_test.py          # Training pipeline sanity checks
├── best_chromosomes/         # Best chromosome per training run (auto-saved)
├── chromosomes/              # Hand-picked / saved chromosomes for play and benchmarking
└── notebooks/                # Exploratory analysis
```

---

## Results

Trained agents were benchmarked over **500 matches** against six fixed opponents:

- a **neutral bot** — all aggressiveness and tightness genes at 0.5, bluffing at 0. This is the bare
  EV strategy with no perceived-win-rate manipulation and no bluffing.
- five **random bots** sampled from a starting population.

| Mutation % | Parents mating | Crossover | Scoring | vs. neutral | vs. randoms |
| ---: | ---: | ---: | --- | ---: | ---: |
| 20 | 2 | 90% | wins | 50.6% | 55.16% |
| 20 | 16 | 90% | wins | 38.6% | 59.24% |
| 10 | 2 | 90% | chips won | 44.6% | 55.24% |
| 20 | 2 | 90% | chips won | 51.8% | 55.60% |
| 20 | 16 | 90% | chips won | **55.2%** | **55.76%** |

Two things stand out:

**More parents mating helps — usually.** Raising `num_parents_mating` produces a visibly more diverse
population, meaning more distinct strategy archetypes get tested and a strong one is more likely to
surface. The best configuration overall was 16 parents with chips-won scoring.

**Poker strategies are non-transitive.** Row 2 is the clearest case: 59.24% against the random bots
but only 38.6% against the neutral bot. A chromosome can hard-counter one opponent while being
hard-countered by another, which is exactly why a single scalar fitness number is a lossy way to
describe a poker strategy.

Overall the trained agents landed around **15% better than a randomly selected chromosome** over 100
generations.

---

## Known Limitations

- **Fitness is relative, not absolute.** Chromosomes are only ever scored against their own
  population, so "fitness" measures being better than your cohort, not being good at poker. There is
  no objective external benchmark and no strong opponent to test against.
- **Poker is stochastic.** With one match per pairing, a meaningful share of every fitness score is
  variance rather than skill. Raising `cycles` helps at a linear cost in runtime.
- **The neutral-bot benchmark is weak.** Beating a 0.5/0.5/no-bluff bot 55% of the time is a low bar;
  it says the agent learned *something*, not that the strategy is strong.
- **Two genes are dead weight.** `dynamic_vs_static` and `bet_size_variability` did not help and are
  still being evolved.
- **Heads-up only.** `MAX_PLAYERS = 2` throughout; the multi-player probability tables exist but the
  agent logic assumes one opponent.

---

## Future Work

1. **Better evaluation metrics.** Score on risk management, bluff success rate, and positional win
   rate rather than a single chip-count number.
2. **Richer chromosomes.** Opponent profiling, explicit risk aversion, and position-dependent
   playstyle would allow strategies the current 11 genes can't express.
3. **Multiplayer tables.** Heads-up was chosen for tractability; a full table needs a different
   approach to hand evaluation and opponent modelling, but would make the agent far more interesting.

---

## Authors

ECE 470, University of Victoria — August 2024

- Kaden J. Taylor
- Job Matthew C. Yap
- Ben D. N. Kennedy
- Chris Hyggen

Upstream repository: [chyggen/chip-fumbler](https://github.com/chyggen/chip-fumbler)

---

## References

- [`texasholdem`](https://github.com/SirRender00/texasholdem) — game engine, hand evaluator, and text GUI
- [PyGAD](https://pygad.readthedocs.io/) — genetic algorithm framework
- [Win Percentage of Every Poker Starting Hand](https://www.raketherake.com/news/2023/05/win-percentage-of-every-poker-starting-hands) — Rake the Rake
- [The Basics of Poker EV | Poker Quick Plays](https://www.youtube.com/watch?v=jiPmaif9szQ)
