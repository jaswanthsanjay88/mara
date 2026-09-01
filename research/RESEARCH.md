# mara research: the FFN-interpolation study

**Thesis.** Needle 2's SAN paper (arXiv:2607.18363) tested only two points:
0% FFN vs 100% FFN at matched parameters. We map the space between —
interleaved, bottlenecked, and looped variants — hypothesizing that a small
amount of FFN capacity recovers parametric recall while keeping SAN's
parameter efficiency.

## Variants (`research/budget.py`)

| name | config | what it tests |
|---|---|---|
| standard | ffn_every=1 | full transformer baseline |
| san | ffn_every=0, qk_norm | their recipe, deepened to match params |
| interleaved2/3 | ffn every 2nd/3rd block | sparse FFN placement |
| bottleneck128/256 | slim FFN everywhere | reduced FFN width instead of count |
| looped2/4 | one block applied k× | compute-for-parameters trade |

All auto-depth-matched to 24,648,192 params ±1% (`budget.match_depth`),
FLOPs/token reported transparently per run.

## Protocol

1. Same data as mara-small pretraining (TinyStories stream), same seed,
   9k steps × 32,768 tok = ~295M tokens each.
2. Metrics: val loss every 500 steps → `runs/<variant>/metrics.csv`.
3. After the matrix: parametric-recall probe (`factworld.py`) and
   tool-call accuracy (`toolbench.py`) on each final checkpoint.

## Running on Kaggle

One-time:
- Upload `mara.tar.gz` contents + `data/train.bin`, `data/val.bin` as a
  private Kaggle dataset named `mara-data`.

Per experiment (~4.5h, fits one GPU session):
```python
!cp -r /kaggle/input/mara-data/mara /kaggle/working/mara
%cd /kaggle/working/mara
!python -m research.run_experiment --variant interleaved2 --data-dir data
```
Run variants sequentially across sessions; commit the notebook after each
so `runs/<variant>/` persists in versions.

## Decision rule

Winner = lowest val loss at matched params AND highest recall-probe
accuracy among within-noise losses (<0.01 nats of best). That config
becomes `mara-nano`, the production architecture for ESP32 deployment.
