# Research log

Decisions, results and open questions for the inductive-bias study, in
order. Times are Adelaide local time (ACDT, UTC+10:30). Newest entries go at
the bottom of each section's timeline.

## CIFAR-100 grid, first pass (finished Tue 17 Nov 2026, 7:16 pm)

Ran on RunPod pod `cuslyj7h4randg` (community RTX 5090, $0.69/h), three
shards sharing the GPU.

| Stage | Finished | Notes |
|---|---|---|
| Grid start | Mon 16 Nov, 10:21 pm | |
| LeJEPA pretraining | Tue 17 Nov, 7:15 am | 9 backbones (CNN, ViT, MLP x 3 seeds), ~2.9 h each, 3 in parallel |
| Tuning sweep | Tue 17 Nov, 1:07 pm | 180 runs: 10 configurations x 3 anchors x 6 (lr, wd) pairs, seed 0 |
| Final runs | Tue 17 Nov, 7:16 pm | 210 runs: 10 configurations x 7 sizes x 3 seeds |

No run diverged or crashed. Results copied to `results/pod/` (CSVs, LeJEPA
summaries, logs, splits) and the final and tune CSVs to `results/`. The pod
was stopped after the copy was checked; its 50 GB volume (LeJEPA `.pt`
checkpoints) is kept.

LeJEPA kNN validation accuracy (seed 0 / 1 / 2): CNN 0.559 / 0.546 / 0.555,
ViT 0.547 / 0.548 / -, MLP 0.287 / 0.283 / 0.283.

Stop rule check (CNN must beat the ViT at the smallest size): passed. Best
validation at 1%: no aug CNN 0.091 vs ViT 0.071; aug CNN 0.145 vs ViT 0.108.

### Best validation accuracy in the sweep

| Model | Treatment | 1% | 10% | 100% |
|---|---|---|---|---|
| CNN | none | 0.091 | 0.192 | 0.541 |
| CNN | aug | 0.145 | 0.423 | 0.738 |
| CNN | LeJEPA | 0.339 | 0.565 | 0.743 |
| ViT | none | 0.071 | 0.180 | 0.482 |
| ViT | aug | 0.108 | 0.297 | 0.622 |
| ViT | LeJEPA | 0.296 | 0.520 | 0.721 |
| ViT | ImageNet-21k | 0.330 | 0.686 | 0.837 |
| MLP | none | 0.075 | 0.167 | 0.303 |
| MLP | aug | 0.107 | 0.212 | 0.404 |
| MLP | LeJEPA | 0.153 | 0.289 | 0.437 |

### Exchange rates, first pass

These predate the learning-rate ladder fix below and will be recomputed.
E = labelled data the first setup needs / data the second needs, to reach
the target. Brackets: 95% seed-bootstrap interval (10,000 draws).

| Comparison | 10% target (0.430) | 50% target (0.666) |
|---|---|---|
| CNN bias vs ViT, no aug | 1.40x [1.37, 1.43] | undefined |
| CNN bias vs MLP, no aug | more than 1.9x | undefined |
| CNN bias vs ViT, with aug | 2.39x [2.29, 2.48] | more than 2.0x |
| CNN bias vs MLP, with aug | more than 10x | more than 2.0x |
| Augmentation, ViT | 3.02x [2.90, 3.13] | undefined |
| Augmentation, MLP | undefined | undefined |
| Augmentation, CNN | 5.14x [4.99, 5.29] | more than 2.0x |
| Self-supervision, ViT | 6.27x [5.76, 6.80] | more than 2.2x |
| Self-supervision, MLP | more than 1.1x | undefined |
| Self-supervision, CNN | 3.90x [3.72, 4.10] | 1.42x [1.36, 1.51] |
| Pretrained start (IN-21k), ViT | 11.96x [11.27, 12.57] | more than 10.4x |
| CNN lead left over best ViT (IN-21k) | 0.20x [0.19, 0.21] | 0.19x [0.17, 0.24] |
| CNN bias with LeJEPA for both | 1.49x [1.36, 1.64] | 1.31x [1.14, 1.44] |

Reading: the CNN's bias is worth roughly 1.4-2.4x in labels on CIFAR-100;
augmentation widens the gap (it helps the CNN more); LeJEPA on both models
closes most of it (to 1.3-1.5x); an ImageNet-21k ViT beats the scratch CNN
with a fifth of the labels, but that uses outside data.

Figure: `figures/curves_cifar100.png`.

## How the targets and exchange rates work

- The reference setup is the CNN with augmentation. Its seed-mean test
  accuracy at 10% of the labels (4,500 images) is the "10% target", 0.430;
  at 50% (22,500 images) it is the "50% target", 0.666. Targets are fixed
  in the config before results, from the reference only.
- For two setups, each curve's crossing of the target is found with an
  isotonic fit and log2 interpolation between grid sizes; the ratio of the
  two data needs is E. Example: at the 10% target the CNN + aug needs about
  4,500 labels and the ViT + aug about 10,750, so E = 2.39x.
- "More than Nx": one setup never reaches the target within the grid, so
  only a lower bound exists. "Undefined": neither setup reaches it.
- Two targets because the value of a bias depends on the data regime: the
  10% target is the low-data regime, the 50% target the higher-accuracy one.

## Decisions and discussion

### Matched ImageNet starts (Tue 17 Nov, evening)

Darcy asked whether an ImageNet-pretrained CNN was tested against the
ImageNet-pretrained ViT. It was not: the brief made the pretrained start
ViT-only, so the 0.20x row compares a pretrained ViT with a scratch CNN.

The ViT's ImageNet weights are ImageNet-21k then 1k; the CNN has no
ImageNet-21k ResNet-18 in timm. For a like-for-like comparison both models
get ImageNet-1k-only weights from similar heavy-augmentation recipes:

- CNN: `resnet18.a1_in1k` ("ResNet strikes back" A1: 600 epochs, LAMB,
  RandAugment, Mixup, CutMix). The pretrained 7x7 stem conv is kept at
  stride 1 with no max-pool, so every layer starts trained and the stages
  see the same resolution as the scratch CNN (which uses a 3x3 stem).
- ViT: `deit_tiny_patch16_224.fb_in1k` (DeiT-Tiny, the same 192/12/3
  network as ViT-Tiny), patch embedding resampled to 4 px as before.
- New treatment `imagenet1k` for both, on images and audio. The original
  ImageNet-21k ViT stays as the brief's treatment.
- New comparisons: "Pretrained start (IN-1k), CNN", "Pretrained start
  (IN-1k), ViT", "CNN bias with IN-1k start for both", and "CNN lead over
  best in-domain ViT" (best ViT chosen by validation among treatments with
  no outside data).
- Caveat for the write-up: ResNet-18 has 11.2M parameters, ViT-Tiny 5.4M,
  the same ratio as the rest of the study, but capacity matters more for
  pretrained weights.

Darcy approved this ("go for it") and asked that every comparison be sound.

### Learning-rate sweep hit the edge of its grid (Tue 17 Nov, evening)

Checking the sweep: in 19 of the 36 (configuration, anchor) pairs the best
learning rate was the smallest (1e-4) or largest (1e-3) tried, so the true
best may lie outside the grid. Examples: CNN with and without aug at 1e-3
at every anchor, still rising; ViT ImageNet-21k at 1e-4 at 10% and 100%,
still rising as lr falls. That under-tunes some setups and biases E.

Fix (`bias.grid`): a learning-rate ladder 1e-5, 3e-5, 1e-4, 3e-4, 1e-3,
3e-3, 1e-2. Where a (configuration, anchor)'s best lr is the smallest or
largest tried, the sweep adds the next rung at both weight decays, round by
round, until the best is bracketed or the ladder ends (at most 6 rounds).
Finals use the new choice; final runs made with a superseded choice stay
in the logs but `bias.analyse` drops them. First round: 38 extra runs.
Weight decay stays at {0.05, 0.1} as the brief sets.

### Running the extra work

Plan: restart the stopped pod, copy code over by rsync (not git), and run
`BIAS_WORKERS=2 scripts/run_all.sh configs/images.yaml configs/audio.yaml`.
Estimate: images extras ~6-7 h (~$4-5), full audio grid ~26-30 h
(~$18-21), about $25 in all at $0.69/h.

Darcy approved the restart. The pod's host had no free GPU and no community
5090 was in stock, so a retry runs every 20 minutes. Not moved to a new pod
because the CIFAR-100 LeJEPA checkpoints are on the stopped pod's volume
(~9 GPU-hours to rebuild).

Commits stay local: the GitHub repo is public, and Darcy asked to keep
everything local for now.

### Is LeJEPA worth it? (open for audio)

Darcy asked what the LeJEPA experiments add. They answer: does the CNN's
bias keep its value when unlabelled data is plentiful (labels scarce, raw
data not)? On CIFAR-100 the CNN's lead drops from 2.39x (aug for both) to
1.49x (LeJEPA for both) at the 10% target, so label-free pretraining buys
back most of the bias. Without LeJEPA the post can say "the bias is worth
~2.4x in labels"; with it, "and unlabelled data buys back most of that".

LeJEPA works on CNNs too and the study runs it (the tenth configuration).
Final test accuracy, seed mean:

| Labels | CNN + aug | CNN + LeJEPA | ViT + aug | ViT + LeJEPA |
|---|---|---|---|---|
| 1% (450) | 0.14 | 0.34 | 0.10 | 0.30 |
| 10% (4,500) | 0.43 | 0.56 | 0.30 | 0.52 |
| 100% (45,000) | 0.74 | 0.74 | 0.62 | 0.73 |

LeJEPA cuts the CNN's label need ~3.9x at the 10% target, the ViT's ~6.3x;
the bias and self-supervision partly substitute for each other. The CNN arm
is needed to tell "self-supervision helps" from "self-supervision closes
the bias gap".

Cost: images, nothing more (backbones exist). Audio LeJEPA is ~12-15 h of
the ~28 h audio grid (~$9-10): 9 backbones x 66k steps, 250-token ViT.

Options for audio, awaiting Darcy's choice:

1. Keep it in full: two modalities make the unlabelled-data finding a
   pattern rather than an anecdote.
2. CNN and ViT only, skip the MLP (recommended): saves ~a third of
   pretraining (~$3-4); the MLP's LeJEPA result adds little.
3. Drop it for audio; present LeJEPA as images-only.

### Targets near saturation (open)

Darcy asked about adding 75% or 100% targets, and whether saturated tasks
skew the exchange rate.

- A 100% target cannot be measured: any setup needing more labels than the
  reference would need more than the 45,000 that exist, so every such
  comparison is a "more than 1x" bound. A 75% target (reference ~0.71)
  would also be mostly bounds: ViT + aug tops out at 0.62, MLP at 0.41.
  The full curves already show the high-data end.
- Saturation is a real concern. E is a ratio of data needs at a fixed
  accuracy, so it depends on curve slopes at the target. Near a ceiling a
  small accuracy gap becomes a huge data ratio (CNN plateaus at 97%, ViT at
  96%: a 96.5% target gives a near-infinite E for a half-point difference);
  with different plateaus E stops being a rate and measures ceilings. Near
  chance, curves are flat too and E is noisy.
- CIFAR-100 is mostly safe: the reference is at 0.74 and still rising, so
  0.43 and 0.67 sit on the steep part. The 10% target is the cleaner one.
- Audio and hyperspectral have high ceilings but do not saturate quickly.
  Expectations from the literature, not yet measured here: Speech Commands
  small CNNs reach ~95-97% at full data, perhaps 30-60% at 1%; the 50%
  target may sit near the plateau. Hyperspectral with random pixel splits
  passes 90% with a few pixels per class, mostly from leakage; spatially
  blocked splits should be lower and less saturated.

Proposed (not yet built), analysis only, no GPU cost:

1. Plot E against the target accuracy for every comparison, from just above
   chance to the highest level both setups reach. A flat line means E is a
   quotable constant; a climb near the top shows saturation. Lead figure.
2. Keep the pre-set 10% and 50% targets in the table, flagging estimates
   where the reference curve has nearly flattened.
3. For audio and hyperspectral, also report error-rate ratios at fixed n
   ("the ViT makes 1.6x as many errors at n = 4,000"), which stay
   meaningful near the ceiling.
4. A pre-stated fallback rule, written before seeing audio numbers: if a
   target lands on the reference's plateau, redefine targets from the
   reference curve alone (e.g. fixed fractions of the way from chance to
   the reference's full-data score). Using only the reference keeps the
   choice from favouring any comparison.

The audio sweep runs first (anchors 1%, 10%, 100%), so the reference's
plateau shows before most compute is spent.

### Do the architectures get enough training steps? (Tue 17 Nov, ~10:30 pm)

Darcy asked whether different learning dynamics (a ViT ramping up slowly,
a CNN starting fast) are dealt with. Two separate effects:

1. Slower learning from data: handled. A ViT that needs more examples shows
   up as a learning curve that starts lower and climbs later; the curves
   and exchange rates measure exactly that, and it is the study's subject.
2. Slower optimisation: not yet checked, and it could bias the result.
   Every model gets the same step budget (200 epochs' worth, clipped to
   2,000-40,000 steps; at full data the cap means ~114 epochs), cosine
   schedule and 5% warmup. ViTs usually need more epochs and warmup than
   CNNs. If the ViT was still improving when training stopped, it is
   under-trained and the CNN's advantage is overstated. The lr ladder does
   not cover this, and the first pass kept no validation history.

Added:

- Every run now appends its validation curve (step, score) to
  `results/history/<dataset>_<stage>.jsonl`.
- A budget-check stage (`--stage budget`, run by `run_grid.sh` after the
  finals): CNN and ViT, with and without aug, at 10% and 100%, seed 0, at
  their final settings, with 2x the step budget. About 380k steps on
  images (~45 min, under $1); the same check runs for audio.
- Rule, fixed before any results: if 2x the steps raises a setup's
  validation score by 0.01 or more, the budget is too short; then max
  steps rises for every model alike and the affected finals are rerun.
  Validation decides; the test set plays no part. `bias.analyse` writes
  `results/budget_check_<dataset>.csv`.

Smoke test of the ImageNet-1k CNN on the Mac (10%, 400 steps, wd 0.05,
validation): scratch CNN + aug 0.119 (lr 3e-4) / 0.162 (lr 1e-3); ImageNet-1k
CNN 0.336 / 0.548. The pretrained start works with the stride-1 7x7 stem.

### Why LeJEPA for the self-supervision treatment? (Tue 17 Nov)

Darcy asked why LeJEPA is needed to test unsupervised learning. It is not
required; any self-supervised method tests "what is unlabelled data worth".
The brief chose LeJEPA, and it suits the study because:

- It is architecture-agnostic: it needs only an encoder that maps an input
  to a vector, so CNN, ViT and MLP get the identical objective. MAE, the
  main alternative, relies on patch tokens and is a ViT method, which
  would favour the ViT in the comparison that matters.
- It is light on tuning: one main setting (lambda), no teacher network, no
  negatives, no stop-gradient tricks. DINO, BYOL and SimCLR would need
  per-architecture, per-modality tuning, with a risk of tuning one model
  better than another.
- The same recipe runs on images, spectrograms and hyperspectral patches.

Caveats for the post:

- One method gives one answer. "LeJEPA closes most of the gap" shows that
  unlabelled data can substitute for the bias; a stronger method might
  close more, so the result is closer to a lower bound on what unlabelled
  data can buy.
- LeJEPA is not prior-free: it makes augmented views agree, and the
  augmentations are hand-designed per modality, encoding invariances much
  as convolutions do. Part of what it buys back is a different
  human-supplied bias, not pure data.
- Semi-supervised methods (e.g. FixMatch) use unlabelled data during
  supervised training instead of pretraining; a different question, left
  out of scope.

## Remaining work before the write-up (as of Tue 17 Nov, ~10:45 pm)

| # | Work | Status | Compute |
|---|---|---|---|
| 1 | Images: lr ladder, ImageNet-1k pair, budget check; rerun table and figure | Code ready, waiting on pod | ~7 h, ~$5 |
| 2 | Audio: full grid (LeJEPA, sweep, finals, budget check) | Code ready; needs the LeJEPA option | ~26-30 h, ~$18-21 |
| 3 | Hyperspectral: Salinas and Pavia grids, random-vs-block leakage demo | Data and models built; demo not written | a few hours |
| 4 | Saturation-aware analysis: E vs target, error-rate ratios, fallback target rule | Proposed, awaiting OK; rule must be fixed before audio results | none |
| 5 | Analysis note across modalities | After 1-4 | none |
| 6 | Blog draft | | |

About two days of compute once a GPU is free, ~$30. Audio and
hyperspectral do not need the stuck pod; a second pod could start audio
now (needs Darcy's OK).

Open decisions: audio LeJEPA option; OK for item 4; second pod for audio.

## Idea: extend to text (raised Tue 17 Nov)

Darcy suggested extending to inductive biases in transformers for text.
For text the transformer is the default, so the question becomes which
biases help it, or what dropping them costs. Candidate biases:

- Position handling: relative/rotary vs absolute vs no position encoding.
- Locality: sliding-window vs full attention.
- Recurrence/order: LSTM or state-space model vs transformer.
- Local n-grams: 1-D CNN (the counterpart of the image CNN).

Treatments map across: self-supervision = masked-LM pretraining on the
unlabelled pool; pretrained start = a small pretrained BERT; augmentation
is weak for text.

Cautions: common classification benchmarks (AG News, IMDB) are largely
solvable from word counts, so architectures converge and the bias would not
show; tasks needing structure (ListOps, synthetic or order-sensitive tasks)
show it better. Dataset licences must be checked first (stop-and-ask rule;
AG News terms are vague).

Recommendation: finish the three planned modalities and that post first,
then scope text as a follow-up study with its own brief.

## Decisions (Tue 17 Nov, ~11 pm)

Darcy said yes to all three open items:

1. Audio LeJEPA: CNN and ViT only. `[mlp, lejepa]` removed from
   `configs/audio.yaml`; audio pretrains 6 backbones, not 9.
2. Saturation-aware analysis: built (`bias.saturation`, plateau rule in
   `bias.analyse`).
3. Second pod for audio: a community 5090 showed LOW stock (CUDA 13.0) but
   was gone by the create call ("no longer any instances available").
   The 20-minute retry now also tries to create `bias-audio-grid` (same
   image, 40 GB disk, 60 GB volume) and launch the audio grid on it; the
   images pod restart runs images only.

### Plateau rule (fixed before any audio or hyperspectral result)

In every config: `chance` (1/classes) and `targets.plateau` with
`min_gain: 0.01` and `headroom` shares 0.5 (low target) and 0.9 (high
target). A preset target whose reference (CNN + aug) seed-mean test score
gains less than 0.01 from its grid size to the next is on the plateau, and
is replaced by chance + share x (reference score at the largest size -
chance). Only the reference curve is used, so the rule cannot favour a
comparison. `bias.analyse` prints which targets were kept or replaced.

CIFAR-100: both targets kept (gain 0.112 from 10% to 20%, 0.073 from 50%
to 100%). For reference, the headroom rule would give 0.374 and 0.665,
close to the presets 0.430 and 0.666.

### Saturation checks on first-pass CIFAR-100

`python -m bias.saturation <config>` writes `rate_vs_target_<dataset>.csv`,
`error_ratios_<dataset>.csv` and `figures/rate_vs_target_<dataset>.png`
(E at 25-30 target levels between just above chance and the highest score
both curves reach; band = seed bootstrap).

E across the target range (first pass, before the lr ladder):

| Comparison | Targets | E min | E max |
|---|---|---|---|
| CNN bias vs ViT, with aug | 0.14-0.62 | 1.94 | 2.72 |
| CNN bias with LeJEPA for both | 0.34-0.73 | 1.26 | 1.58 |
| CNN bias vs ViT, no aug | 0.08-0.48 | 0.92 | 1.83 |
| Self-supervision, CNN | 0.34-0.74 | 1.10 | 5.17 |
| Self-supervision, ViT | 0.30-0.62 | 3.42 | 8.63 |
| Augmentation, CNN | 0.14-0.54 | 3.67 | 6.64 |
| Pretrained start (IN-21k), ViT | 0.32-0.62 | 10.1 | 14.5 |

Readings:

- With aug, the CNN's lead grows steadily with the target (about 2x at low
  accuracy to 2.7x at 0.62). CIFAR-100 is not saturated, so this is a real
  trend: the bias is worth more the higher the accuracy asked for, within
  this range. A single number hides it.
- With LeJEPA for both, E stays at 1.3-1.6x across the range: a quotable
  constant.
- Without aug, E is erratic at low targets (dips below 1 near 0.25, where
  the two curves run close together near chance) and rises to 1.8x by
  0.48. The low end is noise near chance, as predicted.
- LeJEPA's value to the CNN falls from ~5x to ~1.1x as the target rises;
  it matters most when labels are scarce.

Error-rate ratios (ViT error / CNN error at the same n): with aug 1.05x at
1% rising to 1.44x at 100%; with LeJEPA for both a flat 1.06-1.09x; without
aug about 1.0x up to 20%, 1.15x at 100%.

## Images extras running (Wed 18 Nov, 9:10 am)

After ~10 hours of retries (no community 5090 free anywhere, the stopped
pod's host GPU taken), `cuslyj7h4randg` started at 9:10 am. SSH port moved
to 47553. Code synced by rsync (not git); the 9 LeJEPA checkpoints and
first-pass CSVs were intact; 41 tests pass on the pod. Launched
`BIAS_WORKERS=2 scripts/run_all.sh configs/images.yaml`: 74 tune jobs
pending in round 1 (38 ladder runs + 36 for the ImageNet-1k pair), then
more ladder rounds as needed, finals (new configs plus any whose settings
moved), and the budget check.

Audio pod: still no stock; the retry continues every 20 minutes. Health
checks for running pods every 2 hours.

### RunPod balance ran out (Wed 18 Nov, ~11:15 am)

A community 5090 came into stock for the audio pod, but create-pod failed
with 402 "Your account balance is too low to rent a pod". The images pod
`cuslyj7h4randg` was found stopped (EXITED) at the same time: RunPod
stopped it for the balance. Its /workspace volume is kept, so finished runs
and the LeJEPA checkpoints are safe and the grid resumes where it stopped;
any run in progress at the stop is lost and reruns. Last copy to the Mac
was 10:17 am (12 of the first 74 extra sweep runs). Audio retry and health
checks cancelled until Darcy adds funds.

### Audio pod on a faulty host (Wed 18 Nov, 1:30 pm)

Balance topped up. A community 5090 came free at ~1:30 pm and audio pod
`ihkc39u1v8s318` (bias-audio-grid) was created at $0.69/h. Its host has a
CUDA 13.0 driver. The project's torch 2.15.0 is a cu132 build with no cu130
wheel, and on this host every CUDA call failed with "out of memory" from
cuDevicePrimaryCtxRetain. The image's own torch 2.9.1+cu130 failed the
same way on an empty GPU, so the host itself is faulty. The pod was
stopped after ~10 minutes (volume kept, 60 GB, bills at the stopped-disk
rate until terminated; terminating needs Darcy's OK).

The retry now asks for hosts with CUDA >= 13.2 (matching the torch build),
creates `bias-audio-grid-2`, and tests a CUDA matmul before setting up.
Also fixed: rsync `--exclude data` had skipped `src/bias/data`; excludes
are now anchored (`/data`).

The faulty audio pod `ihkc39u1v8s318` was terminated at Darcy's request (Wed 18 Nov, ~1:40 pm); list-pods confirms it is gone. The images pod `cuslyj7h4randg` is kept (stopped) because its volume holds the LeJEPA checkpoints and the runs in progress.

## Audio grid running (Wed 18 Nov, 1:52 pm)

Pod `f5pskuofih3ygv` (bias-audio-grid-2, community RTX 5090, $0.69/h,
CUDA 13.2 host, 24 vCPU, 60 GB volume). GPU matmul test passed; code synced
by rsync with anchored excludes; 41 tests pass. Launched
`BIAS_WORKERS=6 scripts/run_all.sh configs/audio.yaml` at 1:52 pm:
Speech Commands v2 downloaded (2.26 GB, 8.7 GB extracted), LeJEPA
pretraining of 6 backbones (CNN, ViT x 3 seeds; MLP dropped) running 3 at
a time, 30.3 GB of 32 GB GPU memory in use. Then the tune sweep with the
lr ladder, finals, and the budget check.

The images pod is still waiting for its host GPU; its retry continues.
Health checks every 2 hours cover both pods.

## Images resumed (Wed 18 Nov, 5:33 pm)

`cuslyj7h4randg` started at 5:31 pm after ~6 hours of retries; SSH port
47882. GPU test passed; 18 of the 74 round-1 extra sweep runs had finished
before the balance stop (201 tune rows). Relaunched `run_all.sh
configs/images.yaml` (resumable) at 5:33 pm; 12 grid processes, GPU 100%.
Both pods now run; health checks every 2 hours.

## Health check (Wed 18 Nov, 6:45 pm)

Images: 210 tune rows, 54 round-1 extra sweep runs still pending, GPU 99%.

Audio: shard 1 finished LeJEPA ViT seed 0 (66,283 steps) and saved its
checkpoint, then hit CUDA out-of-memory in the k-nearest-neighbour diagnostic,
which fed 1000 spectrograms through the ViT at once (~15 GB) while two other
shards held 8 GB each. The checkpoint is intact; its `.json` summary (knn
score and loss history) was not written, though the loss history is in
`logs/audio_pretrain_1.log`. Because `run_grid.sh` uses a bare `wait`, the
failure would have let tuning start without shard 1's second backbone.

Fix: `_features` now runs chunks of 200. The patched file was copied to the pod,
the old `run_all.sh`/`run_grid.sh` parents were killed (shards 0 and 2 kept
running), and `/workspace/resume_audio.sh` relaunched shard 1 (which skips the
saved checkpoint), waits for every pretrain process, then reruns
`run_all.sh configs/audio.yaml`, which retrains any backbone still missing.
Shards 0 and 2 still run the old code; if their knn check fails the same way,
the rerun covers it. Shard 1's remaining backbone (~5-6 h) should finish
before shards 0 and 2 finish their second ones, so pretraining still ends
around 1:30-2:30 am Thursday and the timeline is unchanged.

## Health check (Wed 18 Nov, 8:45 pm)

Images: the learning-rate ladder settled after two tune rounds (279 tune rows,
0 pending; "images tune done" at 8:25 pm). The final stage is now rerunning
with the new best settings; the budget check follows.

Audio: LeJEPA CNN seeds 0 and 1 finished (knn validation 0.926 and 0.928).
Shard 2 then ran out of GPU memory in the first steps of its ViT (seed 2)
while shard 0, still on the old code, ran its 1000-example knn check (14 GB).
To keep the old check from spiking again, shard 0's ViT (seed 1, 6.6k of 66k
steps, about 30 minutes lost) was stopped, and shards 0 and 2 were relaunched
on the fixed code via `/workspace/relaunch_02.sh`. All three processes now
train ViTs; `resume_audio.sh` still waits for every pretrain process before
the grid continues. Pretraining should end around 1:30 am Thursday.

## Health check (Wed 18 Nov, 10:45 pm)

Images: final stage rerunning; 18 new finals done in about 2.3 hours (the
100%-size runs, the longest), 93 pending. Smaller sizes run faster, so the
finals should end Thursday morning, with the 2x-step budget check (including
100%-size runs) after that, finishing around Thursday midday.

Audio: LeJEPA CNN seed 2 finished (knn 0.929; seeds 0-2: 0.926, 0.928,
0.929). The two remaining ViTs (seeds 1 and 2) are at 13.25k of 66k steps,
about 1.8 steps/s while sharing the GPU, so pretraining ends around 6-8 am
Thursday rather than 1:30 am. ViT seed 0's knn summary is still missing (that
backbone's checkpoint is fine); it can be computed separately.

## Health check (Thu 19 Nov, 12:45 am)

Images: 21 more finals in two hours (249 final rows, 72 pending), no errors.
Audio: the two ViTs sped up to about 5.9 steps/s once shard 1 finished and are
at 55.65k of 66k steps; pretraining should end around 1:15 am, then the audio
tune stage starts on its own.

## Health check (Thu 19 Nov, 2:45 am)

Images: finals done at 2:08 am (321 final rows, 0 pending); the 2x-step
budget check is running.

Audio: LeJEPA pretraining done at 1:14 am. ViT seeds 1 and 2 knn validation
are below; ViT seed 0 has no summary (its knn check hit the OOM), to compute
separately. The tune stage started: 3 rows done, 195 pending.
- ViT seed 1: knn 0.923
- ViT seed 2: knn 0.920

## Images complete (Thu 19 Nov, 3:26 am)

`images budget done` at 3:26 am. All CSVs and history were copied back and
the row counts match the pod (tune 279, final 321, budget 8 rows; history 99 /
111 / 8 lines). The SSH host then stopped answering partway through copying
the raw logs (17 log files copied; the rest are still on the pod volume).
The pod `cuslyj7h4randg` is stopped (EXITED) and only its disk bills now.

### Plateau rule

Neither target needed replacing: the CNN with augmentation gains 0.118 from
10% to the next size and 0.078 from 50%, both well above 0.01. Targets:
0.443 (10%) and 0.678 (50%). They moved slightly from the first pass (0.430,
0.666) because finals now use the ladder-extended best settings.

### Exchange rates (seed-bootstrap 95% intervals)

| Comparison | 10% target | 50% target |
|---|---|---|
| CNN vs ViT, no aug | 1.79x [1.78, 1.80] | undefined |
| CNN vs ViT, with aug | 2.61x [2.53, 2.70] | more than 2.0x |
| CNN vs MLP, with aug | more than 10x | more than 2.0x |
| Augmentation, CNN / ViT | 4.17x / 2.86x | — |
| LeJEPA, CNN / ViT | 3.46x / 6.06x | 1.21x / more than 1.9x |
| IN-21k start, ViT | 12.1x | more than 9.2x |
| IN-1k start, ViT | 16.4x [15.8, 17.2] | more than 10.1x |
| IN-1k start, CNN | 4.00x [3.91, 4.12] | 2.89x |
| CNN bias, LeJEPA for both | 1.49x [1.37, 1.64] | 1.28x [1.10, 1.43] |
| CNN bias, IN-1k start for both | 0.64x [0.61, 0.66] | 0.57x [0.54, 0.60] |
| CNN lead over best in-domain ViT | 0.43x | 1.06x [0.91, 1.15] |

Main new result: with matched ImageNet-1k pretraining (ResNet-18 a1 vs
DeiT-Tiny), the ViT needs only about 0.6x the CIFAR-100 labels the CNN does.
The CNN's in-domain advantage reverses once both start from the same
outside data. Error-rate ratios agree at every size: the IN-1k ViT's error
is 0.86-0.91x the IN-1k CNN's (0.176 vs 0.194 at 100%). Caveat: the
pretrained models differ in pretraining recipe and parameter count, so this
says "a standard pretrained ViT-Tiny beats a standard pretrained ResNet-18
here", not that the bias is worthless.

### Saturation

E against target (min to max over every reachable level): CNN vs ViT with aug
1.9-3.0x, rising toward higher targets; with LeJEPA for both, flat at
1.3-1.6x; with IN-1k for both, flat at 0.57-0.71x. Error-rate ratios for CNN
vs ViT with aug rise from 1.05 (1%) to 1.54 (100%), so the CNN lead grows with
data in error terms too; it is not a ceiling artefact.

### Budget check (2x steps, seed 0)

Too short (validation gain of 0.01 or more) in 2 of 8 cases:
- CNN + aug at 10%: +0.024 validation (+0.030 test). This is the reference
  that sets the 10% target, so the CNN is undertrained there; CNN-advantage
  rates at the 10% target are, if anything, underestimates.
- ViT, no aug, at 100%: +0.015 validation. Only matters near the ViT's
  no-aug ceiling.
The other six moved by less than 0.01, four of them downward. Open question
for Darcy: rerun the small-size CNN+aug finals with a longer budget, or report
the rates as conservative.

## Health check (Thu 19 Nov, 6:45 am): audio is slower than planned

Audio tune: 18 of 198 base jobs done in 5.5 hours (184 pending including the
first ladder rungs), no errors. All 18 are CNN runs at 100% (40k steps), each
taking about 52 minutes with three sharing the GPU, so about 17 GPU-minutes per
full-size run. The CNN keeps full spectrogram resolution (stride 1, no
max-pool), which makes it expensive.

Rough remaining work, in full-size-run equivalents, assuming the ViT and MLP
are no slower than the CNN (the MLP will be faster):
- tune: about 74 more (anchors 100%, 10%, 1% = 40k + 13.3k + 2k steps);
- ladder extensions: about 30 (images added roughly 40%);
- finals: 11 configurations x 3 seeds x 131k steps across sizes = about 109;
- budget check: about 11.
That is roughly 220 runs x 17 min = about 60 hours, so the audio grid would
end around Sunday midday Adelaide at about $40 more, against the earlier
"Thursday evening" estimate (which wrongly assumed image-like run times).
Flagged to Darcy with options.

## Audio pod stopped (Thu 19 Nov, about 7:45 am)

At the 8:43 am check the audio pod `f5pskuofih3ygv` was EXITED with no
runtime. I did not stop it. Hourly billing shows GPU charges ending about
15 minutes into the 7:30-8:30 am hour (UTC 21:00-22:00), the same pattern as
the earlier balance stop on the images pod. A restart attempt was blocked by
the permission classifier, so the pod stays stopped until Darcy restarts it or
allows the start.

State at the stop: audio tune had 18 rows at the 6:45 am copy (all CNN 100%
runs; the copy on the Mac is from then); possibly a few more finished on the
pod before the stop and are on its volume. The grid resumes from its CSVs, so
a restart loses only the runs in flight (up to about 50 minutes each).
Neither pod is running, so the 2-hourly health-check cron was deleted as its
instructions say; it is recreated when the audio pod restarts.

To resume: start the pod (same machine, so the GPU may be taken), update
`scratchpad/pod_audio_ssh.sh` with the new port, then run
`BIAS_WORKERS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
scripts/run_all.sh configs/audio.yaml` detached; pretraining is skipped
because every LeJEPA checkpoint exists.

## Audio dropped; hyperspectral next (Thu 19 Nov, morning)

Decision (Darcy): skip audio and go straight to hyperspectral. Audio becomes
a stated limitation in the write-up. What remains of it: LeJEPA knn scores
(CNN 0.926/0.928/0.929, ViT 0.923/0.920 for seeds 1-2) and 18 tune runs at
100% (CNN none/aug/lejepa, val 0.96-0.98), in `results/pod_audio/`. The audio
pod `f5pskuofih3ygv` was deleted (Darcy's standing rule: delete pods not in
use); its LeJEPA checkpoints went with it.

Hyperspectral changes, made before any hyperspectral result:
- `budget_check` added to both scene configs (CNN and ViT, none and aug, at
  20 px/class and all, 2x steps), matching images.
- New `leakage` stage: CNN+aug and MLP (no aug) at 20 px/class and all,
  every seed, at their final settings, rerun on a random per-pixel split
  (no blocks, no buffer); `analyse` writes `leakage_<scene>.csv` with the
  block-vs-random gap. The MLP mirrors the per-pixel model of the cited
  2026 study (36.9-point gap on Pavia University).
- No ImageNet-1k pair: the hyperspectral CNN has no ImageNet weights.

Mac smoke test (Pavia, CNN+aug, 20 px/class, 300 steps): block split
22,221/6,522/6,427 pool/val/test pixels, test 0.887; random split
25,664/8,557/8,555, test 0.929.

Throughput on the Mac M5 (Pavia, all): CNN 3.7 steps/s (GPU-bound; same with
aug), ViT 16.9, MLP 83. About 5.4M steps per scene (tune ~3.3M with ladder,
finals ~1.4M, LeJEPA, budget, leakage), CNN about 30% of it, so the Mac
would need well over a week. On an RTX 5090, guessing 10x the M5, both
scenes take roughly 25-35 h, about $17-25; to be measured in the first
minutes on the pod and requoted.

## Scope cut to images only (Thu 19 Nov)

Decision (Darcy): no hyperspectral runs either; the study reports CIFAR-100
only. Audio and hyperspectral become stated limitations and future work. The
hyperspectral code (block-and-buffer splits, leakage stage, budget check)
stays in the repo, tested but unrun.

Consequences:
- The CNN+aug budget shortfall at 10% is not rerun; CNN-advantage rates at
  the 10% target are reported as conservative.
- The images pod `cuslyj7h4randg` was deleted (no compute planned). All
  CSVs and validation history were already copied and checked; only some
  raw training logs were lost with it. No pods remain from this study.
- Remaining work: the analysis note and the blog draft, both on CIFAR-100.
