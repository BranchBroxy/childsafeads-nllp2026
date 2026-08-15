# ChildSafeAds 2026 — System Design Report

**Team Nürnberg NLP** · ChildSafeAds Shared Task @ [NLLP](https://nllpw.org/), EMNLP 2026 · Codabench account `broxy`

Our system is an ensemble of error-independent voters, built on the design we used for
GermEval 2026 and PsyDefDetect. Every voter is one point in a grid spanned by three axes:
**which model**, **which adaptation method**, and **which class scope** it trains on. This
report introduces the three axes first, then the ensemble that assembles them, then the
concrete system we submitted.

It contains **no task data** — see [DATA.md](DATA.md) for the licence and ethics terms.

---

## 1 · Class scope: what a voter is allowed to learn

The scope axis decides **which training data a voter sees**. It carries the main idea of
the system: three scopes, three different strengths, three different failure modes.

<p align="center"><img src="figures/fig-scopes.svg" width="100%"
  alt="Class scopes: generalist trains on all subtasks, specialist on one, minority-class specialist on one subtask with the most frequent labels dropped until the dropped share passes 50 percent"></p>

**Generalist (G).** Trains jointly on all three subtasks in a multi-task setup. The shared
signal is meant to build broad domain knowledge — patterns common to the subtasks that no
single one provides alone. Evaluation runs one subtask at a time, so the generalist carries
a subtask switch: the prepended prompt selects the subtask at inference.

**Specialist (S).** Trains only on the data of a single subtask. The narrowed signal fits
that subtask's label set more tightly. In practice it performs on a par with the
generalist; what it mainly contributes is a **different error profile** the vote can use.

**Minority-class specialist (MCS).** Like a specialist, it trains on **one subtask** — but
not on all of that subtask's labels. Sort the labels by frequency and drop them from the
top, one after another, **until the dropped labels together exceed 50 % of the mass**.
Everything below that cut is the MCS's label space, and it never sees the rest.

| Subtask | removed (cumulative share) | MCS trains on |
|---|---|---|
| **ST1** | `physical_goods` 47 % + `digital_content_or_services` 46 % = **93 %** | `physical_services`, `none`, `other` |
| **ST2** | `apps` 29 % + `hardware_electronics` 22 % = **51 %** | the remaining 10 categories |
| **ST3** | `misleading_claim` 54 % | the remaining 7 flags |

Because ST1 is single-label, removing a label removes its **instances** from training and
validation. ST2 and ST3 are multi-label, so only the **label columns** are dropped and the
instances stay.

Removing the dominant classes is meant to shift the model's errors elsewhere and free its
capacity for the fine distinctions between the rare classes — the ones a compliance
monitor actually cares about and where macro-F1 is won.

> **Reading MCS scores.** An MCS is scored only on its own labels, so its cross-validation
> number sits on a different scale and must never be compared directly against a G or S
> branch. On ST1 in particular, a full-label evaluation of an MCS is meaningless: it can
> never hit the two classes covering 93 % of instances.

---

## 2 · Adaptation methods: how a voter is trained

The method axis decides **what is trained**. These are the columns of our search grid.
They are **functional positions**, not method names: what does the same thing on a decoder
and on an encoder sits in the same column, which is what makes the grid comparable across
model families.

<p align="center"><img src="figures/fig-methods.svg" width="100%"
  alt="Adaptation methods as functional positions: OPRO, SFT, ClsHead/FT, Base heads and frozen-generalist heads"></p>

| Column | Decoder | Encoder | What it is |
|---|---|---|---|
| generative | `SFT` | — | The model is fine-tuned to **emit the label as text** from a task prompt containing the full category definitions. Encoders do not generate, so this column is structurally empty for them. |
| trained + heads | `ClsHead` | `FT` | Decoder: a two-layer head on the last-token hidden state of the QLoRA-adapted model, trained with focal loss and inverse-frequency class weights. Encoder: **full** fine-tuning with heads. Same function, different mechanics. |
| frozen base + head | `Base-LR` | `Base-LR` | The **untrained** backbone is frozen, its embeddings cached once, and a logistic regression fitted on them. The cheapest voter in the pool. |
| | `Base-ClsHead` | `Base-ClsHead` | Same cached embeddings, MLP head instead of logistic regression. |
| frozen generalist + head | `SFTf-LR` | `FTf-LR` | The **trained G generalist** is frozen and a logistic regression fitted on its embeddings. It inherits the generalist's domain knowledge without a second training run. |
| | `SFTf-ClsHead` | `FTf-ClsHead` | Same, MLP head. |
| prompted | `OPRO` | — | Prompt optimised by OPRO, no training. The bar the trained methods have to clear. |

Two things are worth spelling out because the naming hides them:

- **The `f` in `SFTf` / `FTf` means "frozen generalist".** These heads always sit on a **G**
  backbone; the scope in a voter's name refers to the **head**, not the backbone. A minority-class logistic regression on a frozen Phi-4 generalist is therefore a
  Phi-4 model in name only — what was trained is the head.
- **A logistic-regression head is fitted per label independently.** As a consequence a
  `…-LR-G` and a `…-LR-S-stX` voter produce, for that subtask, *the same classifier*. We
  measured it: identical on 50 of 50 folds for `FTf-LR`. They must never be used as two
  branches of one ensemble — that would be fake diversity.

Each cell of the resulting grid is one configuration. This is what the search space looks
like once trained:

<p align="center"><img src="figures/voter-grid.png" width="100%"
  alt="Cross-validation grid over model, method, scope and access level"></p>

*Every cell is one configuration's three-best-fold mean. The three schematics above
are TikZ; sources are in [`figures/src/`](figures/src) and
`bash figures/src/build.sh` rebuilds them as PDF (for the paper) and SVG (for this
page).*

---

## 3 · Models, and how a voter is named

The pool spans two model families, because they fail differently. Decoders can be tuned
generatively; encoders cannot generate at all but are structurally better suited to
classification, which is why we did not stay with decoders only.

| | model | size | role |
|---|---|---|---|
| decoder | [Ministral-8B](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410) | 8 B | 4-bit QLoRA |
| decoder | [Phi-4](https://huggingface.co/microsoft/phi-4) | 14 B | 4-bit QLoRA |
| encoder | [ettin-encoder-1b](https://huggingface.co/jhu-clsp/ettin-encoder-1b) | 1 B | full fine-tuning |
| encoder | [EuroBERT-610m](https://huggingface.co/EuroBERT/EuroBERT-610m) | 610 M | full fine-tuning |

Every point in the grid is one **configuration**, and its name states the three axes plus
the data it may read:

```
        phi4  -  SFTf-LR  -  MCS-st3  -  L1234  -ft
        ────     ───────     ───────     ─────  ───
        model    method      scope       levels  full text, no truncation
```

- **model** — the backbone, from the table above.
- **method** — how it was trained (§2). `SFTf-LR` is a logistic regression on the frozen
  generalist; `FTf-` is the same position on an encoder. The scope in the name always refers
  to the **head**, never to the frozen backbone underneath it.
- **scope** — what it was allowed to learn (§1). `G` needs no subtask, `S` and `MCS` name
  theirs.
- **levels** — which access levels the input contains, e.g. `L1234` is everything,
  `L1` the transcript alone.

The axis labels in the grid figure above read the same way.

---

## 4 · From cross-validation to a branch

Every configuration is trained **five times**. The training split is cut into five folds
**f0–f4**, grouped by channel — sponsored segments from one creator share vocabulary,
product mix and disclosure habits, so a random split leaks all of it. Each run produces one
model **M0–M4**: **M**$_i$ trains on the four folds other than **f**$_i$ and is scored on
**f**$_i$, which it never saw and which is what names it.

A **voter** is one such fold-model, and its score on its own held-out fold is its
$F1_{cv}$ — the only honest number we have for it, since nothing else it was trained on can
be used to judge it.

Ranking a configuration's five folds by $F1_{cv}$ and keeping the **top three** does two
things at once. The **mean of those three** becomes the configuration's selection signal,
so configurations compete on where they are strong rather than on an average dragged down
by one hard fold. And exactly those three fold-models are what gets deployed: the weaker
two were trained on the same recipe and would add correlated noise rather than an
independent opinion. Those three voters together are a **branch**, and a branch casts three
votes.

<p align="center"><img src="figures/fig-cv5.svg" width="100%"
  alt="Cross-validation: five fold-models per configuration, each trained on four folds and
  scored on the held-out fifth; the three best folds are deployed as one branch"></p>

---

## 5 · From three branches to nine voters

One branch is one opinion, sampled three times. That is not an ensemble — three voters of
the same configuration agree far too often to arbitrate anything. What makes the vote work
is that the three branches are chosen to be **structurally dissimilar**: they differ in
class scope (§1), and in practice also in method and backbone (§2), so their mistakes are
not the same mistakes.

An **ensemble** is therefore three branches — one **G**, one **S**, one **MCS** — nine
voters in total, deciding every instance by **plain strict majority**: a label fires when
more than half the votes cast carry it.

<p align="center"><img src="figures/fig-ensemble.svg" width="100%"
  alt="Nine-voter ensemble: three branches of three voters each, decided by plain strict majority"></p>

**Fold coverage.** The three fold sets are chosen so that **all five folds appear across
the nine voters**. No slice of the training data goes unused, and the out-of-fold estimate
covers every training instance rather than a convenient subset.

**Voting rules.** ST1 breaks ties toward the majority class and never predicts `other`
(0.07 % of the corpus — under a macro-F1 that averages over *occurring* labels, predicting
an absent class costs twice). ST2 is never empty. ST3 enforces the taxonomy: `no_flag` and
`insufficient_context` are exclusive against any real flag, and `undisclosed_advertising` /
`inadequate_disclosure` are mutually exclusive. The official checker does not test the last
two constraints; ours does.

**Why an MCS branch cannot do harm.** It contributes three of nine votes and never predicts
the majority labels. While the G and S branches agree on a majority label they already hold
six votes and win. The MCS can only move the decision once the other two split — the
uncertain cases where a sharper view of the rare classes helps. The gate is emergent; it
needs no threshold.

---

## 6 · Submitted systems

Five submissions are allowed. Every one of them fills the same three slots per subtask —
one **G**, one **S**, one **MCS** — and which model or method fills which slot is free.

### Submission 1

| | **G** | **S** | **MCS** |
|---|---|---|---|
| **ST1** | Phi-4 · SFT · L1234 | ettin-1b · FT · L1234 | Phi-4 frozen + LR · **L1** |
| **ST2** | Phi-4 · SFT · L1234 | ettin-1b · FT · L1234 | Phi-4 frozen + LR · L1234 |
| **ST3** | Phi-4 · SFT · L1234 | ettin-1b · FT · L1234 | Phi-4 frozen + LR · L1234 |

Two backbones carry all 27 fold-models: Phi-4 for the generative and the frozen-head
branch, ettin-encoder-1b for the specialist. Trained on the training split alone. Eight of
the nine voters read L1234; the ST1 minority specialist reads the transcript alone.

### Submission 2

The same three slots, refit over five folds of **train + dev**. Only ST3's generalist
differs, reading L12 instead of L1234.

### Submissions 3–5

*To be filled as we submit.*

### Results

| # | System | Levels | **test mean** | ST1 | ST2 | ST3 | ST3-family |
|---|---|---|---|---|---|---|---|
| 1 | trained on train | 1–4 | *pending* | | | | |
| 2 | refit on train + dev | 1–4 | *pending* | | | | |
| 3 | — | | | | | | |
| 4 | — | | | | | | |
| 5 | — | | | | | | |

Branches are chosen on cross-validation only; the development set is read once afterwards
as a transfer check, never as a selection criterion. Submission 1 reaches **0.7683** there.

---

## 7 · What it costs to run

A submission run produces labels for the 503 test instances, and the ensemble does that
once per branch and fold. We measured one pass of each branch type and normalised to
seconds per instance, because the three runs cover different numbers of instances.

| Branch | what one pass does | **per instance** |
|---|---|---|
| **G** — generative | Phi-4 decodes the label as text | **3.02 s** |
| **S** — encoder | one forward through ettin-1b | 0.29 s |
| **MCS** — frozen head | one forward through frozen Phi-4, then a logistic regression | 0.19 s |

The generative branch is **sixteen times** more expensive per instance than a frozen head on
the same backbone — autoregressive decoding against a single forward pass. That, not model
size, is what dominates the bill.

Rolled up for one submission over the 503 test instances:

| | one voter | one branch (3 folds) |
|---|---|---|
| G | 25.3 min | 76.0 min |
| S | 2.4 min | 7.2 min |
| MCS | 1.6 min | 4.7 min |
| **one subtask** (three branches, nine voters) | | **88 min** |
| **whole system** (three subtasks) | | **≈ 4.4 h** |

Scaled to all 3,360 instances, which is what the submission form asks for: **≈ 29
GPU-hours**, of which 25 sit in the generative branch. Training the search grid cost far
more; this figure is inference only.

Measured on single A100-SXM4 and H200 cards, 4-bit quantised backbones, batch sizes from
`get_vram_batch_config`. Figures are indicative rather than a benchmark: they include model
loading and were taken on a shared cluster.

---

## 8 · What each data level buys

The task ships four cumulative access levels ordered by collection cost. **The grid was
trained at every level, not just at the top:** each of L1, L12, L123 and L1234 carries
170–174 configurations with all five folds complete, spanning all four models and all nine
methods. The ladder therefore compares like with like rather than a full grid against a
thin one.

**How each rung is measured.** For a given cap we run the same selection as everywhere else
— best G, best S, best MCS by three-best-fold cross-validation — but restricted to voters
that read no level above the cap. The resulting nine-voter system is then scored **on the
development set**, which no voter has seen. So the ladder is a comparison of complete
systems at each data cost, not of individual voters.

| System | mean | ST1 | ST2 | ST3 | Δ over previous rung |
|---|---|---|---|---|---|
| L1 — transcript only | 0.6103 | 0.6492 | 0.6013 | 0.5805 | — |
| ≤ L12 — + video context | 0.7292 | 0.7966 | 0.7214 | 0.6696 | **+0.119** |
| ≤ L123 — + channel | 0.7348 | 0.8181 | 0.7092 | 0.6770 | +0.006 |
| ≤ L1234 — + product page | **0.7675** | 0.8459 | 0.7795 | 0.6770 | +0.033 |

Against the median added tokens per instance (L2 ≈ 290, L3 ≈ 7, L4 ≈ 375):

- **Level 2 is the whole game.** Title, description and the platform's own paid-promotion
  flag buy +0.119 for about 290 tokens. Nothing else comes close.
- **Level 3 is free and worthless.** The channel name costs seven tokens and moves the mean
  by +0.006, inside our noise band. A monitoring system can skip it.
- **Level 4 pays only on ST1 and ST2.** The product page says what is being sold and in
  which category. It does not help with compliance flags: **ST3 is saturated at L12**.

For an authority weighing crawl cost against accuracy: fetch video metadata always, skip
channel context, fetch product pages only if commercial type and product category matter.

Two caveats on the record. The ladder was measured before the full-text retraining
described in §9, so every rung sits on capped inputs; the L1 rung suffers most, because
its scaffold ate a larger share of a short sequence. The gaps are real and the ordering has
held up in everything we have measured since, but the exact sizes are not settled, and we
would not want the +0.119 quoted to three decimals. Re-running the ladder on the retrained
grid is the first thing we will add here.

---

## 9 · Negative results

- **Input truncation.** Our first training wave capped inputs at a sequence length that
  silently cut **16.9 % of product pages** and 7.2 % of descriptions, with the truncation
  side differing between training and inference. We found it by auditing token lengths, not
  by seeing a bad score. After verifying that all 3,360 instances fit in 8,192 tokens at
  every level, we retrained. Measure your truncation before trusting a level ablation.
- **Data augmentation did nothing.** Paired over 68 recipes it moved the mean by +0.007 —
  inside noise. Dropped.
- **Prompting has a ceiling well below training.** OPRO reached ST2 0.59 against 0.89 for a
  trained head, and ST3 0.53 against 0.67.

---

## 10 · Scope, reproducibility, ethics

This repository documents the system. It deliberately contains no task data, no
per-instance predictions (they are tied to identifiable channels), and no trained adapter
weights. Curated source for the metric implementation, the voting rules, the
channel-disjoint split builder and the selection scripts will be added after one pass of
cleaning — the working tree carries internal cluster paths and account names that do not
belong in public.

**Bonus track.** We did not collect off-platform destination data. **Legal material:** we
used only the citations shipped in `legal_provisions.json` and retrieved nothing beyond
them.

The compliance labels are a **research benchmark, not legal advice and not a finding of
unlawfulness**. We do not re-identify creators, do not contact them, and report no result
that would mark an individual creator as non-compliant. All figures here are aggregate.
See [DATA.md](DATA.md).
