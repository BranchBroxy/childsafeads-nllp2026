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

## 3 · The nine-voter ensemble

Three terms build the system from a single vote up.

A **voter** is one model, trained in one method on one class scope, on four of five folds
and evaluated on the held-out fifth that names it.

A **branch** is three voters sharing model, method and scope, differing only in which fold
each held out — **the three folds on which that configuration scored highest**. A branch
therefore casts three votes.

An **ensemble** is three branches — nine voters — deciding every instance by **plain strict
majority** (> 50 % of votes cast).

<p align="center"><img src="figures/fig-cv5.svg" width="100%"
  alt="Cross-validation: five fold-models per configuration, each trained on four folds and
  scored on the held-out fifth; the three best folds are deployed as one branch"></p>

<p align="center"><img src="figures/fig-ensemble.svg" width="100%"
  alt="Nine-voter ensemble: three branches of three voters each, decided by plain strict majority"></p>

**Cross-validation.** Five folds over the training set, grouped by **channel**, with an
assertion that no channel appears in two folds. Sponsored segments from one creator share
vocabulary, product mix and disclosure habits; a random split leaks all of it. Each
configuration is trained five times, giving five per-fold macro-F1 scores; the **mean of
the three best** is its selection signal and names the three voters it deploys.

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

## 4 · Submitted systems

The evaluation phase allows five submissions. Each is one nine-voter ensemble per subtask;
they differ in what the branches are allowed to be, which is what turns the five slots into
an experiment rather than five attempts at the same thing.

Two rules hold across all of them. Branches are chosen on **cross-validation only** — the
mean of each configuration's three best folds — and the development set is read once
afterwards as a transfer check, never as a selection criterion. And the three fold sets of a
subtask together cover all five folds.

### Slot 1 — one decoder, one encoder

The scope assignment is fixed to **SFT = G, SFTf = MCS, FT = S** under the constraint that
the whole system runs on **one decoder and one encoder**. Without that constraint the
ensemble needs a second decoder in memory and the frozen-generalist branch stops being
cheap.

| | **G** — generalist | **S** — specialist | **MCS** — minority-class |
|---|---|---|---|
| **ST1** | Phi-4 · SFT · L1234 | ettin-1b · FT · L1234 | Phi-4 · SFTf-LR · **L1** |
| **ST2** | Phi-4 · SFT · L1234 | ettin-1b · FT · L1234 | Phi-4 · SFTf-LR · L1234 |
| **ST3** | Phi-4 · SFT · L1234 | ettin-1b · FT · L1234 | Phi-4 · SFTf-LR · L1234 |

Two backbones carry all 27 fold-models; the adapters are small and hot-swappable on a
loaded base, so the memory cost is two models, not nine. Trained on the **training split
alone** (2,353 instances) — the development set is in no voter's training data, which is
what makes its dev score a transfer check.

Eight of the nine voters read L1234. The exception is the ST1 minority specialist, which
reads the **transcript alone** and still earns its slot — the one place in the system where
the cheapest data level wins.

### Slot 2 — the same recipe refit on train + dev

Identical selection procedure, run again over five folds of **train + dev** (2,857
instances). It cannot be checked on dev by construction — dev is training data for it —
which is exactly why it is worth a slot of its own: comparing slots 1 and 2 on the test set
is the only way to answer whether the refit transfers.

| | **G** | **S** | **MCS** |
|---|---|---|---|
| **ST1** | Phi-4 · SFT · L1234 | ettin-1b · FT · L1234 | Phi-4 · SFTf-LR · L1234 |
| **ST2** | Phi-4 · SFT · L1234 | ettin-1b · FT · L1234 | Phi-4 · SFTf-LR · L1234 |
| **ST3** | Phi-4 · SFT · **L12** | ettin-1b · FT · L1234 | Phi-4 · SFTf-LR · L1234 |

### Slots 3–5

*To be filled as we submit.* The open questions we would most like answered on the test set
are what a transcript-only system reaches, what the L12 rung reaches, and whether dropping
the one-decoder constraint buys anything.

### Results

Scores as reported by the official scorer. Development-set figures come from our own
implementation, verified against the leaderboard to four decimal places (Δ = 0.0000).

| # | System | Levels | dev mean | **test mean** | ST1 | ST2 | ST3 | ST3-family |
|---|---|---|---|---|---|---|---|---|
| 1 | one decoder + one encoder, train only | 1–4 | 0.7683 | *pending* | | | | |
| 2 | same recipe, refit on train + dev | 1–4 | — | *pending* | | | | |
| 3 | — | | | | | | | |
| 4 | — | | | | | | | |
| 5 | — | | | | | | | |

---

## 5 · What it costs to run

Inference over all 3,360 instances needs, per subtask, three generative decoder passes,
three decoder passes to embed for the frozen head, and three encoder passes — 27 passes in
total, on two loaded backbones.

| | passes | backbone |
|---|---|---|
| generative branch (G) | 9 | Phi-4, 14 B, 4-bit |
| frozen-head branch (MCS) | 9 | Phi-4, frozen, one forward per subtask prompt |
| encoder branch (S) | 9 | ettin-encoder-1b |

*Measured wall-clock figures follow; the number quoted in our submission form is the
inference cost only, not the cost of training the search grid, which is far larger.*

---

## 6 · What each data level buys

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
described in §7, so every rung sits on capped inputs; the L1 rung suffers most, because
its scaffold ate a larger share of a short sequence. The gaps are real and the ordering has
held up in everything we have measured since, but the exact sizes are not settled, and we
would not want the +0.119 quoted to three decimals. Re-running the ladder on the retrained
grid is the first thing we will add here.

---

## 7 · Negative results

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

## 8 · Scope, reproducibility, ethics

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
