# ChildSafeAds 2026 — System Design Report

**Team Nürnberg NLP** · ChildSafeAds Shared Task @ [NLLP](https://nllpw.org/), EMNLP 2026 · Codabench account `broxy`

The core system gives nine fold-models one vote each. They are picked to be **structurally dissimilar** —
different class scope, different training method, different backbone — on the assumption
that models built differently fail differently, so a majority over them is more robust than
any single one.

**Sections 1–5** describe how a voter is built and how nine of them become a decision.
**Section 6** lists what we submitted, **7** what those systems cost to run and **8** what
each level of data access buys — the question the task is built around. **9** collects what
went wrong and what our numbers cannot tell you; **10** states what this repository leaves
out and why.

No task data here; see [DATA.md](DATA.md) for licence and ethics terms.

### Contents

| Section | What it covers |
|---|---|
| **[1 · Class scope](#1--class-scope)** | which labels a voter is trained on: generalist, specialist, minority-class specialist |
| **[2 · Adaptation methods](#2--adaptation-methods)** | what is trained: generative, head on a tuned backbone, head on a frozen one, prompt only |
| **[3 · Models and naming](#3--models-and-naming)** | the four backbones, and how to read a configuration's name |
| **[4 · Cross-validation and branches](#4--cross-validation-and-branches)** | five channel-disjoint folds per configuration; the best three become a branch |
| **[5 · The nine-voter ensemble](#5--the-nine-voter-ensemble)** | three dissimilar branches, plain strict majority, and the taxonomy rules on top |
| **[6 · Submitted systems](#6--submitted-systems)** | what each submission contains, and the results table |
| **[7 · What it costs to run](#7--what-it-costs-to-run)** | seconds per segment and GPU-hours per corpus, measured and reconstructed |
| **[8 · What each data level buys](#8--what-each-data-level-buys)** | the level ladder read against collection cost |
| **[9 · Negative results and limits](#9--negative-results-and-limits)** | a truncation bug found by audit, two things that did not work, and what our numbers cannot settle |
| **[10 · Scope and ethics](#10--scope-and-ethics)** | what this repository deliberately does not contain |

---

## 1 · Class scope

Which training data a voter sees. Three scopes, three different failure modes.

<p align="center"><img src="figures/fig-scopes.svg" width="100%"
  alt="Class scopes: generalist trains on all subtasks, specialist on one, minority-class specialist on one subtask with the most frequent labels dropped until the dropped share passes 50 percent"></p>

**Generalist (G)** trains jointly on all three subtasks; the prompt selects the subtask at
inference. **Specialist (S)** trains on one subtask only — on a par with the generalist in
accuracy, but with a different error profile the vote can use.

**Minority-class specialist (MCS)** trains on one subtask but not on all its labels: sort
the labels by frequency and drop from the top **until the dropped labels exceed 50 % of the
mass**. What remains is its label space. Removing the dominant classes frees capacity for
the rare ones — where macro-F1 is won.

| Subtask | dropped | MCS keeps |
|---|---|---|
| **ST1** | `physical_goods` 47 % + `digital_content_or_services` 46 % = **93 %** | 3 of 5 labels |
| **ST2** | `apps` 29 % + `hardware_electronics` 22 % = **51 %** | 10 of 12 |
| **ST3** | `misleading_claim` 54 % | 7 of 8 |

ST1 is single-label, so dropping a label drops its **instances**; ST2 and ST3 are
multi-label, so only the **label columns** go.

> An MCS is scored only on its own labels, so its cross-validation number sits on a
> different scale and must never be compared against a G or S branch.

---

## 2 · Adaptation methods

What is trained. These are the columns of the grid — **functional positions**, not method
names: what does the same thing on a decoder and on an encoder sits in the same column.

<p align="center"><img src="figures/fig-methods.svg" width="100%"
  alt="Adaptation methods as functional positions: OPRO, SFT, ClsHead/FT, Base heads and frozen-generalist heads"></p>

| Column | Decoder | Encoder | What it is |
|---|---|---|---|
| generative | `SFT` | — | Fine-tuned to **emit the label as text**. Encoders do not generate. |
| trained + head | `ClsHead` | `FT` | Decoder: two-layer head on the QLoRA-adapted model, focal loss, inverse-frequency weights. Encoder: full fine-tuning. |
| frozen base + head | `Base-LR` · `Base-ClsHead` | same | Untrained backbone frozen, embeddings cached once, LR or MLP on top. The cheapest voters. |
| frozen generalist + head | `SFTf-LR` · `SFTf-ClsHead` | `FTf-…` | The **trained G generalist** frozen, LR or MLP on its embeddings — its domain knowledge without a second training run. |
| prompted | `OPRO` | — | Optimised prompt, no training. The bar the trained methods must clear. |

Two things the naming hides:

- **The `f` means "frozen generalist".** These heads always sit on a **G** backbone; the
  scope in a name refers to the **head**, not the backbone under it.
- **An LR head is fitted per label independently**, so `…-LR-G` and `…-LR-S-stX` are *the
  same classifier* for that subtask — identical on 50 of 50 folds for `FTf-LR`. Using both
  as branches would be fake diversity.

<p align="center"><img src="figures/voter-grid.png" width="100%"
  alt="Cross-validation grid over model, method, scope and access level"></p>

*Each cell is one configuration's three-best-fold mean; grey is not trained, a yellow frame
means fewer than five folds are finished and the number is provisional.*

---

## 3 · Models and naming

Two model families, because they fail differently: decoders can be tuned generatively,
encoders cannot generate but are structurally better at classification.

| | model | size | role |
|---|---|---|---|
| decoder | [Ministral-8B](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410) | 8 B | 4-bit QLoRA |
| decoder | [Phi-4](https://huggingface.co/microsoft/phi-4) | 14 B | 4-bit QLoRA |
| encoder | [ettin-encoder-1b](https://huggingface.co/jhu-clsp/ettin-encoder-1b) | 1 B | full fine-tuning |
| encoder | [EuroBERT-610m](https://huggingface.co/EuroBERT/EuroBERT-610m) | 610 M | full fine-tuning |

A configuration's name states the three axes plus the data it may read:

```
        phi4  -  SFTf-LR  -  MCS-st3  -  L1234
        ────     ───────     ───────     ─────
        model    method      scope       levels
```

`G` needs no subtask; `S` and `MCS` name theirs. `L1234` is everything, `L1` the transcript
alone. The axis labels in the grid above read the same way.

---

## 4 · Cross-validation and branches

Every configuration is trained **five times**. The training split is cut into five folds
**f0–f4** grouped by channel — segments from one creator share vocabulary, product mix and
disclosure habits, so a random split leaks all of it. Run *i* trains on the four folds other
than **f**$_i$ and is scored on **f**$_i$, giving model **M**$_i$.

A **voter** is one such fold-model; its score on its own held-out fold is its $F1_{cv}$.
The mean of the **three best folds** is the configuration's selection signal. Deployment
keeps the strongest two and then the best available third fold subject to one ensemble-level
constraint: the three branches together must cover **f0–f4**. This costs very little score,
prevents a whole data slice from disappearing from the ensemble, and leaves three
fold-models per configuration. Those three form a **branch** and cast three votes.

<p align="center"><img src="figures/fig-cv5.svg" width="100%"
  alt="Cross-validation: five fold-models per configuration, each trained on four folds and
  scored on the held-out fifth; the three best folds are deployed as one branch"></p>

---

## 5 · The nine-voter ensemble

One branch is one opinion sampled three times — three voters of the same configuration
agree too often to arbitrate anything. The vote works because the three branches are
**structurally dissimilar**: different class scope (§1), in practice also different method
and backbone (§2), so their mistakes differ.

An **ensemble** is three branches and nine voters, deciding by **plain strict majority**:
a label fires when more than half the votes cast carry it. G × S × MCS was the initial
design, but the evaluation systems treat scope as a diversity axis rather than a fixed
template: a trio may instead be G × 2S or 2G × S when full-label branches rank more
reliably. Every selected trio spans at least two backbones and both generative and
head-based adaptation.

<p align="center"><img src="figures/fig-ensemble.svg" width="100%"
  alt="Nine-voter ensemble: three branches of three voters each, decided by plain strict majority"></p>

**Fold coverage.** The three fold sets together cover all five folds, so no slice of the
training data goes unused.

**Voting rules.** ST1 breaks ties toward the majority class; the main systems also suppressed
`other` because it forms only 0.07 % of the labelled corpus, a decision the hidden test later
showed to be unsafe (§9). ST2 is never empty. ST3 enforces the taxonomy: `no_flag` and
`insufficient_context` are exclusive against any real flag, and `undisclosed_advertising` /
`inadequate_disclosure` are mutually exclusive. The official checker tests neither of the
last two; ours does.

**How an MCS is bounded.** It holds three of nine votes and never predicts the majority
labels. While two full-label branches agree they already have six votes and the MCS cannot
overrule them; it can only arbitrate their disagreements. That structural bound does not
guarantee a gain: ST2 benefited slightly, while ST1's tiny minority validation set made MCS
selection unstable (§9).

---

## 6 · Submitted systems

The evaluation phase allowed five uploads, and all five were used. Submissions 1–3 were
trained on the official training split; Submission 4 refitted the selected configurations
on training plus development data. Submission 5 is an exact field-level composition of
already generated predictions: ST1 and ST3 from Submission 4, ST2 from Submission 3.

| # | Codabench ID | Uploaded (2026) | File | Purpose |
|---|---:|---|---|---|
| 1 | 890039 | 15 Aug, 14:32 | `ens9v_sub1_TEST.zip` | G × S × MCS baseline |
| 2 | 890906 | 16 Aug, 13:47 | `ens9v_top3cv_div_TEST.zip` | strongest diverse CV selections |
| 3 | 891574 | 17 Aug, 10:21 | `ens9v_sub3v2_TEST.zip` | minority-label intervention |
| 4 | 892825 | 18 Aug, 13:21 | `ens9v_sub4_traindev_TEST.zip` | train+development refit |
| 5 | 892837 | 18 Aug, 13:31 | `ens9v_bestof_sub5st1_sub3st2_sub5st.zip` | best scored subtask predictions |

### Submission 1 — G × S × MCS baseline

| | **G** | **S** | **MCS** |
|---|---|---|---|
| **ST1** | Phi-4 SFT, f2/4/0, L1234 | ettin-1b FT, f2/4/1, L1234 | Phi-4 frozen + LR, f4/3/0, L1 |
| **ST2** | Phi-4 SFT, f4/2/1, L1234 | ettin-1b FT, f4/1/0, L1234 | Phi-4 frozen + LR, f3/2/1, L1234 |
| **ST3** | Phi-4 SFT, f0/2/1, L1234 | ettin-1b FT, f4/2/1, L1234 | Phi-4 frozen + LR, f3/2/0, L1234 |

This is the original plain-majority system. Two backbones carry all 27 fold-models; only
the ST1 minority specialist is restricted to transcript-only input.

### Submission 2 — top-CV selections with diversity

Each subtask uses three branches selected by their mean over the strongest CV folds, while
requiring at least two backbones and both generative and head-based prediction.

| | Branch 1 | Branch 2 | Branch 3 |
|---|---|---|---|
| **ST1** | Phi-4 SFT-S, f3/4/0 | Ministral head-S capped, f2/3/4 | Ministral head-S full text, f2/3/1 |
| **ST2** | Ministral head-S, f4/1/3 | Phi-4 head-S, f4/1/2 | Phi-4 SFT-S, f4/1/0 |
| **ST3** | Phi-4 frozen + LR-S, f1/4/2, L123 | Phi-4 SFT-G, f0/2/1, L1234 | ettin-1b frozen + head-S, f1/2/3, L12 |

Unless shown otherwise, branches read L1234. The development set was used only as a
post-selection transfer check.

### Submission 3 — minority-label intervention

ST1 switches to a Phi-4 head-G, Phi-4 SFT-S and Ministral head-S trio. ST2 retains
Submission 2's nine voters and adds three folds of a Phi-4 MCS head as a conservative
per-label cast; ST3 retains Submission 2's voters but lowers the firing threshold from
five to four votes. The intervention gives the best ST2 and ST3-family scores, while its
development-selected ST1 transfers poorly to the hidden test set.

| | Branch composition | Decision rule |
|---|---|---|
| **ST1** | Ministral head-S f3/0/4 · Phi-4 head-G f4/2/3 · Phi-4 SFT-S f3/4/2 | nine-vote majority with rare-label safeguards |
| **ST2** | Submission 2 core + Phi-4 MCS head f4/1/2 | conservative MCS cast over the nine core votes |
| **ST3** | Submission 2 core | each label fires at four of nine votes |

### Submission 4 — train+development refit

Every selected configuration is retrained in channel-disjoint five-fold CV over the
combined training and development pool. All branches read L1234, use no synthetic
augmentation, and each subtask's selected fold union covers f0–f4.

| | Branch 1 | Branch 2 | Branch 3 |
|---|---|---|---|
| **ST1** | Phi-4 frozen + LR-G, f0/2/4 | Phi-4 SFT-S, f1/2/4 | Ministral head-S, f2/3/4 |
| **ST2** | Ministral head-G, f0/2/4 | ettin-1b frozen + head-S, f0/1/4 | Ministral SFT-S, f0/1/3 |
| **ST3** | Phi-4 SFT-G, f1/2/4 | Phi-4 frozen + LR-S, f0/1/2 | ettin-1b frozen + head-S, f1/3/4 |

The refit produces the strongest ST1 and ST3 predictions, but loses substantially on ST2.

### Submission 5 — best-of scored subtasks

Submission 5 combines Submission 4's ST1, Submission 3's ST2 and Submission 4's ST3
predictions without changing a single label inside those fields. This composition was
uploaded and scored as one valid system; it is not a hypothetical post-evaluation result.

### Results

| # | System | Levels | **test mean** | ST1 | ST2 | ST3 | ST3-family |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | G × S × MCS baseline | 1–4 | .6644 | .5944 | .8034 | .5954 | .6958 |
| 2 | top-CV selections, diversity rule | 1–4 | .6974 | .6205 | .8204 | .6512 | .7259 |
| 3 | minority-label intervention | 1–4 | .6688 | .5339 | **.8243** | .6483 | **.7281** |
| 4 | train+development refit, G × 2S | 1–4 | .6904 | **.6464** | .7719 | **.6530** | .7031 |
| **5** | **best-of: Sub4-ST1 + Sub3-ST2 + Sub4-ST3** | **1–4** | **.7079** | **.6464** | **.8243** | **.6530** | .7031 |

The official mean is the arithmetic mean of ST1, ST2 and ST3; ST3-family is reported for
diagnosis but does not enter it. Submission 5 therefore scores
**(.6464 + .8243 + .6530) / 3 = .7079**. Its ST3-family value follows Submission 4 because
both ST3 fields are identical.

---

## 7 · What it costs to run

The figures are inference-only A100-80GB equivalents. Submission 1 is measured end to end;
the remaining rows are reconstructed from measured A100 single-pass wall times and the
exact selected folds. The additional Ministral-SFT rate needed for Submissions 4 and 5 was
checked against the completed cluster banking logs (about 0.52 s per instance and fold).
Model loading and reusable frozen representations are included where the deployed branch
permits reuse; no paid API is used.

| Submission | ST1 (s/segment) | ST2 (s/segment) | ST3 (s/segment) | Total (s/segment) | All 3,360 instances |
|---|---:|---:|---:|---:|---:|
| **1** | 4.4 | 4.4 | 4.4 | 13.3 | **12.4 GPU-h** |
| **2** | 5.4 | 5.8 | 9.6 | 20.8 | **≈19.4 GPU-h** |
| **3** | 5.8 | 7.3 | 9.6 | 22.7 | **≈21.2 GPU-h** |
| **4** | 10.3 | 3.2 | 9.6 | 23.1 | **≈21.6 GPU-h** |
| **5** | 10.3 | 7.3 | 9.6 | 27.2 | **≈25.4 GPU-h** |

---

## 8 · What each data level buys

Four cumulative access levels, ordered by collection cost. **The grid was trained at every
level:** each of L1, L12, L123 and L1234 carries 170–174 complete configurations across all
four models and all nine methods, so the ladder compares like with like. Per rung we run the
same selection — best G, best S, best MCS by three-best-fold cross-validation, restricted to
voters reading no level above the cap — and score the resulting nine-voter system on the
development set.

| System | mean | ST1 | ST2 | ST3 | Δ |
|---|---|---|---|---|---|
| L1 — transcript only | 0.6103 | 0.6492 | 0.6013 | 0.5805 | — |
| ≤ L12 — + video context | 0.7292 | 0.7966 | 0.7214 | 0.6696 | **+0.119** |
| ≤ L123 — + channel | 0.7348 | 0.8181 | 0.7092 | 0.6770 | +0.006 |
| ≤ L1234 — + product page | **0.7675** | 0.8459 | 0.7795 | 0.6770 | +0.033 |

Against the median added tokens (L2 ≈ 290, L3 ≈ 7, L4 ≈ 375):

- **Level 2 is the whole game.** Title, description and the platform's paid-promotion flag
  buy +0.119 for about 290 tokens.
- **Level 3 is free and worthless.** Seven tokens, +0.006 — inside noise. Skip it.
- **Level 4 pays only on ST1 and ST2.** It says what is being sold; it does not help with
  compliance flags. **ST3 is saturated at L12.**

For an authority weighing crawl cost: fetch video metadata always, skip channel context,
fetch product pages only if commercial type and category matter.

*Caveat: the ladder was measured before the retraining in §9, so every rung sits on capped
inputs and L1 suffers most. The ordering has held in everything measured since, but the
exact gaps are not settled. Re-running it is the first thing we will add here.*

---

## 9 · Negative results and limits

- **Input truncation.** Our first wave silently cut **16.9 % of product pages** and 7.2 % of
  descriptions, with the truncation side differing between training and inference. Found by
  auditing token lengths, not by seeing a bad score. Measure your truncation before trusting
  a level ablation.
- **Data augmentation did nothing.** Paired over 68 recipes: +0.007, inside noise. Dropped.
- **Prompting has a ceiling well below training.** OPRO reached ST2 0.59 against 0.89, ST3
  0.53 against 0.67.
- **More labelled data did not improve every subtask.** The train+development refit raised
  test ST1 from .6205 to .6464 and ST3 from .6512 to .6530, but ST2 fell from .8204 to
  .7719. Refit quality and ensemble composition cannot be treated as separable choices.
- **Suppressing `other` was not safe on ST1.** It occurs in only 2 of 2,857 labelled
  training-plus-development instances, but the hidden test metric contained the class.
  Predicting it zero times fixes one of five macro-F1 terms at zero; post-hoc overrides were
  too brittle to solve the problem reliably.

What our numbers cannot settle:

- **The development set is small and noisy.** Split in half and correlated across
  configurations, its halves agree at *r* = 0.06 (ST1), 0.03 (ST2) and 0.37 (ST3). Differences
  below roughly 0.03 there are noise, which is why we never select on it.
- **An MCS on ST1 rests on few points.** After the majority classes are removed only about
  31 validation instances per fold remain, so its ranking is unstable even where its score
  looks high. Submission 3 confirmed the transfer failure: stronger development performance
  became our weakest test ST1.

---

## 10 · Scope and ethics

This repository documents the system. It contains no task data, no per-instance predictions
(they are tied to identifiable channels) and no trained adapter weights. Curated source for
the metric, the voting rules, the split builder and the selection scripts follows after one
pass of cleaning.

**Bonus track:** not attempted. **Legal material:** only the citations shipped in
`legal_provisions.json`.

The compliance labels are a **research benchmark, not legal advice and not a finding of
unlawfulness**. We do not re-identify creators, do not contact them, and report no result
marking an individual creator as non-compliant. All figures are aggregate. See
[DATA.md](DATA.md).
