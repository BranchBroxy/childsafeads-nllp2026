# ChildSafeAds 2026 — System Design Report

**Team Nürnberg NLP** · ChildSafeAds Shared Task @ [NLLP](https://nllpw.org/), EMNLP 2026 · Codabench account `broxy`

This repository is the system design report for our submission to the ChildSafeAds
shared task on commercial content in child-facing YouTube videos. It documents how the
system is built, how it was selected, and what each level of data access actually bought
us — the question the organisers put at the centre of the task.

It contains **no task data**. See [DATA.md](DATA.md) for why, and for the licence and
ethics commitments we work under.

---

## 1 · What the system is

For each subtask, nine fold-models vote by plain majority. The nine come from **three
deliberately dissimilar branches**, each contributing the three folds on which it scored
best in cross-validation:

| Branch | What it is | Scope in the submitted system |
|---|---|---|
| **SFT** | decoder, instruction-tuned generatively with QLoRA | **G** — one generalist covering all three subtasks |
| **SFTf** | the *same* decoder, frozen, with a light classification head | **MCS** — trained only on the subtask's minority classes |
| **FT** | encoder, fully fine-tuned | **S** — one specialist per subtask |

The branches are anti-correlated by construction: a generative decoder, a linear head on
frozen decoder states, and a discriminatively fine-tuned encoder fail in different ways.
That is the point — a majority vote only helps when its members are not copies of each
other.

Backbones: [Ministral-8B](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410),
[Phi-4 (14B)](https://huggingface.co/microsoft/phi-4),
[ettin-encoder-1b](https://huggingface.co/jhu-clsp/ettin-encoder-1b). All open weights.

**Voting rules.** Plain strict majority (> 50 % of votes cast). ST1 breaks ties toward the
majority class and never predicts `other` (0.07 % of the corpus — predicting it costs
twice under a macro-F1 that averages over *occurring* labels). ST2 is never empty. ST3
enforces the taxonomy: `no_flag` and `insufficient_context` are exclusive against any real
flag, and `undisclosed_advertising` / `inadequate_disclosure` are mutually exclusive. The
official checker does not test the last two constraints; ours does.

**Runtime cost.** Because the SFT and SFTf branches share a decoder backbone, the ensemble
needs two decoder models and one encoder in memory rather than nine independent models.
Adapters are small and hot-swappable on a loaded base.

---

## 2 · What each data level buys

The task ships four cumulative access levels ordered by collection cost: the segment
transcript (L1), video metadata including YouTube's paid-promotion flag (L2), channel
context (L3), and the crawled product page (L4). We trained the full ladder and measured
each rung on the development set, using the best nine-voter system we could build under
each cap:

| System | mean macro-F1 | ST1 | ST2 | ST3 | Δ over previous rung |
|---|---|---|---|---|---|
| L1 — transcript only | 0.6103 | 0.6492 | 0.6013 | 0.5805 | — |
| ≤ L12 | 0.7292 | 0.7966 | 0.7214 | 0.6696 | **+0.119** |
| ≤ L123 | 0.7348 | 0.8181 | 0.7092 | 0.6770 | +0.006 |
| ≤ L1234 | **0.7675** | 0.8459 | 0.7795 | 0.6770 | +0.033 |

Read against the token cost of each level (median added tokens per instance: L2 ≈ 290,
L3 ≈ 7, L4 ≈ 375), three findings stand out:

- **Level 2 is the whole game.** Roughly 290 extra tokens — title, description, and the
  platform's own paid-promotion flag — buy +0.119. Nothing else comes close.
- **Level 3 is free and worthless.** The channel name costs about seven tokens and moves
  the mean by +0.006, inside our noise band. A monitoring system can skip it.
- **Level 4 pays only on ST1 and ST2.** The product page tells you what is being sold and
  in which category. It does not help with compliance flags: **ST3 is saturated at L12** —
  our best ST3 branch uses no product page at all.

For an authority weighing crawl cost against accuracy, that is the practical shape of the
trade-off: fetch video metadata always, skip channel context, and fetch product pages only
if commercial type and product category matter to you.

One caveat we want on the record: our L1 systems were trained before we fixed an input
truncation bug (§4), so the bottom rung of this ladder is pessimistic. The gap between L1
and L12 is real but its exact size is not settled.

---

## 3 · How branches were selected — and why not on dev

The selection procedure is the part we would defend hardest.

**Cross-validation is channel-disjoint.** Five folds are built over the training set only,
grouped by channel, with an assertion that no channel appears in two folds. Sponsored
segments from one creator share vocabulary, product mix, and disclosure habits; a random
split leaks all of it and flatters every number.

**Selection uses only cross-validation.** For every configuration we take the mean of its
three best folds; per subtask we then take the best G, the best MCS, and the best S branch.
The development set is never consulted during selection.

That last rule was not a matter of taste. We measured the reliability of the development
set by splitting it in half and correlating the two halves across configurations:

| | ST1 | ST2 | ST3 |
|---|---|---|---|
| split-half correlation *r* | 0.06 | 0.03 | 0.37 |

With 504 instances, differences below roughly 0.03 on dev are noise. A search that ranks
candidates on dev is fitting that noise, and we could show it: freely searching fold-models
against dev produced systems that looked better on dev and were not. So dev is used for
exactly one thing — a transfer check, read once, after selection is frozen.

**The metric is verified, not assumed.** The official scorer is not public, so we
reimplemented it and calibrated against the leaderboard: our local score matched the
returned score to four decimal places across every column (Δ = 0.0000). Every number in
this report comes from that implementation.

![Cross-validation grid over model × method × scope × level](figures/voter-grid.png)

*Each cell is one configuration's three-best-fold mean; the grid is how we see the search
space at a glance.*

---

## 4 · Things that did not work, and one that nearly cost us

Negative results are cheap to hide and expensive to rediscover, so:

- **Input truncation.** Our first training wave capped inputs at a sequence length that
  silently cut **16.9 % of product pages** and 7.2 % of descriptions, with the truncation
  side differing between training and inference. We found it by auditing token lengths
  rather than by seeing a bad score. After verifying that all 3,360 instances fit in 8,192
  tokens at every level, we retrained. Anyone building on this data should measure their
  truncation before trusting a level-ablation.
- **Data augmentation did nothing.** Paired over 68 recipes, augmentation moved the mean by
  +0.007 — inside noise. We dropped it.
- **Prompting has a ceiling well below training.** Optimised prompts (OPRO) reached ST2
  0.59 against 0.89 for a trained head, and ST3 0.53 against 0.67.
- **Minority-class specialists cannot carry a label alone.** A specialist trained only on
  the rare classes contributes three of nine votes, so it can break ties on rare labels but
  can never fire one against the other six votes. That is the intended behaviour, and it
  means such a branch raises rare-label recall only in combination, never on its own.

---

## 5 · Reproducibility and scope

This repository documents the system. It deliberately does not contain:

- **the task data**, in any form, including derived splits — see [DATA.md](DATA.md);
- **per-instance predictions**, which are tied to identifiable channels;
- **trained adapter weights**, whose licence status as derivatives of a share-alike dataset
  we have not settled.

Curated source for the parts that carry the method — the metric implementation, the voting
rules, the channel-disjoint split builder, and the selection scripts — will be added here.
It is held back for one pass of cleaning: the working tree carries internal cluster paths
and account names that do not belong in public.

**Bonus track.** We did not collect off-platform destination data. **Legal material:** we
used only the citations shipped in `legal_provisions.json` and retrieved nothing beyond
them.

---

## 6 · Ethics

The compliance labels in this task are a **research benchmark, not legal advice and not a
finding of unlawfulness**. Our system outputs risk flags for a benchmark; it does not
determine that any creator violated any provision.

We do not re-identify creators, do not contact them, and report no result that would mark
an individual creator as non-compliant. All figures here are aggregate. See
[DATA.md](DATA.md) for the licence terms we accepted with the data.
