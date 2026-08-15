# Data, licence, and ethics

## Why there is no data here

The ChildSafeAds corpus is derived from [SponsorBlock](https://sponsor.ajay.app/) and is
distributed under **CC BY-NC-SA 4.0** together with the shared task's own data-use terms,
which we accepted at registration. Those terms do not permit us to redistribute the
dataset, and share-alike would attach to anything we published that embeds it.

So this repository contains no instances, no splits derived from instances, no per-instance
predictions, and none of the material shipped alongside the data (`labels_taxonomy.md`,
`legal_provisions.json`, the starting kit). All of that comes from the organisers, and it
is theirs to distribute.

**To obtain the data:** register for the competition on
[Codabench](https://www.codabench.org/competitions/17595/) and download it there. Our code,
once published here, expects the official files unchanged in a local `data/raw/`.

## What the data is

3,360 instances, each one sponsored segment from a YouTube video on a channel curated to
plausibly reach a significant teenage audience, paired with the product page its
description links to. Splits are **channel-disjoint**: train 2,353 / dev 504 / test 503,
with no channel appearing in more than one split.

Two properties are given by the dataset and are not re-assessed by our system: the channel
is child-facing, and the segment is commercial.

## Predictions and model weights

We do not publish per-instance predictions. Each instance carries a channel identifier, and
a published prediction file would amount to a per-creator compliance assessment — exactly
what the task's ethics terms and our own commitments rule out.

We also do not currently publish trained adapter weights. They are derived from a
share-alike corpus, and we would rather leave that question open than answer it carelessly.

## Ethics commitments

Working with this dataset we commit to:

- **no re-identification** of creators or channels, and **no contact** with them;
- **no material that marks an individual creator as non-compliant** — quotations in any
  publication are anonymised, and results are reported only in aggregate;
- **no manual labelling of the test set**;
- one account per team, use restricted to this shared task.

The compliance risk flags are a research benchmark grounded in EU consumer law (UCPD,
AVMSD, DSA, CRD). They are **not legal advice** and constitute **no finding that any
content is unlawful**.

## Licence

The code we publish in this repository is released under the MIT licence (see
[LICENSE](LICENSE)). That licence covers **our code only**. It does not extend to the task
data, to the organisers' materials, or to any model weights.
