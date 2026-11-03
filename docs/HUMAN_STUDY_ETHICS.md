# Human studies: ethics, protocol, and aggregate results

This page documents the two human studies reported in the paper and the
ethics/consent arrangements around them. The repository releases only aggregate
results; it does **not** contain participant-level or annotator-level data.

## Study A — query naturalness and intent clarity

- **Participants:** 32.
- **Scale:** 5-point Likert, rated separately for naturalness and intent clarity.
- **Query sources:** real user queries (CHAT), an LLM baseline (LLM), and
  Fashion-RQBench (RQBENCH).
- **Reliability:** Cronbach's α = 0.977 (naturalness) and 0.969 (intent clarity).

| Source | Naturalness ↑ | Intent clarity ↑ |
|---|---:|---:|
| CHAT | 3.52 | 2.83 |
| LLM | 2.84 | 3.76 |
| RQBENCH (ours) | **3.62** | **3.91** |

Pairwise differences in intent clarity were statistically significant; this is
the dimension on which RQBENCH separates most clearly from real user queries.

## Study B — SID relevance-label validation

- **Design:** blinded comparison of RQ-VAE configurations with and without the
  exclusivity loss.
- **Queries:** 150 randomly sampled queries, 75 per configuration.
- **Items:** 8 products per query, 2 per relevance grade → 1,200 query–product
  pairs.
- **Annotated:** 990 pairs (495 per configuration; the per-grade `Pairs` column
  below is per configuration), with 3 judgments per pair → 2,970 judgments.
- **Annotators:** 38.
- **Scale:** 4-point relevance (0–3); the human label for a pair is the rounded
  mean rating.

| Grade | Pairs | Rel. (no excl.) | Agree. (no excl.) | MAE (no excl.) | Rel. (excl.) | Agree. (excl.) | MAE (excl.) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exact (E) | 124 | 2.32 | 57.3% | 0.68 | 2.92 | 93.5% | 0.08 |
| Sibling (S) | 124 | 1.32 | 42.7% | 0.87 | 2.04 | 92.7% | 0.07 |
| Related (R) | 124 | 1.68 | 36.3% | 0.84 | 1.10 | 91.1% | 0.10 |
| Irrelevant (I) | 123 | 0.57 | 64.2% | 0.57 | 0.05 | 96.7% | 0.05 |
| **Overall** | **495** | — | **50.1%** | **0.74** | — | **93.5%** | **0.07** |

With the exclusivity loss, quadratic-weighted Cohen's κ improves from 0.501 to
0.962, and Spearman's ρ from 0.478 to 0.977. Mean human relevance follows the
expected grade ordering with the loss (2.92 > 2.04 > 1.10 > 0.05); without it,
Sibling and Related are substantially less separable.

## Ethics and consent

Participation in both studies was voluntary. Informed consent was obtained
before any task. The studies collected only ratings of queries and
query–product pairs; no personally identifying information is included in this
repository. Results are reported only in aggregate, and no participant-level or
annotator-level records are included in this release. Raw responses are not
included in this release.

> **Authors to complete before submission:** insert the IRB/ethics approval
> reference, the exact consent wording, and compensation details. These are not
> present in the current release materials.

## What is released

- The aggregate tables above, reproduced in the paper.
- No raw annotations, annotator identifiers, or participant-level responses.

Any request for raw annotations would require a separate consent/privacy review.
