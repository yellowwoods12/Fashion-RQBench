# Legacy: the Gemini-authored pipeline

These two files are the **original** Fashion-RQ query pipeline, kept so the methodology
behind the Gemini-authored numbers can be inspected and cited. They are not wired into this
release.

| File | Role |
|---|---|
| `benchmark_creation_unfiltered_public_dataset_purity_v3.py` | stage 1 — cluster analysis + query authoring via the hosted Gemini API |
| `enrich_benchmark_with_query_attributes.py` | stage 2 — per-query attribute extraction via Gemini |

## Why they are here

They retain the original pipeline's local-path placeholders and API-client assumptions,
including `DATA_DIR = Path("/path/to/data")` and a Gemini client. They are reference
material, not runnable code; changes beyond path redaction would obscure the provenance
that makes them worth including.

Use them to compare prompts, budgets and cluster-selection rules against the Qwen3-8B
pipeline in the parent directory. To actually reproduce a corpus, use
`../generate_cluster_queries.py` and `../enrich_query_attributes.py`; the task builders and
scorers are shared between both generations.

## What changed in the port

Porting from the hosted Gemini to a local 8B model was not a drop-in substitution. Three
differences had to be handled, all documented in [../../../docs/BENCHMARK.md](../../../docs/BENCHMARK.md):

1. **Quantity.** Gemini returned 9 queries for the largest cluster on the original prompt;
   Qwen3-8B returned 31, reading the prompt's "one query per major product type" literally
   and enumerating colours. The prompt now states a hard maximum and adds a code-level
   truncation as a postcondition.
2. **Context length.** Qwen3-8B's window is 40,960 tokens; stage-1 prompts reach ~56k. The
   client loads it with YaRN RoPE scaling (factor 4 → 163,840).
3. **Register.** With the enumeration wording, Qwen3-8B produced terse 5.8-word keyword
   strings at 88% uniqueness; banning attribute-value enumeration and near-duplicate
   rephrasings moved it to 11.2 words at 99.1% unique.

The released corpus therefore differs from the Gemini-authored target in volume (+7.9%) and
mean query length (+12%) while matching it exactly on cluster counts (17 / 135 / 436) and
cluster-size distributions. `benchmark/stats.txt` shows both columns side by side.
