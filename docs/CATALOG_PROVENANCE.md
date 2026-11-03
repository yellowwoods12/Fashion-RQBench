# Catalog provenance

How the released `items_with_sids.parquet` relates to the public source catalog,
and which steps this repository can and cannot reproduce.

## Source catalog

The catalog is derived from the **Fashion Product Images Dataset** distributed
on Kaggle by Param Aggarwal:

<https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset>

Its `styles.csv` contains **44,441** product rows. The released parquet has
**44,446** rows: the 44,441 source rows plus **5 rows with a null `imageURL`**
that were retained because their attributes are complete. This matches the
counts and the figure in [DATA_CARD.md](DATA_CARD.md).

## Columns from the source dataset

These columns come from `styles.csv`:

`id`, `gender`, `masterCategory`, `subCategory`, `articleType`, `baseColour`,
`season`, `year`, `usage`, `productDisplayName`

Their distinct-value counts are documented in
[DATA_CARD.md](DATA_CARD.md#schema-of-items_with_sids-parquet).

## Additional columns in the released parquet

| Column | Status |
|---|---|
| `brandName`, `ageGroup`, `price`, `discountedPrice`, `fashionType`, `displayCategories`, `articleAttributes`, `description`, `imageURL` | Present in the authors' working catalog. Their derivation from the raw Kaggle release is **not shipped**. |
| `item_context` | Built from the fields below; template documented here, builder script **not shipped**. |
| `embedding` | 1024-d Qwen3-Embedding-0.6B vector over `item_context`. Shipped separately; regeneration is not supported (see [DATA_CARD.md](DATA_CARD.md#derived-columns)). |
| `sid_0`, `sid_1`, `sid_2`, `semantic_id` | RQ-VAE codebook indices and the resulting semantic ID. The checkpoint is released and the inference path is in [`src/rqvae/generate_sids.py`](../src/rqvae/generate_sids.py); see [REPRODUCE_SIDS.md](REPRODUCE_SIDS.md). |

## `item_context` template

`item_context` is a single flat string with ` | ` separators:

```text
Product: {productDisplayName} | masterCategory: {masterCategory} | subCategory: {subCategory} | articleType: {articleType} | Brand: {brandName} | Gender: {gender} | Age Group: {ageGroup} | Color: {baseColour} | Season: {season} | Usage: {usage} | Fashion Type: {fashionType} | Attributes: {articleAttributes} | Price: {price} | Description: {description}
```

Fields absent for a row are stored as empty values. Length statistics for the
released file: median **954** characters, maximum **52,860** (the long tail is
driven by `Description`).

## Source of record

The released parquet is the **canonical source of record** for the benchmark:

| File | SHA-256 |
|---|---|
| `items_with_sids.parquet` | `ce3431c09daf158b16b3cb9456e6ffa8ddf114e7fc488d6a4f2d62683ed5b8eb` |

Rebuilding the file from the public Kaggle download is **not supported**: some
columns were enriched in the authors' working catalog, the `item_context`
builder is not part of this release, and the embedding/model pipeline has
environment-specific dependencies. Use the checksummed parquet above and verify
it with `python scripts/validate_release.py`.

## Licensing

- The upstream catalog is distributed under the MIT licence (as stated by the
  dataset's distributor); see [../NOTICE](../NOTICE) and
  [DATA_CARD.md](DATA_CARD.md#licence).
- The SID columns and benchmark artifacts are CC BY 4.0.
- Product images are hotlinked to the Myntra CDN, not redistributed; brand
  names remain their owners' trademarks.
