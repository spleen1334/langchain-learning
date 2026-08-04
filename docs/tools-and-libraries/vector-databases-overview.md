# Vector Databases — Concepts

Theory only: what a vector database is and how it decides "closest match," independent of any vendor. For the store this course uses, see `vector-databases-chroma.md`. For a side-by-side of Chroma/Qdrant/Pinecone/Weaviate/FAISS, see `vector-databases-comparison-vendors.md`.

## What problem it solves

Regular databases match on exact values (`WHERE id = 5`). A vector database matches on **meaning**: text/images/audio get converted into embeddings — lists of numbers (vectors) — such that similar meaning → nearby vectors. The database's job is answering, fast, "which stored vectors are nearest this query vector?"

```
"How do I reset my password?"  →  [0.12, -0.87, 0.33, ...]   ← embedding
                                          │
                                          ▼
                          find nearest vectors in the store
                                          │
                                          ▼
        "To reset your password, go to Settings..."   ← closest match
```

## Distance vs. similarity

Two mirror-image ways to say the same thing — how close two vectors are:

| | Meaning | Best value | Trend |
|---|---|---|---|
| **Distance** | how far apart | `0` | *lower* = better |
| **Similarity** | how alike | `1` (or 100%) | *higher* = better |

> They're not two different computations — one is just the other flipped. Most similarity scores are literally `1 - distance` under the hood.

**Watch for this:** an API call returns *whichever form the store's default metric produces* — some hand back a similarity (higher = better), others a raw distance (lower = better). Never assume; check.

## The three common metrics

The specific yardstick used for "how close" is chosen when a collection/index is created:

| Metric | What it measures | Notes |
|---|---|---|
| **Euclidean / L2** | Straight-line distance between two points | Sensitive to vector *length*, not just direction |
| **Cosine** | The *angle* between two vectors | Ignores magnitude — the default for text embeddings, since it treats a short and a long passage on the same topic as similar |
| **Dot product** | Like cosine, but magnitude still counts | Used when the embedding model's own training objective relies on it |

## Why the metric matters in practice

- **Scores aren't comparable across stores or metrics.** A cosine similarity of `0.8` and an L2 distance of `0.8` mean nothing next to each other — don't hardcode a similarity threshold without knowing which metric produced it.
- **Ranking survives the conversion either way.** Sorting ascending by distance or descending by similarity produces the *same order* — flipping distance↔similarity only changes what the number looks like, not which results win.
- **The metric is a setup-time choice**, not something you pick per query — it's baked into the collection/index when it's created.

## How the search itself stays fast

Naively, finding the nearest vector means comparing the query against *every* stored vector — fine for hundreds of vectors, too slow for millions. Vector databases avoid that with **Approximate Nearest Neighbor (ANN)** indexing (e.g. HNSW): a graph/tree structure built over the vectors so a query only has to check a small, promising subset. The trade is in the name — "approximate": near-certain but not guaranteed to return the *exact* top-k, in exchange for orders-of-magnitude faster search.

## Beyond plain top-k: MMR

Plain nearest-neighbor search has a blind spot: the top `k` results can all be near-duplicates of each other (five chunks that basically say the same thing), because each one is scored only against the query — never against each other. That's wasted space in a `k`-sized result set.

**Maximal Marginal Relevance (MMR)** fixes this by scoring candidates on two things at once:
- **Relevance** — how close to the query (the normal similarity score).
- **Diversity** — how *different* from results already picked.

```
1. Fetch a larger candidate pool (fetch_k), not just k
2. Pick the single most relevant candidate → add to results
3. Repeat: pick the candidate that best balances
     (relevant to the query) vs. (different from what's already picked)
   until k results are chosen
```

Two knobs control the trade-off:

| Parameter | Controls |
|---|---|
| `fetch_k` | Size of the initial candidate pool MMR selects from (bigger = more room to diversify) |
| `lambda_mult` (0–1) | Relevance vs. diversity balance — `1` = pure relevance (behaves like plain top-k), `0` = pure diversity |

**Use it when:** the corpus has redundant/overlapping chunks and you want the result set to *cover more ground*, not repeat itself — e.g. summarizing a topic from multiple angles rather than answering one narrow fact. **Skip it when:** you want the single most relevant fact and don't care about variety — MMR adds compute for no benefit there.

## The other half: metadata filtering

Vectors alone can't express "similar, but only from *this* source" or "...published after March." Vector databases pair each vector with a metadata payload (arbitrary key/value fields) and let a query combine similarity search with metadata filters — narrowing the candidate set before or alongside the nearest-neighbor search, rather than requiring a separate database.
