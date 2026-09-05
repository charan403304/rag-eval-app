# Docs RAG Assistant — Hand-Rolled Hybrid RAG with a Real Eval Harness

A retrieval-augmented generation app that answers questions over a small
internal-docs corpus (deployment process, API guidelines, incident response,
DB migrations), with citations, an explicit "I don't know" guardrail, and —
the part most RAG demos skip — a **quantitative evaluation harness** that
measures retrieval precision/recall, citation validity, refusal accuracy,
and answer quality, instead of just eyeballing a few example outputs.

The retrieval and fusion logic is implemented by hand (BM25 + vector search
+ reciprocal rank fusion) rather than delegated to a framework, so the
internals are inspectable and the design trade-offs below are actually
mine to defend.

**Runs fully offline out of the box** — deterministic hash-based embeddings
and a mock LLM backend mean `make test` and `make eval` need zero API keys.
Swap two environment variables to point it at real embeddings (Voyage AI /
OpenAI) and a real LLM (Claude / GPT-4o-mini).

---

## Architecture

```
                    ┌─────────────────┐
   data/corpus/*.md │  Header-aware,   │
   ───────────────► │  sentence-safe   │
                     │  chunking        │
                     └────────┬─────────┘
                              │ chunks
                 ┌────────────┴────────────┐
                 ▼                         ▼
        ┌─────────────────┐      ┌──────────────────┐
        │  BM25 (sparse,   │      │  Embed + store    │
        │  keyword)        │      │  (dense, vector)  │
        └────────┬─────────┘      └─────────┬────────┘
                  │                          │
                  └──────────┬───────────────┘
                              ▼
                  Reciprocal Rank Fusion (RRF)
                              │
                    top-K fused chunks
                              │
              score < threshold?──yes──► "I don't know" (no LLM call)
                              │no
                              ▼
                 LLM generation w/ citation
                 + refusal prompt guardrail
                              │
                              ▼
                    cited answer + stats
                    (tokens, cost, latency)
```

Every layer (`chunking.py`, `embeddings.py`, `bm25.py`, `vectorstore.py`,
`retriever.py`, `generator.py`) is a narrow, independently testable module.
`pipeline.py` is the single entry point the Streamlit app, the eval
harness, and (if you add one) an API layer would all call — so eval is
always exercising the exact code path the app runs, not a parallel
reimplementation that can drift out of sync.

## Design decisions (and why)

- **Chunking snaps to sentence boundaries within a target token window**,
  and never crosses a markdown `##` header. Pure fixed-size chunking is
  simpler but regularly slices a sentence in half, which measurably hurts
  retrieval quality — the resulting chunk embedding represents a sentence
  fragment, not a coherent idea. See `src/chunking.py`.
- **Hybrid retrieval (BM25 + vector), fused with Reciprocal Rank Fusion**,
  not vector search alone. BM25 catches exact keyword/identifier matches
  ("429", "healthz", "v1") that embeddings blur together; vectors catch
  paraphrases BM25 misses entirely. RRF combines them by *rank*, not raw
  score, because BM25 scores and cosine similarities live on incomparable
  scales — averaging them directly would need per-corpus hand-tuning to
  mean anything. See `src/retriever.py`.
- **Two-layer refusal guardrail**: if the top fused retrieval score is
  below a threshold, the app returns "I don't know" *without calling the
  LLM at all* — cheaper and more reliable than hoping the model declines
  gracefully. There's also a prompt-side instruction to refuse rather than
  guess, as a second line of defense. See `src/pipeline.py::answer`.
- **Citation validity is checked deterministically**, not just requested
  in the prompt. The eval harness parses `[n]` markers out of the answer
  and verifies each one points at a chunk that was actually retrieved —
  this catches citation hallucination without needing an LLM judge.
- **Brute-force cosine search over a numpy matrix**, not a vector DB. At
  this corpus size (dozens–low thousands of chunks) it's sub-millisecond
  and has zero operational cost. `VectorStore`'s interface (`add`,
  `search`, `save`, `load`) is narrow on purpose — swapping in
  Chroma/pgvector/Pinecone at scale is a one-file change (see "Scaling
  this up" below), not a rewrite.
- **Offline-by-default providers.** `HashEmbedder` (char n-gram hashing
  trick) and `MockBackend` (extractive stand-in) mean the entire pipeline
  — chunking, fusion, guardrails, citation checking, and eval — is
  provably exercised end-to-end without secrets, in CI, on a laptop with no
  network. This is the same reasoning as testing with an in-memory DB
  instead of requiring a live Postgres for unit tests.

## Evaluation results

Run with `make eval`. These are the actual numbers from the offline
providers (`hash` embeddings + `mock` LLM) against the 18-question eval set
in `data/eval/eval_set.json` (14 answerable, 4 deliberately out-of-scope):

| Metric | Score | What it measures |
|---|---|---|
| Retrieval recall | 100% | Right document retrieved when one exists |
| Retrieval precision | 47.6% | Fraction of retrieved chunks that were actually relevant |
| Citation validity | 100% | Citation markers point at real, retrieved chunks |
| Refusal accuracy | 77.8% | Refused OOD questions AND answered in-scope ones, correctly |
| Keyword recall (answer) | 46.4% | Expected key facts present in the generated answer text |

### Reading these numbers honestly

These are **not** the numbers you'd get with real providers — and that gap
is the point of having an eval harness at all:

- **Refusal accuracy (77.8%) is the weakest score**, and it's a known,
  explainable limitation: the hash embedder has no real semantic
  generalization (it can't tell "capital of France" is unrelated to
  deployment docs the way a learned embedding model would), so it
  sometimes returns a false positive on out-of-scope questions instead of
  triggering the refusal threshold. Run `EMBEDDING_PROVIDER=voyage make
  index && make eval` with a real embedding model and this number should
  jump substantially — that's a concrete, testable prediction, not a hand
  wave.
- **Keyword recall (46.4%) is low because `MockBackend` doesn't synthesize**
  — it deterministically extracts the first two sentences of the top
  chunk rather than composing an answer, so it often retrieves the right
  chunk (100% retrieval recall) but phrases the answer in a way that
  misses the specific keyword list. This is expected and is exactly why
  `MockBackend`'s docstring calls it a plumbing stand-in, not a quality
  baseline. `LLM_PROVIDER=anthropic make eval` replaces it with a real
  synthesized, cited answer.
- **Retrieval precision (47.6%) vs. recall (100%)**: the retriever is
  tuned to favor recall (`top_k_final=4`) — for a docs assistant, missing
  the right chunk is worse than including one extra irrelevant one, since
  the LLM (or a human skimming the "retrieved context" panel) can ignore
  noise but can't answer from a chunk it never saw.

Add `--llm-judge` (needs `ANTHROPIC_API_KEY`) to also score answer
correctness with Claude as a judge instead of relying on keyword matching
alone — keyword recall is a blunt proxy that penalizes correct answers
phrased differently than expected.

## Quickstart

```bash
git clone <this-repo>
cd rag-eval-app
pip install -r requirements.txt

# Runs fully offline, no API keys needed
python scripts/build_index.py
streamlit run app.py
```

Open http://localhost:8501, ask something like *"What is the API rate
limit?"* — and something out of scope like *"What's our PTO policy?"* to
see the refusal guardrail fire.

### Using real providers

```bash
cp .env.example .env
# edit .env: set EMBEDDING_PROVIDER=voyage (or openai) and LLM_PROVIDER=anthropic
# fill in the corresponding API key
source .env  # or use python-dotenv / your shell's env loading
python scripts/build_index.py   # rebuild the index with real embeddings
streamlit run app.py
```

### Docker

```bash
docker compose up --build
```

### Running tests and eval

```bash
make test          # 16 unit tests, pure offline, ~1s
make eval          # full eval harness, keyword-based scoring
make eval-judge     # eval harness + Claude-as-judge scoring (needs ANTHROPIC_API_KEY)
```

## Project layout

```
rag-eval-app/
├── app.py                    # Streamlit UI
├── src/
│   ├── config.py              # every tunable knob lives here
│   ├── chunking.py            # header-aware, sentence-safe chunking
│   ├── bm25.py                 # hand-rolled BM25 (no external dep)
│   ├── embeddings.py          # pluggable: hash (offline) / openai / voyage
│   ├── vectorstore.py         # numpy brute-force cosine search + persistence
│   ├── retriever.py           # hybrid retrieval via reciprocal rank fusion
│   ├── generator.py            # LLM call + citation extraction + guardrails
│   ├── cost_tracker.py        # per-query token/cost/latency stats
│   └── pipeline.py            # RAGPipeline — the single shared entry point
├── eval/
│   ├── metrics.py              # retrieval P/R, citation validity, LLM-judge
│   └── run_eval.py             # CLI eval runner, prints + dumps JSON report
├── data/
│   ├── corpus/*.md             # the knowledge base (sample docs included)
│   └── eval/eval_set.json      # 18 labeled questions (14 answerable, 4 not)
├── tests/                      # 16 unit tests, offline
├── scripts/build_index.py      # build + persist the vector index
├── Dockerfile / docker-compose.yml
└── .github/workflows/ci.yml    # tests + eval run on every push
```

## Scaling this up

Honest notes on what would need to change for this to handle a real corpus
(thousands of documents, concurrent users) rather than a portfolio-sized one:

- **Vector store**: swap `VectorStore` (numpy brute force) for pgvector or
  Chroma once the corpus exceeds ~50-100k chunks or needs concurrent
  writes; the `add`/`search`/`save`/`load` interface is designed to make
  this a single-file swap.
- **BM25**: the hand-rolled implementation is O(n) per query over all
  documents — fine for thousands of chunks, not for millions. At that
  scale, use Elasticsearch/OpenSearch's BM25 implementation instead.
- **Reranking**: add a cross-encoder reranking step after RRF fusion and
  before the top-k cutoff — this is usually the single highest-ROI
  addition to a hybrid retrieval pipeline once the basics are working.
- **Chunking**: the current sentence-window approach doesn't handle tables,
  code blocks, or very long unstructured paragraphs well; a
  format-specific chunker (or a semantic chunker using embedding
  similarity to find natural breakpoints) would help on messier real-world
  docs.
- **Index freshness**: there's no incremental indexing — `build_index.py`
  rebuilds from scratch. A real deployment needs to detect changed/added
  docs and re-embed only the delta.

## Known limitations (current state)

- The mock LLM backend doesn't synthesize answers — see "reading these
  numbers honestly" above.
- Hash embeddings have no real semantic generalization; they're a
  plumbing/testing stand-in, not a quality baseline — real semantic
  matching requires `EMBEDDING_PROVIDER=voyage` or `openai`.
- No conversation memory — each query is independent (no multi-turn
  follow-up handling).
- No auth/rate-limiting on the Streamlit app itself (out of scope for this
  project — see the URL-shortener project in this portfolio for that).

## License

MIT
