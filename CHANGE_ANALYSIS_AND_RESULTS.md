# MeshAgent Change Analysis and Results

## Executive Summary

This repository was adapted from the original MeshAgent codebase into a local, reproducible execution setup. The main change was replacing Azure-based LLM, embedding, and search dependencies with a local stack built around vLLM, `sentence-transformers`, and Qdrant. The original application flow was intentionally preserved: retrieve relevant constraints and tools, generate step-by-step code with an LLM, execute the generated `process_graph` function, run verification, self-debug failures, and compare the final output against golden answers when available.

After the local migration, the later commits focused on operational robustness. They added flexible Qdrant configuration, ensured log directories are created automatically, introduced NetworkX JSON compatibility handling, and improved evaluation behavior so missing golden answers or execution failures are logged instead of stopping an entire run.

## Commit-Level Change History

### Baseline

The upstream baseline consists of the initial code import, a cleanup commit, and the addition of missing data folders:

- `9428a4f` - first commit.
- `874a6e2` - cleanup of unused files.
- `0fb45c7` - addition of missing data folders.

The active branch then diverges from `upstream/main` with a sequence of local-execution and robustness changes.

### `7b099a2` - Local vLLM and Qdrant Migration

This was the largest functional change. It introduced a local execution architecture and removed the hard dependency on Azure services.

Key additions:

- `.env.example` with local configuration defaults.
- `requirements.txt` with the new local dependencies.
- `common/config.py` for centralized environment and collection configuration.
- `common/llm_provider.py` for OpenAI-compatible chat access through vLLM.
- `common/embeddings_provider.py` for local `sentence-transformers` embeddings.
- `common/vector_store.py` for Qdrant-backed vector search.
- `common/rag_indexer.py` for building local RAG indexes from each app's JSON files.
- `common/chain_compat.py` to keep legacy `.run(...)` call sites working.
- `scripts/build_qdrant_indexes.py` to build indexes for all supported apps.
- `scripts/test_embeddings.py`, `scripts/test_vllm.py`, and `scripts/test_retrieval.py` as smoke tests.
- `LEGACY_AZURE.md` files under each app's `create_RAG_index` folder to mark the old Azure notebooks as historical references.

The apps were modified to use the new shared modules while preserving their previous control flow. The RAG functions now call `search_constraints(...)` and `search_tools(...)`, and the LLM chains now use the local OpenAI-compatible provider instead of the previous remote configuration.

### `9283e88` - Flexible Qdrant Configuration

This commit refined the Qdrant setup so the repository can run in multiple environments:

- `QDRANT_MODE=local` uses embedded Qdrant storage on disk.
- `QDRANT_MODE=server` connects to a Qdrant HTTP service.
- `QDRANT_MODE=memory` creates an in-process, non-persistent Qdrant instance for quick tests.

It also added `QDRANT_PATH` so local storage can be configured explicitly. Relative paths are resolved from the repository root. If an old `.env` file only defines `QDRANT_URL`, the code infers the appropriate mode for backward compatibility.

### `993f3d6` - Log Directory Creation

The main scripts now create their log directory before writing JSONL output. This prevents runs from failing simply because paths such as `logs/gpt4/` or `logs/debug/` do not exist yet.

### `670e5c8` - NetworkX Node-Link Compatibility

This commit introduced `common/graph_json.py` with `node_link_graph_compat(...)`. The helper handles the difference between node-link data that uses the `links` key and newer NetworkX behavior that may expect `edges`.

The app helpers were updated to use this compatibility function when converting serialized graph outputs back into NetworkX graphs. This is important because generated code may return graph-like JSON structures, and the verifier expects a valid NetworkX graph.

### `9a80e9b` - More Robust Query Execution

This commit improved `userQuery(...)`, mainly in `app-CRG` and `app-traffic-analysis`.

Before this change, a prompt without a matching golden answer could terminate the run. The updated flow can run in inference-only mode when ground truth is unavailable. It still generates code, executes it, verifies invariants, and logs the result, but it skips the final accuracy comparison.

The change also improved error logging around generated-code execution so failures at different steps are easier to trace.

### `0a24c86` - Safer Error Checking

`app-CRG/error_check.py` was made more defensive:

- `ret_graph` and `ret_list` are now checked explicitly against `None`.
- Empty graphs are no longer accidentally treated as missing values due to Python truthiness.
- `evaluate_all(...)` returns a clear error when both graph and list outputs are missing.
- The CRG execution loop was simplified so error conditions break out more predictably.

This change improves verifier reliability, especially for edge cases where an empty graph may still be a valid output.

### `6241057` - Results and Persisted Qdrant Storage

This commit added execution artifacts rather than changing core logic:

- `app-malt/results.txt`
- `app-CRG/results.txt`
- `app-traffic-analysis/results.txt`
- `qdrant_storage/` with persisted Qdrant SQLite storage files.

The persisted Qdrant storage makes the repository easier to inspect or run without rebuilding indexes immediately, but it also means generated binary/cache artifacts are versioned.

### `db3b005` - Missing Golden Answer Handling in MALT

This commit applied the missing-ground-truth robustness pattern to `app-malt/full_cot_with_tools.py`.

If a prompt does not exactly match a key in `prompt_golden_ans.json`, the run now:

- Logs the prompt.
- Writes a `Skip` result to the JSONL output.
- Records the reason as an exact-string key mismatch.
- Continues to the next prompt instead of aborting.

## Architecture After the Changes

### Configuration Layer

`common/config.py` is the central source of runtime configuration. It loads `.env` from the repository root and exposes a frozen `LocalConfig` dataclass containing LLM, embedding, and Qdrant settings.

Collection names are generated from a base name plus the app name. For example, `meshagent_constraints` plus `app-malt` becomes `meshagent_constraints_app_malt`. Per-app collection overrides are also supported through environment variables.

### LLM Layer

`common/llm_provider.py` constructs a LangChain `ChatOpenAI` client using the configured `LLM_BASE_URL`, `LLM_MODEL`, and `LLM_API_KEY`. Because vLLM exposes an OpenAI-compatible API, the existing LangChain chat model interface can be reused.

The default model in `.env.example` is:

```text
Qwen/Qwen2.5-Coder-14B-Instruct
```

This is consistent with the repository's goal of generating executable Python code locally.

### Embedding Layer

`common/embeddings_provider.py` uses `sentence-transformers` to embed both queries and indexed documents locally. Embeddings are normalized before being stored or searched, and the default model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

### Vector Search Layer

`common/vector_store.py` creates a cached Qdrant client based on the configured mode:

- Embedded local storage through `QdrantClient(path=...)`.
- In-memory storage through `QdrantClient(location=":memory:")`.
- Server mode through `QdrantClient(url=...)`.

Search results are converted back into dictionaries and enriched with `@search.score`. This preserves compatibility with legacy helper functions that expect Azure Search-like result payloads.

### RAG Indexing

`common/rag_indexer.py` indexes three supported apps:

- `app-malt`
- `app-CRG`
- `app-traffic-analysis`

For each app, it reads:

- `data/rag_constraints.json`
- `data/rag_tools.json`

Constraints are embedded from the `constraint` field. Tools are embedded from the `description` field. Each point receives a deterministic UUID generated from the app name, kind, item ID, and item index.

### Application Flow

The application-level flow remains close to the original code:

1. Load graph data.
2. Retrieve relevant constraints and tools through RAG.
3. Ask the LLM to produce a three-step plan.
4. Generate Python code for each step.
5. Execute the generated `process_graph(graph_data)` function.
6. If execution fails, call the self-debugging chain.
7. If verification fails, retrieve additional constraints from the verifier error and self-debug again.
8. Compare the final output against a golden answer when available.
9. Write structured output to JSONL logs.

This approach minimized the migration surface area. The implementation changed the providers underneath the app logic instead of rewriting the entire pipeline.

## Results

The repository includes result logs for all three applications. They show that the local pipeline can run end-to-end, retrieve constraints, generate code, execute it, and perform verifier/golden-answer checks. The results also reveal that the generated code is not uniformly correct, especially when constraints are over-applied or when a query requires precise aggregation semantics.

### MALT Results

`app-malt/results.txt` contains three completed evaluated prompts and one unsupported prompt at the end.

Observed outcomes:

- 1 prompt passed with `Testing accuracy: 1.0`.
- 2 prompts failed with `Testing accuracy: 0.0`.
- 1 later prompt was unsupported because no ground truth was available.

The passing case asked for the total capacity of all ports of packet switch `ju1.a1.m1.s2c1`. The generated code correctly identified the packet switch, summed `physical_capacity_bps` over its port neighbors, passed verifier checks at each step, and matched the golden answer.

The first failure asked how many packet switches have more capacity than `ju1.a2.m3.s2c3` and requested an example. The model found the correct count, `3`, but produced a different valid-looking example switch than the golden answer expected. The golden answer expected `ju1.a1.m1.s2c4`, while the model returned `ju1.a1.m1.s2c1`. This may indicate either a true mismatch or an evaluation strictness issue if multiple examples are valid.

The second failure asked for the typical number of packet switches in a chassis and examples with fewer or more switches. The golden answer expected a typical count of `1.0` and no examples with fewer or more switches. The model returned `320`, which suggests it misunderstood the aggregation level and counted across a broader hierarchy than requested.

### CRG Results

`app-CRG/results.txt` contains three prompts. All three ran in inference-only mode because no exact matching golden answers were found.

Observed outcomes:

- 3 prompts completed execution.
- 3 prompts skipped accuracy evaluation with `Testing accuracy: skipped (no ground truth)`.
- Verifier checks passed during the generated-code steps.

The logs demonstrate that the new missing-ground-truth behavior works as intended. Instead of terminating the run, CRG still executes the full generation and verification path and clearly records that accuracy was skipped.

Because there were no golden answers for these prompts, the result logs confirm operational execution but do not establish task correctness.

### Traffic Analysis Results

`app-traffic-analysis/results.txt` contains six evaluated prompts and one unsupported prompt at the end.

Observed outcomes:

- 3 prompts passed with `Testing accuracy: 1.0`.
- 3 prompts failed with `Testing accuracy: 0.0`.
- 1 later prompt was unsupported because no ground truth was available.

The passing cases include:

- Counting nodes and edges in the graph.
- Counting unique nodes connected to nodes with label `app:prod` while not themselves containing that label.
- Another graph/statistics query that matched the golden answer after verification.

The first failure asked for the number of nodes. The generated code incorrectly treated graph connectivity as a prerequisite and returned `"The graph is not fully connected."` instead of the golden answer `100`. This shows that retrieved constraints can sometimes be over-applied: an invariant about isolated nodes was interpreted as a reason to reject the query rather than simply count nodes.

Another failure involved grouping by IP prefix. The golden answer grouped by broad prefixes, such as `149` and `15`, but the model grouped by full IP addresses, producing many one-count rows. This indicates a semantic interpretation error rather than an execution failure.

The final failure involved average byte and connection weights. The model output was numerically close but not equal to the golden answer, suggesting a mismatch in filtering, denominator choice, or aggregation scope.

### Aggregate Result Summary

Across the logged evaluated prompts:

| App | Evaluated prompts | Passed | Failed | Skipped / unsupported |
| --- | ---: | ---: | ---: | ---: |
| MALT | 3 | 1 | 2 | 1 |
| CRG | 0 with ground truth | 0 | 0 | 3 |
| Traffic Analysis | 6 | 3 | 3 | 1 |
| Total | 9 | 4 | 5 | 5 |

Among prompts with golden answers, the observed pass rate is:

```text
4 / 9 = 44.4%
```

This number should be interpreted cautiously. The result files are execution logs, not a controlled benchmark report. They appear to capture a small number of runs, often with one run per prompt. Some failures may reflect strict golden-answer comparisons rather than unequivocally wrong generated outputs, especially where the prompt allows multiple valid examples.

## Interpretation

The migration was technically successful: the system can run locally with vLLM, local embeddings, and Qdrant; indexes can be built from app JSON files; and the original generation/debug/evaluation loop still operates.

The main remaining weakness is answer reliability. The logs show that generated code often passes invariant checks but still fails semantic correctness against golden answers. This means the verifier catches structural issues but does not fully validate whether the generated algorithm answers the user's intent.

The most common failure modes are:

- Over-applying retrieved constraints as hard preconditions.
- Misinterpreting aggregation level.
- Returning a valid-looking but non-golden example.
- Grouping by overly specific fields instead of the intended prefix/category.
- Producing near-but-not-exact numeric aggregates.

## Notable Engineering Tradeoffs

The implementation favors compatibility over deep refactoring. This made the migration smaller and safer, but it leaves substantial duplication across the three apps. Similar functions for RAG retrieval, self-debugging, verifier handling, and output comparison are repeated in multiple files.

The repository also versions generated artifacts such as Qdrant SQLite storage and result logs. This helps reproduce the current state quickly, but it may not be ideal for long-term repository hygiene unless these artifacts are intentionally part of the deliverable.

## Recommended Next Steps

1. Consolidate the duplicated pipeline logic shared by `app-malt`, `app-CRG`, and `app-traffic-analysis`.
2. Add a structured benchmark runner that outputs machine-readable pass/fail summaries instead of relying on terminal logs.
3. Separate structural verifier checks from semantic answer checks.
4. Add query-specific validators for cases where multiple correct examples are possible.
5. Review whether `qdrant_storage/` and `results.txt` should remain versioned or be regenerated by documented commands.
6. Improve prompt instructions so retrieved constraints are treated as guidance unless the query explicitly requires them.
7. Add regression tests around the known failure cases captured in the result logs.
