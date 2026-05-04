# app-malt

This app now uses the repository-level local providers:

- vLLM through `LLM_BASE_URL` and `LLM_MODEL`
- sentence-transformers through `EMBEDDING_MODEL`
- Qdrant through `QDRANT_URL`

Install dependencies and configure `.env` from the repository root:

```bash
pip install -r ../requirements.txt
cp ../.env.example ../.env
```

## Build RAG

The RAG source files remain:

- `data/rag_constraints.json`
- `data/rag_tools.json`

Build the local Qdrant collections from the repository root:

```bash
python scripts/build_qdrant_indexes.py --app app-malt
```

The old notebooks under `create_RAG_index/` are legacy Azure references and are no longer needed for local runs.

## Build error_checker

## Experiment instruction
It adds on the module one by one.

1. Baseline: All constraints as static prompt.
```
python baseline_static_prompt.py
```
2. Query-specific constraints.
```
python query_specific_constraint_prompt.py
```
3. Query-specific constraints + CoT.
```
python cot_with_query_specific.py
```
4. Query-specific constraints + CoT + error reduce.
```
python cot_with_error_check.py
```
5. Query-specific constraints + CoT + error reduce + tools.
```
python full_cot_with_tools.py
```