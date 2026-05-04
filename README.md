# MeshAgent Local Setup

This repository runs the existing agentic graph-analysis pipeline locally:

- vLLM serves an OpenAI-compatible chat model.
- `sentence-transformers` computes local embeddings.
- Qdrant stores and searches RAG constraints and tools.
- The original app flow is preserved: RAG, step generation, code generation, execution, self-debugging, verification, and evaluation.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install vLLM separately using the command recommended for your CUDA/PyTorch setup:

```bash
pip install vllm
```

## Configure

```bash
cp .env.example .env
```

Default local settings:

```bash
LLM_BASE_URL=http://localhost:8000/v1
LLM_MODEL=Qwen/Qwen2.5-Coder-14B-Instruct
LLM_API_KEY=EMPTY
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
QDRANT_URL=http://localhost:6333
QDRANT_CONSTRAINT_COLLECTION=meshagent_constraints
QDRANT_TOOL_COLLECTION=meshagent_tools
```

## Start Local Services

Start Qdrant:

```bash
docker run --rm -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

Start vLLM for a 36 GB VRAM machine with moderate context and low concurrency:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-Coder-14B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 2
```

If your vLLM/GPU stack supports a 4-bit quantized variant, set `LLM_MODEL` to that model id and pass the matching vLLM quantization flags. Keep the initial context at 8k or 16k and concurrency low on 36 GB VRAM.

## Build RAG Indexes

The local indexer reads each app's existing JSON files:

- `data/rag_constraints.json`
- `data/rag_tools.json`

Build all indexes:

```bash
python scripts/build_qdrant_indexes.py --app all
```

Build one app:

```bash
python scripts/build_qdrant_indexes.py --app app-malt
python scripts/build_qdrant_indexes.py --app app-CRG
python scripts/build_qdrant_indexes.py --app app-traffic-analysis
```

## Smoke Tests

```bash
python scripts/test_embeddings.py
python scripts/test_vllm.py
python scripts/test_retrieval.py --app app-malt
```

## Run Apps

Run from each app directory so existing relative data paths continue to work:

```bash
cd app-malt
python full_cot_with_tools.py

cd ../app-CRG
python full_cot_with_tools.py

cd ../app-traffic-analysis
python full_cot_with_tools.py
```

## Legacy Azure Notebooks

The `create_RAG_index/rag_azure_*.ipynb` notebooks are retained as historical references only. Local indexing now uses `scripts/build_qdrant_indexes.py`.
