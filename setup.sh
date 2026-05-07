#!/usr/bin/env bash

set -e

PROJECT_NAME="eliza_private_equity_rag_project"

mkdir -p $PROJECT_NAME

cd $PROJECT_NAME

# Root files
touch \
  AGENTS.md \
  DECISIONS.md \
  README.md \
  pyproject.toml \
  docker-compose.yml \
  Dockerfile \
  .env.example \
  app.py

# Directories
mkdir -p \
  scripts \
  src/interfaces \
  src/preprocessing \
  src/embedders \
  src/vector_stores \
  src/retrieval \
  src/llm \
  src/pipeline \
  tests

# Scripts
touch scripts/build_index.py

# Interfaces
touch src/interfaces/__init__.py

# Preprocessing
touch \
  src/preprocessing/parser.py \
  src/preprocessing/chunker.py

# Embedders
touch src/embedders/__init__.py

# Vector stores
touch \
  src/vector_stores/chroma_store.py \
  src/vector_stores/qdrant_store.py \
  src/vector_stores/__init__.py

# Retrieval
touch \
  src/retrieval/query_analyzer.py \
  src/retrieval/hybrid_retriever.py

# LLM
touch src/llm/__init__.py

# Pipeline
touch \
  src/pipeline/rag_pipeline.py \
  src/pipeline/prompt_builder.py \
  src/pipeline/__init__.py

# Tests
touch \
  tests/test_preprocessing.py \
  tests/test_retrieval.py

# Git
git init

# Python env
uv venv

echo ""
echo "✅ Project scaffold created successfully."
echo ""
echo "Next steps:"
echo "  cd $PROJECT_NAME"
echo "  source .venv/bin/activate"
echo "  claude"
```
