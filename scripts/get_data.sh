#!/usr/bin/env bash
# Development filing: NIKE, Inc. FY2023 Form 10-K (the example filing used in LangChain's official RAG tutorial)
set -e
mkdir -p "$(dirname "$0")/../data"
curl -L -o "$(dirname "$0")/../data/nke-10k-2023.pdf" \
  https://raw.githubusercontent.com/langchain-ai/langchain/v0.3/docs/docs/example_data/nke-10k-2023.pdf
