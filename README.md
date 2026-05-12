# LLM Engineering Journey

A personal learning repo for working through LLM engineering — from RAG and prompting to agents, evaluation, and beyond. Each numbered folder is a self-contained module with its own code and notes.

This repo was **originally built for my Intel GPU Software Development Internship learning** (the LangChain RAG tutorial in module 01). I'm now expanding it to learn further across the LLM engineering stack — prompt engineering, agents, evaluation, fine-tuning, and whatever else turns out to be load-bearing.

## Modules

| # | Module | What it covers |
|---|--------|----------------|
| 01 | [langchain-rag-tutorial](01-langchain-rag-tutorial/) | Retrieval-Augmented Generation with LangChain — comparing a chain approach (retrieve-then-generate) with an agent approach (LLM-controlled retrieval), using Chroma + HuggingFace embeddings and Claude as the generator. |
| 02 | [openai-prompt-engineering](02-openai-prompt-engineering/) | Prompt engineering patterns with the OpenAI API — structuring prompts, controlling output, and iterating on prompt design. *(in progress)* |

More modules will be added as the journey continues.

## Repo Layout

```
llm-engineering-journey/
├── 01-langchain-rag-tutorial/      # RAG: chain vs. agent, Chroma, embeddings
├── 02-openai-prompt-engineering/   # Prompt engineering with OpenAI
└── README.md
```

## Setup

Each module has its own dependencies, but they share a single `.venv` and `.env` at the repo root.

```bash
python -m venv .venv
source .venv/bin/activate
```

Create a `.env` file at the repo root with whichever keys the module you're running needs:

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
```

Then `cd` into a module and follow its README.

## Why this repo exists

LLM engineering is a moving target — the useful skill is not memorizing one framework but understanding how the pieces (retrieval, prompting, tool use, evaluation, agents) fit together. This repo is where I work through each piece with small, runnable examples and keep notes on what actually mattered versus what was just framework noise.
