# LangChain RAG Tutorial

A hands-on practice project for learning Retrieval-Augmented Generation (RAG) with LangChain. I built this while preparing for my Intel GPU Software Development Internship.

## What I'm Learning

LLMs don't know about private or recently updated information. RAG fixes this by fetching relevant documents at runtime and passing them as context. For a debugging assistant, this is important — bug reports need domain knowledge, and the model should answer from evidence rather than guess.

## Two Approaches

I implemented two architectures and compared them:

**Chain** — retrieves context upfront and injects it into the system prompt. The LLM just answers.
```
Query → Retriever → System Prompt with Context → LLM → Answer
```

**Agent** — gives the LLM a retrieval tool it can call on its own. It decides when and what to retrieve, which allows multi-step reasoning.
```
Query → LLM → [calls retrieve tool] → Docs → LLM → Answer
```

Both are applied to two document sources: a blog post about AI agents, and song lyrics.

## Pipeline

```
Raw Document → WebBaseLoader → Text Splitter → HuggingFace Embeddings → Chroma → LLM (Claude)
```

## File Structure

```
├── exampleChain.py      # chain approach on blog post
├── exampleAgent.py      # agent approach on blog post
├── chain_lyrics.py      # chain approach on song lyrics
├── agentic_lyrics.py    # agent approach on song lyrics
├── loadKey.py           # loads API keys from .env
├── test_setup.py        # quick check that LLM connection works
└── loader/              # experiments with PDF, CSV, and web loaders
```

## Setup

```bash
pip install langchain langchain-community langchain-anthropic langchain-huggingface langchain-chroma langchain-text-splitters python-dotenv beautifulsoup4 chromadb sentence-transformers
```

Create a `.env` file:
```
ANTHROPIC_API_KEY=your_key_here
```

Then run any example:
```bash
python exampleAgent.py
python agentic_lyrics.py
```
