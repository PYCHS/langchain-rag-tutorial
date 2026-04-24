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

## RAG Overview

RAG has two main phases:

### 1. Indexing (offline, done once)

**Load** → **Split** → **Store**

- **Load**: pull raw documents in using loaders (web pages, PDFs, CSVs). In this project I used `WebBaseLoader` with BeautifulSoup to scrape only the relevant HTML sections.
- **Split**: break documents into smaller chunks using `RecursiveCharacterTextSplitter`. Chunking matters because embedding a 10,000-word article as one vector loses detail — smaller chunks give the retriever more precise targets.
- **Store**: embed each chunk into a vector and store it in a vector database (Chroma). Embedding converts text into a numeric representation so we can do similarity search later.

```
Raw Document → WebBaseLoader → RecursiveCharacterTextSplitter → HuggingFace Embeddings → Chroma Vector Store
```

### 2. Retrieval and Generation (online, per query)

**Retrieve** → **Generate**

- **Retrieve**: when the user asks a question, embed the query using the same embedding model, then run a similarity search against the vector store to find the most relevant chunks.
- **Generate**: pass the retrieved chunks as context to the LLM (Claude). The model answers based on that evidence rather than relying on what it memorized during training.

```
User Query → Embed Query → Similarity Search → Retrieved Chunks → LLM (Claude) → Answer
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
