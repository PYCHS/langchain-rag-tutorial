# 02 — OpenAI Prompt Engineering

Practicing prompt engineering patterns with the OpenAI API. Part of the broader [LLM Engineering Journey](../) — picking up where [01-langchain-rag-tutorial](../01-langchain-rag-tutorial/) (built during my Intel internship learning) left off, and going further into the rest of the LLM stack.

## Goals

- Get hands-on with the raw OpenAI API (no framework abstractions in the way).
- Learn prompt patterns that actually move output quality: role/system prompts, structured output, few-shot examples, chain-of-thought, self-consistency.
- Build intuition for *why* a prompt works, not just *which* one worked.

## Status

In progress. Examples will land in `main.py` and additional files as I work through each pattern.

## Setup

Make sure the repo-root `.env` has:
```
OPENAI_API_KEY=your_key_here
```

Install the SDK:
```bash
pip install openai python-dotenv
```

Run:
```bash
python main.py
```
