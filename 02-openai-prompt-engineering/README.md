# 02 — OpenAI Prompt Engineering

Notes and small practice scripts based on the [OpenAI Prompt Engineering Guide](https://developers.openai.com/api/docs/guides/prompt-engineering). Part of the broader [LLM Engineering Journey](../) — picking up where [01-langchain-rag-tutorial](../01-langchain-rag-tutorial/) (built during my Intel internship learning) left off, and going further into the rest of the LLM stack.

The OpenAI guide is mostly conceptual rather than code-heavy, so this README is the main artifact: a concise summary of the ideas, paired with minimal scripts that exercise them. The running example throughout is an **Intel GPU debugging assistant** — same domain as module 01, but built directly on the OpenAI Responses API instead of LangChain.

## What the OpenAI guide covers

### 1. Choose the right model
- **GPT models** (e.g. `gpt-4.1`, `gpt-4.1-mini`): fast and cheap; respond best to **explicit, detailed instructions**.
- **Reasoning models** (e.g. `o`-series): better at multi-step problems via internal chain-of-thought; respond best to **high-level goals**, not micromanagement.
- Pick the smallest model that meets quality; larger = broader knowledge but slower/more expensive.
- In production, **pin a model snapshot** (e.g. `gpt-4.1-2025-04-14`) so behavior doesn't drift when OpenAI updates the alias.

### 2. Message roles and the chain of command
The Responses API takes a list of messages, each with a role. The role establishes **priority**:

| Role | Purpose | Priority |
|------|---------|----------|
| `developer` | System rules, identity, behavior policy | highest |
| `user` | End-user input | medium |
| `assistant` | Model's prior responses | lowest |

Mental model from the guide: **developer message = function definition, user message = the arguments**. The developer message defines what the assistant *is*; the user message is just one call into it.

This hierarchy is also the first line of defense against prompt injection — when a user message tries to override the developer message ("ignore your previous instructions…"), the model is trained to favor the higher-priority role. Not foolproof, but real.

### 3. Prompt structure (recommended order)
Inside the developer message, the guide recommends this rough layout:

1. **Identity** — who the assistant is, tone, audience.
2. **Instructions** — explicit rules and dos/don'ts.
3. **Examples** — 2–3 input/output pairs (few-shot learning).
4. **Context** — domain-specific or proprietary information.

Use **Markdown headings** and/or **XML tags** to separate sections — models follow structured prompts more reliably than wall-of-text prompts.

### 4. Few-shot learning
You don't always need fine-tuning. Showing 2–3 diverse input/output examples in the prompt is often enough for the model to pick up the pattern. Diversity matters more than quantity — examples should cover the edge cases you care about.

### 5. Retrieval-Augmented Generation (RAG)
When the answer depends on private or recent data, attach the relevant context to the prompt (via vector search, file search tools, etc.) instead of relying on the model's training data. Module 01 covers this in depth — here it's just acknowledged as one of the levers.

### 6. Model-specific prompting style
- **GPT-5 / GPT-4.1**: highly steerable, give precise logic and data.
- **Reasoning models**: treat like a senior colleague — describe the goal and the constraints, then trust them with the *how*.

### 7. Operational concerns
- **Build evals** before tuning prompts. Without measurement, "better prompt" is vibes.
- **Position reusable context early** in the prompt — earlier tokens benefit from prompt caching, which cuts cost and latency.
- **Watch the context window** — GPT-4.1 supports up to ~1M tokens, but cost and latency scale with input size.
- **Reusable prompt templates** with placeholders (`{{user_name}}`) make iteration cheaper.

## Practice scripts

| File | Concept exercised |
|------|-------------------|
| [step0_smoke_test.py](step0_smoke_test.py) | Minimal `client.responses.create()` call — verify SDK + API key wiring. |
| [step1_basic.py](step1_basic.py) | Interactive REPL with only a `user` message — establishes the baseline before adding structure. |
| [step2_developer_message.py](step2_developer_message.py) | Adds a `developer` message defining identity, scope, and behavior rules. Demonstrates the role hierarchy in action. |
| [step3_structured.py](step3_structured.py) | Iterates on the developer prompt with explicit sections (identity / topics / behavior rules) and bait-test ideas in comments for chain-of-command robustness. |

The Intel-GPU-debug-assistant scenario was chosen on purpose: it forces the prompt to **define scope** (refuse off-topic), **set tone** (technical audience), and **constrain behavior** (ask one focused question instead of guessing) — all things you can't get from prompting alone without thinking about structure.

## Setup

Make sure the repo-root `.env` has:
```
OPENAI_API_KEY=your_key_here
```

Install the SDK:
```bash
pip install openai python-dotenv
```

Run any step:
```bash
python step0_smoke_test.py
python step2_developer_message.py
```

## What I'd revisit next

- **Evals before prompts.** The guide is explicit that prompts without evals are guesswork. A small eval harness (even 10 hand-written test cases) belongs before any further prompt tweaking.
- **Few-shot examples** inside the developer prompt — none of the current scripts demonstrate this yet.
- **Prompt caching** — measure the latency/cost difference of putting the long developer prompt first vs. last.
