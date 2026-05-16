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
You don't always need fine-tuning. Showing 2–3 diverse input/output examples in the prompt is often enough for the model to pick up the pattern. A few things worth internalizing here (exercised in [step4_few_shot_example.py](step4_few_shot_example.py)):

- **Models follow examples more reliably than rules.** Rules are abstract; examples are unambiguous. When a rule isn't sticking ("ask ONE clarifying question, not a bulleted list"), an example demonstrating the desired behavior usually fixes it where more rule text doesn't.
- **Wrap examples in XML tags** (e.g. `<example>...<user_query>...</user_query><assistant_response>...</assistant_response></example>`). Tags act as scope markers — they tell the model *"this is illustrative data, not the live conversation"*. Without them, `My kernel is broken` inside the prompt could be read as the current user's input.
- **Diversity over quantity.** If all three examples are vague bug reports, the model concludes its job is to ask clarifying questions — and then it asks one even when the user types "what is SYCL?". Each example should teach a *different* behavior (one clarifying question, one refusal, one concept answer, one real debug answer).
- **Demonstrate behavior, not facts.** A good example for "what is SYCL?" doesn't teach the model what SYCL *is* — it teaches the model *how to answer concept questions*: concise, technical, no clingy follow-up. The model generalizes from the style.

### 5. Multi-turn conversation (you maintain state)
The single most counterintuitive thing about the Responses API: **the API has no memory**. Every call is stateless — the model only knows what you put in `input` for that specific call. ChatGPT-style "remembering" is an illusion the *application* maintains by replaying the entire conversation history on every call. Every. Single. Call. Exercised in [step5_multi_turn.py](step5_multi_turn.py).

Two ways to handle this with the Responses API:

| Concern | Option A: manual history | Option B: `previous_response_id` |
|---|---|---|
| State storage | Your app holds it | OpenAI holds it |
| Bandwidth per call | Sends full history each turn | Sends only the new message |
| Cost | Same — billed on tokens *processed*, not *sent* | Same |
| Control over context | Total (edit, trim, summarize, drop) | Limited — whatever OpenAI stored |
| Cross-restart persistence | You handle it (save to disk) | Built-in (ID is durable for a while) |
| Works on other providers | Yes — universal pattern | No — OpenAI-specific |

For learning, use Option A — it makes the "you manage state" reality concrete. Switching to B later is trivial.

A few things worth knowing:

- **The three roles each serve a purpose in history.** `developer` = your instructions (sent once, at the start of history). `user` = what the human typed. `assistant` = what the model said previously. Replaying the assistant messages is how the model stays *consistent* — it sees its own prior answers and won't contradict them.
- **The `instructions` parameter doesn't persist with `previous_response_id`.** It applies only to the current call. To carry the developer message across turns with Option B, you'd have to repeat it each call or fall back to the message-array form.
- **The unbounded-context problem.** A 100-turn conversation means a 100-turn prompt on every call: cost grows linearly, latency grows, and eventually you hit the context window and the model errors out. Production apps solve this with summarization (compress old turns), sliding window (keep only the last N turns), or importance scoring (keep what matters, drop trivia). Don't solve it prematurely — just be aware.

### 6. Retrieval-Augmented Generation (RAG)
When the answer depends on private or recent data, attach the relevant context to the prompt (via vector search, file search tools, etc.) instead of relying on the model's training data. Module 01 covers this in depth — here it's just acknowledged as one of the levers.

### 7. Model-specific prompting style
- **GPT-5 / GPT-4.1**: highly steerable, give precise logic and data.
- **Reasoning models**: treat like a senior colleague — describe the goal and the constraints, then trust them with the *how*.

### 8. Operational concerns
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
| [step4_few_shot_example.py](step4_few_shot_example.py) | Adds a `# Examples` section with XML-tagged few-shot input/output pairs, each targeting a different behavior (clarifying question, refusing creative writing, concept answer, real debug answer). |
| [step5_multi_turn.py](step5_multi_turn.py) | Adds conversational memory by maintaining a `history` list across turns (Option A: manual history). Appends `user` and `assistant` messages each turn and re-sends the full history every call. Includes `reset` (clear back to the developer message) and `history` (print message count) commands; a sample transcript is in [step5_result.txt](step5_result.txt). |

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
- **Prompt caching** — measure the latency/cost difference of putting the long developer prompt first vs. last.
