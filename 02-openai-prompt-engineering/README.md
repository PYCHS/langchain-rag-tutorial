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
An LLM on its own has three real limits: a **knowledge cutoff** (it doesn't know anything after its training date), **no private knowledge** (it has never seen your team's wiki, internal docs, or codebase), and **hallucination on specifics** (ask about a niche API and it may confidently invent details). RAG fixes all three by attaching relevant documents to the prompt so the model can answer *from* them instead of *from* training data.

Exercised by hand in [step6_mini_RAG.py](step6_mini_RAG.py) (knowledge file: [knowledge.md](knowledge.md), transcript: [step6_result.txt](step6_result.txt)). Module 01 (LangChain) did the **R**etrieval (vector embeddings, similarity search); step 6 does the **A**ugmentation and **G**eneration by hand — *retrieval is an information-retrieval problem, augmentation is a prompt-engineering problem*, and that's this module. The point is to demystify what LangChain was doing: chunk docs → embed → store → embed query → similarity search → **stuff results into a prompt string** → call LLM. The last two steps are exactly what `.replace()` does here, just without the fancy retrieval step in front.

The key concepts:

- **Grounding — the most important new word.** Grounding means instructing the model to answer *from the provided context*, not from its training knowledge. This is a behavior you must **explicitly request** — paste a document and ask a question without grounding rules and the model will happily blend the document with its own training data, and you won't know which parts came from where. The rules look like: "answer using only the information in the Context section", "if the Context doesn't contain the answer, say so — do not make it up", "when you state a fact, it must be supported by the Context".
- **The "I don't know" behavior is the most valuable RAG property.** A RAG system that hallucinates is *worse* than no RAG, because it launders fabrications through an authoritative-looking interface. The single rule "*if it's not in the context, say you don't know*" is what separates a trustworthy system from a confident liar.
- **Context placement matters — "lost in the middle".** A documented phenomenon: models pay most attention to the **beginning and end** of long prompts and can miss things buried in the middle. So: static content (identity, instructions, examples) goes at the top — cacheable. Dynamic content (retrieved context, user query) goes at the bottom. And the user's actual question should be the **last** thing the model reads, after the context.
- **Structure context with XML tags and source labels.** Don't paste a raw wall of text — wrap each document like `<doc title="Intel Arc Memory Model" source="oneapi-guide-ch3">...</doc>`. The tags delimit documents cleanly, and the `title` / `source` attributes let the model cite where an answer came from ("According to the Intel Arc Memory Model doc…").
- **Use `.replace()` over f-strings for prompt templating.** When the prompt body contains literal `{` / `}` (XML attributes, JSON examples, etc.), f-strings choke. `template.replace("__KNOWLEDGE__", KNOWLEDGE)` sidesteps the escaping problem entirely.
- **Made-up facts are how you prove grounding works.** [knowledge.md](knowledge.md) contains invented details (error code `XE-4471`, Arc B580 work-group size limit of 512) the model *cannot* know from training. That's the point — if the model answers correctly, it had to have read the context; if you ask about a fake B999 GPU not in the file, it should say "the docs don't cover this" instead of inventing a number. That second test is the real graduation check.

The parallel with step 5 is the thing to internalize: "memory" isn't the model remembering — it's your code replaying history into the prompt. "Retrieval" isn't the model fetching — it's your code injecting documents into the prompt. In both cases the model is a stateless function that can only see what's in its prompt *right now*. Everything clever — memory, retrieval, tools — is your **application code** assembling the right prompt.

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
| [step6_mini_RAG.py](step6_mini_RAG.py) | Mini-RAG by hand: loads [knowledge.md](knowledge.md) into a `KNOWLEDGE` string, injects it into the developer prompt's `# Context` section via `.replace()`, and adds grounding rules under `# Instructions` (answer from context, admit when context lacks the answer, cite the source). The knowledge file deliberately contains *made-up* facts (e.g. error code `XE-4471`, Arc B580 work-group limit 512) so grounding can be proved — the model answers them correctly only because it read the context, and says "the docs don't cover this" for a fake `B999`. Sample transcript: [step6_result.txt](step6_result.txt). |

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
