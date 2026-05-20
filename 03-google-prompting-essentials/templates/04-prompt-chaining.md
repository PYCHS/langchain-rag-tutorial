# 04 — Prompt Chaining

Guide the AI through a series of interconnected prompts, where each one builds on the previous output. Best for complex projects where the final goal requires several layers of refinement.

## Template

```
PROMPT 1 — Generate Options
Act as {persona}.
Based on {attached material / context}, generate {N} options for {first goal}.
Each option should be {criteria}.

PROMPT 2 — Refine / Combine (after seeing output 1)
Take the previous options and {combine / refine} them.
Focus on {specific aspect}.
The result should be {criteria}.

PROMPT 3 — Expand (after seeing output 2)
Using the result from above, now {next action}.
Include {specific elements}.

PROMPT 4 — Finalize
Take everything above and produce a final {deliverable} that includes {details}.
```

## Example — Marketing Plan for a Novel

```
PROMPT 1:
You're a book marketing expert. I've attached my novel manuscript.
Generate three options for a one-sentence summary of this novel.
The summary should be similar in voice and tone to the manuscript but more catchy and engaging.

PROMPT 2 (after seeing the three options):
Create a tagline that combines the previous three options, with a special focus on the
exciting plot twist and mystery of the book. Find the catchiest and most impactful combination.
The tagline should be concise and leave the reader hooked.

PROMPT 3 (after picking a tagline):
Using that tagline, now write three social media post variants (Instagram, X, LinkedIn)
that announce the book launch in a way that matches the tagline's tone.

PROMPT 4 (final):
Generate a six-week promotional plan for the book tour, including locations and the
channels to promote each stop. Use the tagline and social posts above to inform the messaging.
```

## Tip
Use a tool with a long context window (like Google AI Studio or Claude) when chaining over large source material like manuscripts, transcripts, or documents.
