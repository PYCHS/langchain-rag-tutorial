# 10 — Iteration Techniques (Ramen Saves Tragic Idiots)

When your prompt gets you ~80% there but you need the final 20%. Pick the method that matches your symptom.

---

## Method 1 — Revisit the Framework
**Symptom:** Output is generic, vague, or off-tone.
**Fix:** Add persona, context, or references that were missing.

```
Add to your previous prompt:
- Persona: "Act as {role}..."
- Context: "{audience, tone, constraints, background}"
- References: "Here are examples of what I'm looking for: {example 1, example 2}"
```

---

## Method 2 — Shorten Into Smaller Sentences
**Symptom:** Output feels confused, misses pieces of a long prompt.
**Fix:** Break one wall-of-text prompt into a sequence of small ones.

**Before:**
```
Write a marketing plan for my new app that targets freelance designers and writers
and emphasizes the AI brainstorming feature while also covering pricing, launch
timing, social channels, and PR outreach all in a friendly approachable tone aimed
at people who hate marketing speak.
```

**After (broken into 3 prompts):**
```
PROMPT 1: List the 5 most important messages for a marketing plan targeting freelance
designers and writers, given that my app's standout feature is AI brainstorming.

PROMPT 2: For each of those messages, suggest the best channel (social, email, PR,
content) and timing.

PROMPT 3: Take the above and write a friendly, jargon-free 1-page marketing plan.
```

---

## Method 3 — Try Different Phrasing or an Analogous Task
**Symptom:** Output is technically correct but boring or lifeless.
**Fix:** Reframe the task as something more evocative.

**Before:**
```
Write a marketing plan for my product.
```

**After (analogous task):**
```
Marketing is really just telling a compelling story. Write a story about how my product
fits into the everyday life of a freelance designer named Maya, from the moment she
hears about it to the moment she can't imagine working without it.
```

---

## Method 4 — Introduce Constraints
**Symptom:** Output is too broad or generic. You said "anything" and got mush.
**Fix:** Add tight constraints to narrow the focus.

**Before:**
```
Generate a road trip playlist.
```

**After:**
```
Generate a road trip playlist with these constraints:
- 20 songs
- Released between 1995 and 2010
- Tempo between 90 and 130 BPM
- Theme: heartbreak and reinvention
- No song over 5 minutes long
- No artist repeated
```

---

## The ABI Reminder
**Always Be Iterating.** A great prompt is rarely written on the first try — it's refined through 2–5 rounds of feedback. That's not a failure of prompting; that *is* prompting.
