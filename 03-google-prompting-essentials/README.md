# 03 — Google Prompting Essentials

Notes and reusable prompt templates from Google's Prompting Essentials course (Coursera, ~9 hours). 
[My notion notes: Google Prompting Essentials Specialization](https://www.notion.so/Google-Prompting-Essentials-Specialization-3650e208cd7f80339e21d741a6fa3bed?source=copy_link)

- **[Google_Prompting_Notes.md](Google_Prompting_Notes.md)** — full course notes: the 5-step framework, all 4 modules, reasoning techniques (chain-of-thought, tree-of-thought, meta prompting), AI agents, and my personal practice templates.
- **[templates/](templates/)** — 10 reusable prompt templates below, each demonstrating one technique with `{placeholders}` to swap in for your own use case.

## The Core Framework

Every good prompt builds on **Task → Context → References → Evaluate → Iterate**
(mnemonic: *Thoughtfully Crafting Really Excellent Inputs*).

When a prompt isn't quite right, try the **4 iteration methods** (*Ramen Saves Tragic Idiots*):
1. **R**evisit the framework — add persona, context, references
2. **S**horten — break into smaller sentences
3. **T**ry different phrasing or an analogous task
4. **I**ntroduce constraints

## Templates in This Repo

| # | Template | Technique |
|---|----------|-----------|
| 01 | [Basic Framework](templates/01-basic-framework.md) | Task + Persona + Format |
| 02 | [Full Framework with Context & References](templates/02-full-framework.md) | All 5 steps |
| 03 | [Multimodal Prompting](templates/03-multimodal.md) | Image / audio / video input |
| 04 | [Prompt Chaining](templates/04-prompt-chaining.md) | Sequential prompts |
| 05 | [Chain of Thought](templates/05-chain-of-thought.md) | Step-by-step reasoning |
| 06 | [Tree of Thought](templates/06-tree-of-thought.md) | Multiple parallel paths |
| 07 | [Meta Prompting](templates/07-meta-prompting.md) | AI helps write your prompt |
| 08 | [Agent Sim (Simulation)](templates/08-agent-sim.md) | Role-play scenarios |
| 09 | [Agent X (Expert Feedback)](templates/09-agent-x.md) | Personal tutor / consultant |
| 10 | [Iteration Techniques](templates/10-iteration.md) | Refining stuck prompts |

## How to Use

1. Pick the template that matches your task.
2. Replace `{placeholders}` with your specifics.
3. Iterate — prompting is rarely one-and-done. **ABI: Always Be Iterating.**
4. Keep a human in the loop. Always verify outputs for hallucinations and bias.

## Credits

Templates adapted from Google's Prompting Essentials course.
