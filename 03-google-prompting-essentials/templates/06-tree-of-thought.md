# 06 — Tree of Thought Prompting

Explore multiple reasoning paths in parallel. Imagine several experts each pitching their version, then pick the strongest. Best for abstract, creative, or complex problems with many possible directions.

## Template

```
Imagine {N} different {experts / designers / strategists} are pitching their ideas to me.

Each expert will:
1. Write down one step of their thinking
2. Share it with the group
3. Continue to the next step

If any expert realizes they're wrong at any point, they leave the conversation.

THE QUESTION:
{your actual prompt}

Show me {N} suggestions in clearly different styles, from {style A} to {style B}.
```

## Example — Landing Page Image Concept

```
Imagine three different designers are pitching their designs to me.

Each designer will write down one step of their thinking, then share it with the group,
then all designers go on to the next step. If any designer realizes they're wrong at
any point, they leave the question.

THE QUESTION:
Generate an image concept that's visually energetic and features art supplies and computers.
Show me three suggestions in very different styles, from simple and minimal to detailed and complex.
```

## Combine with Chain of Thought (Pro Tip)
Add this line to make each branch show its reasoning:
```
At each step, each expert should explain their reasoning so I can give feedback before they continue.
```

## When to Use
- Brainstorming creative concepts
- Drafting outlines for long documents
- Designing characters, plots, branding
- Any decision where exploring options beats committing to one path
