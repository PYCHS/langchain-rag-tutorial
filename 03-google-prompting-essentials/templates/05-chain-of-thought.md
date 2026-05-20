# 05 — Chain of Thought Prompting

Ask the AI to explain its reasoning as a step-by-step process. Helps you spot where its logic goes wrong and improves decision-making.

## Template

```
Act as {persona}.

TASK:
{what you want the AI to do}

CONTEXT:
{relevant background}

REQUIREMENT:
Before giving your final answer, explain your thought process step by step.
Show your reasoning at each stage so I can see how you arrived at the conclusion.

FORMAT:
1. Reasoning (numbered steps)
2. Final answer
```

## Example — Choosing a Pricing Strategy

```
Act as a SaaS pricing strategist.

TASK:
Recommend a pricing model for my new productivity app aimed at freelancers.

CONTEXT:
- Target users: freelance designers and writers
- Competitors charge $10–$25/month
- My app has one premium feature (AI brainstorming) that competitors lack
- I want strong word-of-mouth growth in the first 6 months

REQUIREMENT:
Explain your thought process step by step. Walk me through how you're weighing
each factor before reaching your recommendation.

FORMAT:
1. Reasoning (numbered steps)
2. Final pricing recommendation with rationale
```

## When to Use
- Strategic decisions
- Math or logic problems
- When you want to *learn* from the AI's reasoning, not just take the answer
- When you suspect the AI might be hallucinating and want to audit its logic
