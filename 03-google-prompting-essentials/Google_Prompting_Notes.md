# Google Prompting Essentials — Course Notes

> A structured summary of **Google Prompting Essentials Specialization**, a ~9-hour Coursera course on practical prompt engineering, everyday AI workflows, data analysis, presentation building, advanced prompting, and AI agents.

<img width="729" height="558" alt="截圖 2026-05-20 下午3 44 51" src="https://github.com/user-attachments/assets/63a0c61e-b7c3-4cc5-a326-62dae2a07f90" />

## Course Overview

| Item | Details |
|---|---|
| **Course** | Google Prompting Essentials Specialization |
| **Platform** | Coursera |
| **Status** | Completed |
| **Main Focus** | Prompt engineering, productivity workflows, data analysis, presentations, and AI agents |
| **Modules** | 4 |

## Module Map

| Module | Title | Core Topic |
|---|---|---|
| **1** | Start Writing Prompts Like a Pro | Prompting fundamentals |
| **2** | Design Prompts for Everyday Work Tasks | Email, writing, brainstorming, tables, summaries |
| **3** | Speed Up Data Analysis and Presentation Building | Spreadsheets, data insights, slide creation |
| **4** | Use AI as a Creative or Expert Partner | Prompt chaining, reasoning, agents, expert feedback |

---

# Module 1 — Start Writing Prompts Like a Pro

<img width="728" height="559" alt="截圖 2026-05-20 下午3 45 11" src="https://github.com/user-attachments/assets/e5654cf3-c5af-45ee-8f18-4782d8d8596a" />

## What Is Prompting?

**Prompting** means giving specific instructions to a generative AI tool to get new information or achieve a desired outcome.

AI input and output can include:

- Text
- Images
- Video
- Sound
- Code

---

## The 5-Step Prompting Framework

**Task → Context → References → Evaluate → Iterate**
(mnemonic: *Thoughtfully Crafting Really Excellent Inputs*)

| Step | What It Means |
|---|---|
| **Task** | What you want the AI to do |
| **Context** | Background information that improves output quality |
| **References** | Examples or source materials that clarify what you want |
| **Evaluate** | Check whether the output matches your goal |
| **Iterate** | Refine the prompt until the output improves |

## T — Task

Clearly say what you want the AI to do.

### Example

```text
Suggest an anime-related birthday gift.
```

### Improve with persona

```text
Act as an anime expert. Suggest an anime-related birthday gift.
```

### Improve with format

```text
Organize the gift ideas into a table.
```

---

## C — Context

More relevant context usually leads to better output.

Include:

- Audience
- Goal
- Constraints
- Background information
- Preferences

### Example

```text
My friend is turning 29. Her favorite anime are Solo Leveling, Naruto, and Shangri-La Frontier.
```

---

## R — References

Give examples or source material when:

- Tone is hard to describe
- You want AI to match a style
- You have past examples
- You want a specific structure

### Example

```text
Here are birthday gifts she liked before. Suggest similar but more creative options.
```

---

## E — Evaluate

After getting the output, check:

- Is it accurate?
- Is it useful?
- Does it match my goal?
- Is the tone right?
- Is anything missing?

---

## I — Iterate

Prompting is usually **not one-shot**.

Think of it like debugging code:

| Programming Analogy | Prompting Equivalent |
|---|---|
| Initial implementation | First prompt |
| Test result | AI output |
| Bug fixing | Prompt refinement |
| Working program | Final useful output |

**Motto:** ABI — **Always Be Iterating**

---

## Two Ways to Elevate the Task

| Method | Meaning | Example |
|---|---|---|
| **Persona** | Give AI a role | “Act as an anime expert.” |
| **Format** | Define output structure | “Organize this into a table.” |

---

## The 4 Iteration Methods

Use these when the first prompt gets you close, but not quite there.
(mnemonic: *Ramen Saves Tragic Idiots*)

| Method | Meaning |
|---|---|
| **Revisit the framework** | Add missing task, context, references, persona, or format |
| **Shorten into smaller sentences** | Break complex prompts into clearer steps |
| **Try different phrasing / analogous task** | Reframe the task to get better output |
| **Introduce constraints** | Narrow the scope to improve quality |

### Example: Analogous Task

Instead of:

```text
Write a marketing plan.
```

Try:

```text
Write a story about how this product fits into the daily life of our target customer.
```

---

## Multimodal Prompting

You can prompt with more than text.

| Input Type | Example Use |
|---|---|
| **Image** | Suggest recipes from a fridge photo |
| **Audio** | Use a music clip to inspire story atmosphere |
| **Video** | Analyze visual content or actions |
| **Code** | Debug or explain a program |

The same framework still applies:

```text
Task + Context + References + Evaluate + Iterate
```

---

## Two Major Risks of AI Tools

| Risk | Meaning | Solution |
|---|---|---|
| **Hallucination** | AI gives incorrect, inconsistent, or nonsensical output | Verify important information |
| **Bias** | AI reflects human biases from training data | Review output critically |

**Best practice:** keep a **human-in-the-loop**. Accuracy is still your responsibility.

---

# Module 2 — Design Prompts for Everyday Work Tasks

<img width="726" height="563" alt="截圖 2026-05-20 下午3 46 12" src="https://github.com/user-attachments/assets/faa4207d-8eb3-4dac-8bfe-ca9a7a4ab311" />

## Main Idea

Module 2 applies the core prompting framework to daily work tasks.

Common use cases:

- Writing emails
- Brainstorming
- Building tables
- Summarizing documents
- Drafting essays, articles, newsletters, or reports

---

## Key Tip

For low-stakes writing, a basic prompt may be enough.

For important writing, be specific about:

- Tone
- Word choice
- Audience
- Purpose
- Format
- Past writing references

---

## Email Prompt Formula

```text
Act as [role].

Write an email to [audience].

Purpose:
[goal]

Important details:
[specific details]

Tone:
[tone]

Length:
[short / detailed]

Format:
[easy to skim / bullet points / table]
```

### Example

```text
I am a gym manager.

Write an email informing staff about the new gym schedule.

Highlight that the Monday/Wednesday/Friday Cardio Blast class changed from 7:00 a.m. to 6:00 a.m.

Make the email professional, friendly, and short so the reader can skim it quickly.
```

---

## Tone Control

Avoid vague tone instructions.

| Quality | Prompt |
|---|---|
| **Weak** | Write it casually. |
| **Better** | Write it in a friendly, easy-to-understand tone, like explaining to a curious friend. |
| **Best** | Match the tone of the examples below. |

**Key idea:** specific tone descriptions work better than generic adjectives.

---

## Use References to Match Style

You can give AI:

- Past emails
- Previous articles
- Sample newsletters
- Writing examples
- Brand voice examples

### Example

```text
Use the same tone and structure as the examples below.
```

This helps the output sound like **you**, not generic AI writing.

---

## Prompt Library

A **prompt library** is a reusable collection of prompts that work well.

| Category | Example |
|---|---|
| **Email** | Schedule changes, announcements, follow-ups |
| **Summary** | Meeting notes, articles, PDFs |
| **Brainstorming** | Project ideas, titles, slogans |
| **Tables** | Comparison tables, planning tables |
| **Writing** | Essays, newsletters, reports |

---

## Best Practical Prompt Template

```text
Act as [role].

Task:
Write / summarize / brainstorm / organize [specific task].

Context:
[audience, situation, background, goal]

Important details:
- [detail 1]
- [detail 2]
- [detail 3]

Tone:
[specific tone description]

Output format:
[bullets / table / email / paragraph / checklist]

Constraints:
[length, must include, must avoid]
```

---

## Module 2 Core Takeaways

- Daily work prompting is structured instruction.
- Give AI a role, task, context, and format.
- Use specific tone descriptions, not vague words.
- Provide references when style matters.
- Build a reusable prompt library.
- Always review and iterate.

---

# Module 3 — Speed Up Data Analysis and Presentation Building

<img width="728" height="557" alt="截圖 2026-05-20 下午3 46 54" src="https://github.com/user-attachments/assets/802618f5-4a51-4fd7-8503-67a299b718a9" />


## Core Flow

**Data privacy → Spreadsheet help → Data insights → Presentation building → Verify results**

| Section | What It Means |
|---|---|
| **Data Privacy** | Be careful what data you upload into AI tools |
| **Spreadsheet Help** | AI can explain formulas, create columns, and help with Excel/Google Sheets |
| **Data Insights** | AI can analyze relationships between variables and find trends |
| **Presentations** | AI can turn analysis into slide outlines, speaker notes, and visuals |
| **Verification** | AI output must still be checked manually |

---

## Rule: Protect Sensitive Data

Do **not** upload sensitive or confidential data unless you are sure it is allowed.

Risky data includes:

- Company data
- Customer data
- Financial data
- Private personal information
- Internal documents

---

## AI for Spreadsheet Tasks

AI can help when you are unfamiliar with Excel or Google Sheets.

| Task | Example Use |
|---|---|
| **Formula writing** | Calculate average sales per customer |
| **New columns** | Create derived metrics |
| **Data cleaning** | Find missing or inconsistent values |
| **Basic analysis** | Summarize trends |
| **Explanation** | Teach how a formula works |

### Example Prompt

```text
Attached is a Google Sheet of store data.

How can I create a new column in Sheets that calculates the average sales per customer for each store?
```

---

## AI for Data Insights

You can ask AI to analyze patterns in a dataset.

### Example Dataset

| Variable | Meaning |
|---|---|
| **Store area** | Size of the store |
| **Items available** | Number of available products |
| **Daily customer count** | Number of customers per day |
| **Store sales** | Total sales |

### Example Prompt

```text
Give me insights into the relationship between daily customer count, items available, and sales based on the given data.
```

### Possible Output

| Insight Type | Example |
|---|---|
| **Correlation** | Sales may increase with customer count |
| **Weak relationship** | Items available may not strongly affect sales |
| **Unexpected trend** | Bigger store does not always mean higher sales |
| **Follow-up question** | Why do some stores sell more with fewer items? |

---

## Good Data Analysis Prompt Structure

```text
I have a dataset about [topic].

Columns include:
[column names]

My goal is to understand:
[analysis goal]

Analyze:
[specific relationship]

Output as:
[summary / table / chart suggestions]
```

### Example

```text
I have store sales data with columns for store area, items available, daily customer count, and total sales.

Analyze which variables seem most related to sales.

Give me a concise summary, possible explanations, and 3 follow-up analyses.
```

---

## AI for Presentations

AI can transform raw information into a presentation.

| Presentation Task | AI Can Help With |
|---|---|
| **Slide outline** | Create slide-by-slide structure |
| **Speaker notes** | Write what to say during the presentation |
| **Visual ideas** | Suggest charts, diagrams, or images |
| **Simplification** | Make complex ideas easier to explain |
| **Audience adaptation** | Adjust tone for students, managers, clients, etc. |

---

## Good Presentation Prompt Structure

```text
Create a presentation about [topic].

Audience:
[audience]

Goal:
[goal]

Length:
[number of slides / minutes]

Tone:
[academic / professional / simple]

Include:
[required points]

Output format:
slide-by-slide outline
```

### Example

```text
Create a 6-slide presentation about store sales analysis.

Audience: business managers.

Goal: explain what factors may affect sales.

Include customer count, items available, and sales trends.

Give slide titles, bullet points, and speaker notes.
```

---

## Presentation Workflow

| Step | What To Do |
|---|---|
| **1** | Use AI to create the first structure |
| **2** | Provide source notes, reports, or data |
| **3** | Ask AI to critique and improve the slides |
| **4** | Ask AI to generate speaker notes |
| **5** | Practice once, then ask AI to shorten or polish |

---

## Presentation Template 1 — Build from Scratch

```text
Act as an expert presentation strategist and university-level teaching assistant.

Task:
Create a complete presentation outline about [TOPIC].

Context:
- Audience: [AUDIENCE]
- Presentation length: [TIME LIMIT]
- Number of slides: [SLIDE COUNT]
- Goal: [WHAT THE AUDIENCE SHOULD UNDERSTAND]
- My background: [YOUR BACKGROUND]
- Course/event context: [CLASS / MEETING / COMPETITION / WORK]

References:
Use the following materials:
[PASTE NOTES / ARTICLE / DATA / REQUIREMENTS]

Output format:
Create a table with these columns:
1. Slide number
2. Slide title
3. Key message
4. Bullet points
5. Suggested visual
6. Speaker notes

Constraints:
- Keep each slide concise.
- Avoid long paragraphs.
- Use clear logic from basic to advanced.
- Make the presentation sound professional but easy to understand.
- Add examples where useful.

Before finalizing:
Evaluate whether the flow is logical, whether each slide has one clear message, and whether the presentation fits the time limit.
```

---

## Presentation Template 2 — Turn Notes / Reports / Data into Slides

```text
Act as a presentation editor and information designer.

Task:
Transform the following material into a clear slide presentation.

Context:
- Audience: [AUDIENCE]
- Presentation length: [TIME LIMIT]
- Number of slides: [SLIDE COUNT]
- Goal: [MAIN GOAL]
- Desired tone: [TONE]

References:
Here is my source material:
[PASTE MATERIAL]

Output format:
Use this format:
| Slide | Title | Core idea | Slide bullets | Visual suggestion | Speaker notes |

Constraints:
- Do not copy long sentences directly.
- Compress the material into presentation-friendly points.
- Keep one main idea per slide.
- Highlight important concepts.
- Add transitions between slides.
- If information is missing, list what is missing.

Iteration:
After creating the first version, suggest 3 ways to improve:
1. More professional version
2. More beginner-friendly version
3. More persuasive version
```

---

## Presentation Template 3 — Improve Existing Slides

```text
Act as a strict but helpful presentation coach.

Task:
Review and improve my presentation.

Context:
- Audience: [AUDIENCE]
- Presentation length: [TIME LIMIT]
- Goal: [GOAL]
- Evaluation criteria: [RUBRIC / GRADING STANDARD]
- My concern: [WHAT YOU ARE WORRIED ABOUT]

References:
Here is my current presentation draft:
[PASTE SLIDES / OUTLINE / SCRIPT]

Analyze using tree-of-thought:
Give three different review perspectives:
1. Content logic
2. Slide design
3. Oral delivery

For each perspective, provide:
- Strengths
- Weaknesses
- Concrete fixes
- Revised example

Output format:
Use tables.

Constraints:
- Be direct and specific.
- Do not give vague advice.
- Focus on high-impact improvements.
- Keep my original meaning.
- Make the presentation sound more confident and professional.

Final step:
Give me a revised version of the weakest 2 slides.
```

---

## Module 3 Core Takeaways

- AI is useful for spreadsheet formulas, data analysis, and presentation planning.
- Always protect private or sensitive data.
- Ask specific data questions, not vague ones.
- Use AI to find patterns, but verify the numbers yourself.
- AI can turn analysis into slides, speaker notes, and visual suggestions.

**Best workflow:**

```text
Provide data context → Ask focused questions → Get insights → Build slides → Verify manually
```

---

# Module 4 — Use AI as a Creative or Expert Partner

<img width="725" height="556" alt="截圖 2026-05-20 下午3 47 36" src="https://github.com/user-attachments/assets/8955078d-e49c-429f-b1ef-73fb158cbb29" />


## Core Flow

**Prompt chaining → Chain-of-thought style reasoning → Tree-of-thought → Meta prompting → AI agents**

| Technique | What It Means | Best For |
|---|---|---|
| **Prompt Chaining** | Break one big task into multiple connected prompts | Long projects, writing, planning |
| **Chain-of-Thought Style Prompting** | Ask AI to explain reasoning step by step | Logic, decisions, debugging |
| **Tree-of-Thought Prompting** | Explore multiple possible directions before choosing | Brainstorming, strategy, design |
| **Meta Prompting** | Ask AI to help write a better prompt | When you are stuck |
| **AI Agents** | Create a role-based helper for repeated interaction | Practice, feedback, coaching |

---

## Prompt Chaining

Prompt chaining means using a sequence of prompts where each output becomes input for the next step.

### Example: Marketing a Novel

| Step | Prompt Goal |
|---|---|
| **1** | Generate 3 one-sentence summaries |
| **2** | Combine the best parts into a tagline |
| **3** | Refine the tagline |
| **4** | Create a 6-week promotional plan |

### Example Prompts

```text
Generate three options for a one-sentence summary of this novel manuscript.

The summary should match the voice and tone of the manuscript but be more catchy and engaging.
```

```text
Create a tagline that combines the previous three options.

Focus on the exciting plot twist and mystery of the book.

Make it concise and engaging.
```

---

## Chain-of-Thought Style Prompting

This means asking AI to give a concise step-by-step explanation.

| Use Case | Why Helpful |
|---|---|
| **Math / logic** | Shows reasoning path |
| **Decision-making** | Makes tradeoffs visible |
| **Debugging** | Helps locate where logic fails |
| **Planning** | Breaks large tasks into steps |

### Example

```text
Give a concise step-by-step explanation of how you reached the answer.

Focus only on the important reasoning steps.
```

---

## Tree-of-Thought Prompting

Tree-of-thought prompting asks AI to explore multiple paths before choosing one.

| Feature | Meaning |
|---|---|
| **Multiple branches** | AI explores several possible solutions |
| **Comparison** | AI evaluates strengths and weaknesses |
| **Selection** | AI recommends the best option |
| **Iteration** | You can expand the most promising branch |

### Example

```text
Give me three very different approaches to this presentation topic.

For each approach, include:
1. Main idea
2. Strengths
3. Weaknesses
4. Best use case

Then recommend the strongest approach.
```

### Example Output

| Approach | Strength | Weakness | Best For |
|---|---|---|---|
| **Story-driven** | Engaging and memorable | Less formal | General audience |
| **Data-driven** | Persuasive and concrete | Can feel dry | Business or technical audience |
| **Problem-solution** | Clear logic | Predictable structure | Academic presentation |

---

## Combining Reasoning Techniques

You can combine tree-of-thought and step-by-step explanation.

### Example

```text
Generate three different slide structures for this topic.

For each structure, explain the reasoning behind the order of slides.

Then compare them and recommend the best one for a 5-minute presentation.
```

| Technique | Role |
|---|---|
| **Tree-of-thought** | Generates multiple possible structures |
| **Step-by-step reasoning** | Explains why each structure makes sense |

---

## Meta Prompting

Meta prompting means asking AI to help you write a better prompt.

Use it when:

- You do not know how to ask
- The task is vague
- Output quality is poor
- You need a stronger prompt structure

### Example

```text
Help me write a better prompt for this task:
[Describe the task]

Before writing the final prompt, ask me what information is missing.
```

### Practical Uses

| Situation | Meta Prompt |
|---|---|
| **Need better slides** | Help me write a prompt to create a professional presentation. |
| **Need better analysis** | Help me design a prompt for analyzing this dataset. |
| **Need better feedback** | Help me write a prompt to critique my essay. |

---

## AI Agents

An AI agent is a customized AI role designed to help with a specific task or interaction.

| Agent Type | Purpose | Example |
|---|---|---|
| **Simulation Agent** | Roleplay practice | Interview simulator |
| **Expert Feedback Agent** | Critique and improve your work | Pitch coach or presentation reviewer |

---

## Agent Type 1 — Simulation Agent

Used for practicing real conversations or scenarios.

| Component | What To Include |
|---|---|
| **Persona** | What role AI should play |
| **Task** | What skill you want to practice |
| **Context** | Scenario and background |
| **Interaction rule** | How AI should talk with you |
| **Stop phrase** | Phrase that ends the simulation |
| **Feedback** | What AI should give after stopping |

### Template

```text
Act as [simulation role].

Your task is to help me practice [skill].

Context:
I am [my role].
You are [AI's role].
The scenario is [scenario].

Interaction rules:
- Ask me realistic questions.
- Respond naturally based on my answers.
- Challenge me when appropriate.
- Continue the simulation until I say "[STOP PHRASE]".

After I say "[STOP PHRASE]":
Give me feedback on:
1. Strengths
2. Weaknesses
3. Specific improvements
4. A better version of my answers
```

### Example

```text
Act as a career development interview simulator.

Your task is to help interns practice interview skills.

Context:
I am an intern preparing for a final job assessment.
You are a potential manager interviewing me.

Interaction rules:
- Ask questions about strengths, skills, and future career goals.
- Let me answer one question at a time.
- Continue the roleplay until I say "jazz hands."

After I say "jazz hands":
Give me key takeaways and skills I should improve.
```

---

## Agent Type 2 — Expert Feedback Agent

Used when you want critique from a specific expert perspective.

| Component | What To Include |
|---|---|
| **Expert persona** | Who AI should act as |
| **Your role** | Who you are in the scenario |
| **Reference material** | Brief, rubric, data, draft |
| **Feedback style** | Direct, detailed, supportive |
| **Stop phrase** | Ends the conversation |
| **Final summary** | Improvement suggestions |

### Template

```text
Act as [expert persona].

Context:
You are [expert background].
I am [my role].
I am working on [task/project].

Reference:
Use this material to evaluate my work:
[PASTE MATERIAL]

Task:
When I provide my answer/draft/pitch, critique it.
Ask follow-up questions if needed.
Continue until I say "[STOP PHRASE]."

After I say "[STOP PHRASE]":
Summarize:
1. What I did well
2. What needs improvement
3. Concrete fixes
4. A stronger revised version
```

### Example

```text
Act as my potential client.

You are the VP of advertising at a world-famous sports car company known for innovation, performance, and engineering excellence.

Context:
I am a design director pitching a new campaign to attract younger buyers.

Task:
When I give my campaign idea, critique it as the client.
Ask follow-up questions.
Continue the conversation until I say "break."

After I say "break":
Summarize how I can improve my pitch.
```

---

## General AI Agent Creation Framework

| Step | What To Do | Example |
|---|---|---|
| **1. Assign persona** | Define who AI is | Act as a fitness trainer |
| **2. Give context** | Explain the situation | I want to improve my lifestyle |
| **3. Define interaction** | Tell AI how to respond | Ask me about workouts and meals |
| **4. Set stop phrase** | Decide how to end | Stop when I say “no pain no gain” |
| **5. Request feedback** | Ask for a final summary | Give advice and improvement areas |

---

## Module 4 Core Takeaways

- Use prompt chaining for complex tasks.
- Use step-by-step explanation when reasoning matters.
- Use tree-of-thought when you need multiple options.
- Use meta prompting when you do not know how to write the prompt.
- Use simulation agents to practice conversations.
- Use expert feedback agents to improve work.
- The better the persona, context, and rules, the better the agent.

---

# My Personal Practice Templates

These templates adapt the course techniques to my most common AI workflows.

---

## 1. Exam Preparation with Professor Slides

```text
Act as my exam-focused technical tutor.

Task:
Help me prepare for an exam using the professor's slides.

Context:
- Course: [COURSE NAME]
- Exam type: [midterm/final/quiz]
- My current level: [beginner/intermediate/confident]
- Time left before exam: [TIME]
- My goal: understand the concepts and answer exam questions correctly

References:
I will provide professor slides, screenshots, or notes.
Use them as the primary source.
Do not add unrelated content unless it helps explain the slide.

Output format:
1. High-yield summary
2. Concept table:
   | Concept | Meaning | Why it matters | Possible exam question |
3. Step-by-step explanation of difficult parts
4. Key formulas / algorithms / definitions
5. Common mistakes
6. Practice questions with answers
7. Final cheat sheet

Constraints:
- Focus on what is likely testable.
- Explain from basic to advanced.
- Use examples.
- If something is unclear from the slides, say so.
- Do not hallucinate missing slide content.

Iteration:
After the first explanation, ask me:
"Do you want me to quiz you, explain harder, or make a cheat sheet?"
```

---

## 2. Learning from Official Documentation

```text
Act as a senior software engineer and technical documentation tutor.

Task:
Teach me the following topic using the official documentation.

Context:
- Topic/tool: [TOPIC]
- My background: CS student familiar with [LANGUAGES / TOOLS]
- My goal: [understand concept / implement feature / debug issue / prepare for internship]
- Depth: [beginner-friendly / practical / advanced]

References:
Use this official documentation as the source:
[PASTE LINK OR DOC CONTENT]

Output format:
1. What this tool/concept is
2. Why it exists
3. Core mental model
4. Important terms table:
   | Term | Meaning | Example |
5. Minimal working example
6. Common use cases
7. Common mistakes
8. How I should learn/practice it
9. Summary in 5 bullets

Constraints:
- Prioritize official docs over blog-style assumptions.
- Explain with practical code examples.
- Use CS/system-design analogies.
- If the docs are ambiguous, clearly say what is uncertain.
- Do not skip prerequisites.

Prompt chaining:
First explain the concept.
Then give examples.
Then quiz me.
Then help me build a small project using it.
```

---

## 3. Career Planning

```text
Act as a strategic CS career advisor specializing in US graduate school, AI/software engineering careers, and Taiwan-to-US career planning.

Task:
Help me make a career decision or plan a roadmap.

Context:
- Current status: [YEAR / SCHOOL / MAJOR]
- GPA: [GPA]
- Experience: [INTERNSHIPS / PROJECTS / RESEARCH]
- Skills: [TECH STACK]
- Career goal: [GOAL]
- Target location: [US/Taiwan/other]
- Time horizon: [1 year / 3 years / 5 years]
- Constraints: [money, military service, visa, family, relationship, time]

References:
Use my background and any resume/project information I provide:
[PASTE RESUME / PROFILE / OPTIONS]

Output format:
1. Situation diagnosis
2. Best strategic direction
3. Option comparison table:
   | Option | Upside | Risk | Difficulty | Expected value |
4. Recommended roadmap:
   - Next 1 month
   - Next 3 months
   - Next 6 months
   - Next 1 year
5. Skills/projects to prioritize
6. Risks and mitigation
7. Final recommendation

Constraints:
- Be realistic, not motivational fluff.
- Explain tradeoffs clearly.
- Prioritize high-ROI actions.
- If data is uncertain, state assumptions.
- Give concrete next steps.
```

---

## 4. Social Advice with Partner or Friends

```text
Act as a calm, emotionally intelligent communication coach.

Task:
Help me understand this social situation and decide how to respond.

Context:
- Relationship: [partner/friend/classmate/teammate]
- Situation: [WHAT HAPPENED]
- My feeling: [HOW I FEEL]
- Other person's possible feeling: [YOUR GUESS]
- My goal: [comfort / apologize / explain / set boundary / discuss future]
- Desired tone: natural, sincere, respectful, not too dramatic

References:
Here is the conversation or message:
[PASTE MESSAGE]

Output format:
1. Situation summary
2. What the other person may be feeling
3. What I may be feeling
4. Main communication goal
5. Suggested response
6. Why this response works
7. Alternative versions:
   | Version | Best for |
   | Soft | comforting |
   | Direct | serious conversation |
   | Playful | light mood |
8. What not to say

Constraints:
- Use my natural tone.
- Do not sound robotic or overly formal.
- Do not manipulate or guilt-trip.
- Be emotionally honest but not overwhelming.
- If I am overthinking, tell me gently.
```

---

# Final Cheat Sheet

## Prompt Formula

```text
Act as [role].

Task:
[What I want]

Context:
[Background, goal, audience, constraints]

References:
[Examples, files, links, past work]

Output format:
[Table / bullets / checklist / code / email / slides]

Constraints:
[Must include / must avoid / length / style]

Evaluate:
Before finalizing, check whether the answer is accurate, useful, specific, and actionable.
```

## Best Workflow

```text
Start simple → Add context → Add references → Specify format → Evaluate → Iterate
```

## Core Takeaway

Good prompting is not just writing a question.  
It is designing a clear instruction system that tells AI:

1. What role to play
2. What task to complete
3. What context to use
4. What examples to follow
5. What output format to produce
6. How to improve through iteration

---

# Repository Purpose

This repository documents my learning from Google Prompting Essentials and organizes practical prompt templates that I can reuse for:

- Exam preparation
- Technical documentation learning
- Presentation building
- Data analysis
- Career planning
- Social communication
- AI-assisted productivity workflows

---

# License / Disclaimer

These notes are my personal learning summary and practice templates.  
They are not official Google or Coursera course materials.
