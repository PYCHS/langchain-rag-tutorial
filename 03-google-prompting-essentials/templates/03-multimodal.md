# 03 — Multimodal Prompting

For prompts that include images, audio, video, or code as input. The framework stays the same — just be specific about what kind of input you're attaching and what to do with it.

## Template

```
Act as {persona}.

I'm attaching {type of input: image / audio / video / code}.

TASK:
{what to do with the attached input}

CONTEXT:
- {how the AI should interpret the input}
- {any relevant background}
- Tone / style: {desired feel}

FORMAT:
{how the output should be structured}
```

## Example — Social Media Post from an Image

```
Act as a social media marketer for a small creative business.

I'm attaching an image of my new nail art collection.

TASK:
Write a social media post featuring this image.

CONTEXT:
- The post should be fun and short
- Highlight that this is a new collection I'm selling
- Audience: Instagram followers, mostly women aged 18–35
- Tone: playful and trendy

FORMAT:
- One caption (under 100 words)
- 5–7 relevant hashtags
- A clear call-to-action
```

## Other Multimodal Ideas
- **Photo of fridge ingredients** → recipe suggestions
- **Brand logo + colors** → digital teaser for an event
- **Music clip** → atmospheric details for a story chapter
- **Screenshot of error** → debugging help
