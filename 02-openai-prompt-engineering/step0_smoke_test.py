from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()                     # reads .env into environment variables
client = OpenAI()                 # automatically picks up OPENAI_API_KEY env var
# Step 0 — Environment Setup
# Use client.responses.create(model=..., input=...)
response = client.responses.create(
    model="gpt-4.1-mini",
    input="Say 'hello, Intel GPU world' and nothing else."
)

print(response.output_text)