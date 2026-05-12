from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()                     # reads .env into environment variables
client = OpenAI()                 # automatically picks up OPENAI_API_KEY env var

print("🐛 Intel GPU Debug Helper — ask me anything (type 'quit' to exit)\n")
# Step 1 — Your First Real Prompt
while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("quit", "exit"):
        print("Goodbye!")
        break
    if not user_input:           
        continue
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=user_input
        )
        print(f"Assistant: {response.output_text}\n")
    except Exception as e:
        print(f"⚠️  Error: {e}\n")

# print(response.output_text)