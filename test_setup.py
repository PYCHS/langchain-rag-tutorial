from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-6")
response = llm.invoke("hey i just charged by you with 5 USD, so how many times can i call you like this with 5 USD?")
print(response.content)