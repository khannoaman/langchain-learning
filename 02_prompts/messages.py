from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
messages = [
    SystemMessage(content="You are a helpful assistant that provides information about Data Science. Answer the user's questions in a concise manner."),
]

while True:
    user_input = input("User: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("Assistant: Exiting the chatbot. Goodbye!")
        break

    messages.append(HumanMessage(content=user_input))
    response = model.invoke(messages)
    print(f"Assistant: {response.content}")
    messages.append(AIMessage(content=response.content))


print("Chatbot session ended.")

print(messages)

