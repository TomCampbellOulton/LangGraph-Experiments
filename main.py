from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, StateGraph, START, END

load_dotenv()

llm = init_chat_model('openai:qwen3.5-9b')

def prompt_llm(state: MessagesState):
    response = llm.invoke(state['messages'])

    # Appends to list of messages instead of replacing them
    return {'messages': [response]}

graph_builder = StateGraph(MessagesState)

graph_builder.add_node(prompt_llm)
graph_builder.add_edge(START, 'prompt_llm')
graph_builder.add_edge('prompt_llm', END)

graph = graph_builder.compile()

user_message = input("Enter a message:")
print(graph.invoke({'messages': [{'role': 'user', 'content': user_message}]}))