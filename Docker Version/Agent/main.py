from dotenv import load_dotenv
# Running this first to ensure everything else is imported correctly and
# to block hugging face from downloading more
loaded = load_dotenv(override=True) # Allows the dotenv to overwrite env variables already set in the OS environment


import uuid

# Previously had this but we're making our own custom one isntead now
# from langgraph.graph import MessagesState
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver # Lets us save context!
# Now implement RAG (vector stores to hold the data for the RAG)
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
# Add 'Human In The Loop'
from langgraph.types import interrupt, Command


# Needed to run the coder
import subprocess
import os

# Now set the directory to the workspace so all files saved and written will default to there
workspace = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workspace')
os.makedirs(workspace, exist_ok=True)
os.chdir(workspace)



if os.getenv('HF_HUB_DISABLE_PROGRESS_BARS'):    
    # Block the hugging face logging bars
    from transformers.utils import logging as hf_logging
    hf_logging.disable_progress_bar()

# And import embeddings
from langchain_huggingface import HuggingFaceEmbeddings # Using local embeddings to keep offline (downloaded transformers from huggingface)


KNOWLEDGE = [
    "Tom is a programmer",
    "LangGraph is a library for building AI agents"
]


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents([Document(page_content=text) for text in KNOWLEDGE])


llm = init_chat_model(
    os.getenv("LLAMA_CPP_MODEL"),
    model_provider="openai",
    base_url=os.getenv("LLAMA_CPP_BASE_URL"),
    api_key=os.getenv("LLAMA_CPP_API_KEY"),
)

class IntentClassifier(BaseModel):
    message_intent: Literal['chat', 'knowledge', 'code'] = Field(..., description='Classify whether the user wants to just chat, ask for knowledge or change code in the project')

class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_intent: str | None

    # Used for human in the loop - approval or denial
    next_node: str | None

def classify_intent(state: State):
    structured_llm = llm.with_structured_output(IntentClassifier)

    result = structured_llm.invoke([
        {'role': 'system', 'content': 'Determine / classify whether the user wants to chat ("chat"), retrieve knowledge ("knowledge") or change code ("code").'},
        {'role': 'user', 'content': state['messages'][-1].content}
    ])

    return {'message_intent': result.message_intent}

def prompt_llm_chat(state: State):
    messages = [
             {'role': 'system', 'content': 'You are a talkative chatbot for fun. Be nice'}
        ] + state['messages']

    response = llm.invoke(messages)

    return {'messages': [{'role': 'assistant', 'content':response.content}]}

def prompt_llm_rag(state: State):
    query = state['messages'][-1].content
    documents = vector_store.similarity_search(query, k=3) # Retrieve 3 most relevant answers

    context = '\n'.join(f'- {doc.page_content}' for doc in documents)

    messages = [
             {'role': 'system', 'content': f'You are a RAG agent. Answer the user using only the context below. If the anwer is not in it, say you don\'t know. \n\nContext:\n{context}'}
        ] + state['messages'] 

    response = llm.invoke(messages)

    return {'messages': [{'role': 'assistant', 'content':response.content}]}

def prompt_claude_code(state: State):
    user_prompt = state['messages'][-1].content
    workspace = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workspace')

    result = subprocess.run(
        # Uses Claude code on Allow Edits mode, headless so we don't see the console
        ['claude', '-p', user_prompt, '--permission-mode', 'acceptEdits'],
        cwd=workspace,
        capture_output=True,
        text=True
    )

    # Get the output if there is one, otherwise return the error message
    output = result.stdout.strip() or result.stderr.strip()

    return {'messages': [{'role': 'assistant', 'content':output}]}

def prompt_llm_code(state: State):
    user_prompt = state['messages'][-1].content
    workspace = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workspace')

    result = subprocess.run(
        [
            'pi', '-p', user_prompt,
            '--provider', os.getenv('PI_PROVIDER'),
            '--model', os.getenv('PI_MODEL'),
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
    )

    output = result.stdout.strip() or result.stderr.strip()
    return {'messages': [{'role': 'assistant', 'content': output}]}

# Accept Coding NODE
def accept_coding(state: State):
    user_prompt = state['messages'][-1].content
    decision = interrupt(f'About to run the coder with request:\n\n{user_prompt}\n\nApprove? (yes/no, or type a revised request)')

    text = str(decision).strip().lower()

    # Accept
    if text in ['yes', 'y', 'approve', 'ok']:
        return {'next_node': 'coding_agent'} # send it to the coding agent because we approved
    # Deny
    if text in ['no', 'n', 'deny', 'cancel']:
        return {'messages': [{'role': 'assistant', 'content': 'Coding request was denied by the user'}], 'next_node': 'denied'}

    # Just reprompt the LLM instead (since the user has sent a revised prompt instead) and send this back to the accept_coding node
    return {'messages': [{'role': 'user', 'content': text}], 'next_node': 'accept_coding'}

# Node that takes your context and your coding request and writes an appropriate coding request to send to the coder LLM
def prepare_coding_request(state: State):
    messages = [
        {
            'role': 'system',
            'content': 'Rewrite the latest user coding request into a clear instruction for Qwen Coder. Use the conversation history as context. Only output the instruction, no explanation.'
        }
    ] + state['messages']
    response = llm.invoke(messages)

    return {'messages': [{'role': 'user', 'content': response.content}]}

# Entire Graph
graph_builder = StateGraph(State)

# NODES
graph_builder.add_node('classifier', classify_intent)
graph_builder.add_node('chat_agent', prompt_llm_chat)
graph_builder.add_node('rag_agent', prompt_llm_rag)
graph_builder.add_node('coding_agent', prompt_llm_code)
graph_builder.add_node('prepare_coding_request', prepare_coding_request) # Instead of going directly from coding prompt to accept coding, first redirect to get the appropriate prompt written out for the users approval
graph_builder.add_node('accept_coding', accept_coding)

# EDGES
graph_builder.add_edge(START, 'classifier')
graph_builder.add_edge('chat_agent', END)
graph_builder.add_edge('rag_agent', END)
graph_builder.add_edge('prepare_coding_request', 'accept_coding')
graph_builder.add_edge('coding_agent', END)

# Now the conditional edges!
graph_builder.add_conditional_edges(
    'accept_coding', # The origin node (where this edge is coming *from*)
    lambda state: 'end' if state.get('next_node') == 'denied' else state['next_node'], # Returns a label (the key for the path_map below) which gets used to find its destination node (from the path map below)
    {
        # key: value - the key being used and the destination node that key links to
        'end': END, # Destination node name ('end' is same as the lambda state above, END being the actual destiantion node)
        'coding_agent': 'coding_agent', # so key of coding agent should have the destination of coding agent (we're being told coding agent is where we want to go, then we're setting the destination as the coding agent)
        'accept_coding': 'prepare_coding_request' # Loop back to prepare a coding request
    }
)
graph_builder.add_conditional_edges(
    'classifier',
    lambda state: state['message_intent'],
    {
        'chat': 'chat_agent',
        'knowledge': 'rag_agent',
        'code': 'prepare_coding_request' # Was previously coding_agent but we want to get an appropriate coding prompt written first then route all traffic through the human acceptance first
    }
)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

# To visualize our graphs
graph.get_graph().draw_mermaid_png(output_file_path = "graph.png")

config={'configurable': {'thread_id': uuid.uuid4()}}

while True:
    user_message = input('Enter message:')
    result = graph.invoke({'messages': [{'role': 'user', 'content': user_message}]}, config=config)

    while '__interrupt__' in result:
        prompt = result['__interrupt__'][0].value
        decision = input(f'{prompt}\n> ')
        result = graph.invoke(Command(resume=decision), config=config)

    print(result['messages'][-1].content)