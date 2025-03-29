from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
from typing import Sequence
import requests
import re
import os
from langchain.tools.retriever import create_retriever_tool
from langchain.embeddings.base import Embeddings
import ollama

# Custom Ollama Embedding Class
class OllamaNomicEmbeddings(Embeddings):
    def __init__(self, model: str = "nomic-embed-text"):
        self.model = model

    def embed_documents(self, texts):
        return [ollama.embeddings(self.model, prompt=text)["embedding"] for text in texts]

    def embed_query(self, text):
        return ollama.embeddings(self.model, prompt=text)["embedding"]

# Create Dummy Data
research_texts = [
    "Research Report: Results of a New AI Model Improving Image Recognition Accuracy to 98%",
    "Academic Paper Summary: Why Transformers Became the Mainstream Architecture in Natural Language Processing",
    "Latest Trends in Machine Learning Methods Using Quantum Computing"
]

development_texts = [
    "Project A: UI Design Completed, API Integration in Progress",
    "Project B: Testing New Feature X, Bug Fixes Needed",
    "Product Y: In the Performance Optimization Stage Before Release"
]

# Text splitting settings
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)

# Generate Document objects from text
research_docs = splitter.create_documents(research_texts)
development_docs = splitter.create_documents(development_texts)

# Create vector stores
# embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-large",
#     # With the `text-embedding-3` class
#     # of models, you can specify the size
#     # of the embeddings you want returned.
#     # dimensions=1024
# )
embeddings = OllamaNomicEmbeddings()

research_vectorstore = Chroma.from_documents(
    documents=research_docs,
    embedding=embeddings,
    collection_name="research_collection",
    persist_directory="./chroma_db"  # 👈 Ensure persistent storage
)

development_vectorstore = Chroma.from_documents(
    documents=development_docs,
    embedding=embeddings,
    collection_name="development_collection",
    persist_directory="./chroma_db"
)

research_retriever = research_vectorstore.as_retriever()
development_retriever = development_vectorstore.as_retriever()
# Check research collection
print("Research Collection Info:", research_vectorstore._collection.count())
# Check development collection
print("Development Collection Info:", development_vectorstore._collection.count())
research_tool = create_retriever_tool(
    research_retriever,  # Retriever object
    "research_db_tool",  # Name of the tool to create
    "Search information from the research database."  # Description of the tool
)

development_tool = create_retriever_tool(
    development_retriever,
    "development_db_tool",
    "Search information from the development database."
)

# Combine the created research and development tools into a list
tools = [research_tool, development_tool]

class AgentState(TypedDict):
    messages: Annotated[Sequence[AIMessage|HumanMessage|ToolMessage], add_messages]

def agent(state: AgentState):
    print("---CALL AGENT---")
    messages = state["messages"]

    if isinstance(messages[0], tuple):
        user_message = messages[0][1]
    else:
        user_message = messages[0].content

    # Structure prompt for consistent text output
    prompt = f"""Given this user question: "{user_message}"
    If it's about research or academic topics, respond EXACTLY in this format:
    SEARCH_RESEARCH: <search terms>
    
    If it's about development status, respond EXACTLY in this format:
    SEARCH_DEV: <search terms>
    
    Otherwise, just answer directly.
    """

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer sk-1cddf19f9dc4466fa3ecea6fe10abec0",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    # response = requests.post(
    #     "https://api.deepseek.com/v1/chat/completions",
    #     headers=headers,
    #     json=data,
    #     verify=False
    # )
   
    # if response.status_code == 200:
    #     response_text = response.json()['choices'][0]['message']['content']
    #     print("Raw response:", response_text)

    response = ollama.chat(model="deepseek-r1", messages=[{"role": "user", "content": prompt}])
    
    response_text = response['message']['content']
    print("Raw response:", response_text)
        
        # Format the response into expected tool format
    if "SEARCH_RESEARCH:" in response_text:
        query = response_text.split("SEARCH_RESEARCH:")[1].strip()
        # Use direct call to research retriever
        results = research_retriever.invoke(query)
        return {"messages": [AIMessage(content=f'Action: research_db_tool\n{{"query": "{query}"}}\n\nResults: {str(results)}')]}
    elif "SEARCH_DEV:" in response_text:
        query = response_text.split("SEARCH_DEV:")[1].strip()
        # Use direct call to development retriever
        results = development_retriever.invoke(query)
        return {"messages": [AIMessage(content=f'Action: development_db_tool\n{{"query": "{query}"}}\n\nResults: {str(results)}')]}
    else:
        return {"messages": [AIMessage(content=response_text)]}
    # else:
    #     raise Exception(f"API call failed: {response.text}")

def simple_grade_documents(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    print("Evaluating message:", last_message.content)
    
    # Check if the content contains retrieved documents
    if "Results: [Document" in last_message.content:
        print("---DOCS FOUND, GO TO GENERATE---")
        return "generate"
    else:
        print("---NO DOCS FOUND, TRY REWRITE---")
        return "rewrite"

def generate(state: AgentState):
    print("---GENERATE FINAL ANSWER---")
    messages = state["messages"]
    question = messages[0].content if isinstance(messages[0], tuple) else messages[0].content
    last_message = messages[-1]

    # Extract the document content from the results
    docs = ""
    if "Results: [" in last_message.content:
        results_start = last_message.content.find("Results: [")
        docs = last_message.content[results_start:]
    print("Documents found:", docs)

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer sk-1cddf19f9dc4466fa3ecea6fe10abec0",
        "Content-Type": "application/json"
    }
    
    prompt = f"""Based on these research documents, summarize the latest advancements in AI:
    Question: {question}
    Documents: {docs}
    Focus on extracting and synthesizing the key findings from the research papers.
    """
    
    data = {
        "model": "deepseek-chat",
        "messages": [{
            "role": "user",
            "content": prompt
        }],
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    print("Sending generate request to API...")
    # response = requests.post(
    #     "https://api.deepseek.com/v1/chat/completions",
    #     headers=headers,
    #     json=data,
    #     verify=False
    # )

    response = ollama.chat(model="deepseek-r1", messages=[{"role": "user", "content": prompt}])
    
    response_text = response['message']['content']
    print("Final Answer:", response_text)
    return {"messages": [AIMessage(content=response_text)]}
    
    # if response.status_code == 200:
    #     response_text = response.json()['choices'][0]['message']['content']
    #     print("Final Answer:", response_text)
    #     return {"messages": [AIMessage(content=response_text)]}
    # else:
    #     raise Exception(f"API call failed: {response.text}")

def rewrite(state: AgentState):
    print("---REWRITE QUESTION---")
    messages = state["messages"]
    original_question = messages[0].content if len(messages)>0 else "N/A"

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer sk-1cddf19f9dc4466fa3ecea6fe10abec0",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": [{
            "role": "user",
            "content": f"Rewrite this question to be more specific and clearer: {original_question}"
        }],
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    print("Sending rewrite request...")
    # response = requests.post(
    #     "https://api.deepseek.com/v1/chat/completions",
    #     headers=headers,
    #     json=data,
    #     verify=False
    # )

    response = ollama.chat(model="deepseek-r1", messages=[{"role": "user", "content": prompt}])
    
    response_text = response['message']['content']
    print("Rewritten question:", response_text)
    return {"messages": [AIMessage(content=response_text)]}
    
    # print("Status Code:", response.status_code)
    # print("Response:", response.text)
    
    # if response.status_code == 200:
    #     response_text = response.json()['choices'][0]['message']['content']
    #     print("Rewritten question:", response_text)
    #     return {"messages": [AIMessage(content=response_text)]}
    # else:
    #     raise Exception(f"API call failed: {response.text}")

tools_pattern = re.compile(r"Action: .*")
def custom_tools_condition(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    content = last_message.content

    print("Checking tools condition:", content)
    if tools_pattern.match(content):
        print("Moving to retrieve...")
        return "tools"
    print("Moving to END...")
    return END

workflow = StateGraph(AgentState)

# Define the workflow using StateGraph
workflow.add_node("agent", agent)
retrieve_node = ToolNode(tools)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)

# Define nodes
workflow.add_edge(START, "agent")

# If the agent calls a tool, proceed to retrieve; otherwise, go to END

workflow.add_conditional_edges(
    "agent",
    custom_tools_condition,
    {
        "tools": "retrieve",
        END: END
    }
)

# After retrieve, determine whether to generate or rewrite
workflow.add_conditional_edges("retrieve", simple_grade_documents)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# Compile the workflow to make it executable
app = workflow.compile()

def process_question(user_question, app, config):
    """Process user question through the workflow"""
    events = []
    for event in app.stream({"messages":[("user", user_question)]}, config):
        events.append(event)
    return events