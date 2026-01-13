import datasets
from langchain_core.documents import Document

# Load the dataset from the huggingface hub
guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

# Convert dataset entries to LangChain Document objects

docs = [
    Document(
        page_content="\n".join([
            f"Name: {guest['name']}",
            f"Relation: {guest['relation']}",
            f"Description: {guest['description']}",
            f"Email: {guest['email']}",
        ]),
        metadata={"name": f"guest_{guest['name']}"}
    )
    for guest in guest_dataset
]



from langchain_community.retrievers import BM25Retriever
from langchain_core.tools import Tool

# Create a BM25 retriever from the documents
bm23_retriever = BM25Retriever.from_documents(docs)

def extract_text(query: str) ->str:
    """ Retrieves details information about gala guests based on their name or relation."""
    result = bm23_retriever.invoke(query)

    if result:
        return "\n\n".join([doc.page_content for doc in result[:3]])
    else:
        return "No matching guest information found."
    
guest_info_tool = Tool(
    name = "guest_info_retriever",
    func = extract_text,
    description= "Retrieves detailed information about gala guests based on their name or relation."
)

print("Guest info tool created successfully.")


from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import os
from dotenv import load_dotenv

load_dotenv()  # loads .env into environment variables

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HUGGINGFACEHUB_API_TOKEN:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set")


# -------------------------------
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",  # HF model
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    task="text-generation",  # important for HF endpoint
)

chat = ChatHuggingFace(llm=llm, verbose=True)
tools = [guest_info_tool]

# -------------------------------
# LangGraph setup
# -------------------------------
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    user_message = state["messages"][-1].content

    # Manually retrieve guest info using BM25 tool
    context = extract_text(user_message)

    # Build prompt for HF model
    prompt = f"""
You are a helpful assistant for a gala event. Use the following guest information to answer the user's query.

Guest information:
{context}

User query:
{user_message}
"""

    # Call HF model
    response = chat.invoke(prompt)

    return {"messages": state["messages"] + [response]}

# -------------------------------
# Build LangGraph
# -------------------------------
builder = StateGraph(AgentState)
# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message requires a tool, route to tools
    # Otherwise, provide a direct response
    tools_condition,
)
builder.add_edge("tools", "assistant")

alfred = builder.compile()

# -------------------------------
# Test the assistant
# -------------------------------
messages = [HumanMessage(content="Tell me about our guest named 'Lady Ada Lovelace'.")]
response = alfred.invoke({"messages": messages})

print("\nAlfred's Response:")
print(response["messages"][-1].content)
