
# -------------------------------------
### 1st Search Tool ###
# -------------------------------------
from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()
result = search_tool.invoke(" Who's the current President of France? ")
print(result)

# -------------------------------------
### 2nd Custom Tool for Weather ###
# -------------------------------------

from langchain_core.tools import Tool
import random

def get_weather_info(location: str) -> str:
    """ Fetches dummy weather information for a given location. """
    # Dummy weather data
    weather_condidtions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Clear", "temp_c": 25},
        {"condition": "Windy", "temp_c": 20}
    ]
   # Randomly select a weather condition 
    data = random.choice(weather_condidtions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']} °C"

# Initialize the tool
weather_info_tool = Tool(name = 'get_weather_info',
                         func = get_weather_info,
                         description= "Fetch the dummy weather information for a given location."
                         )

# ------------------------------------------------------
### 3rd Hub Stats Tool for Influential AI Builders ###
# ------------------------------------------------------

from huggingface_hub import list_models
from langchain_core.tools import Tool

def get_hub_stats(author: str) -> str:
    """Fetches the most downloaded model from a specific author on the Hugging Face Hub."""
    try:
        # Fetch models (no sort, no direction)
        models = list(list_models(author=author, limit=20))

        if not models:
            return f"No models found for author {author}."

        # Manually find most downloaded model
        top_model = max(models, key=lambda m: m.downloads or 0)

        return (
            f"The most downloaded model by {author} is "
            f"{top_model.modelId} with {top_model.downloads:,} downloads."
        )

    except Exception as e:
        return f"Error fetching models for {author}: {str(e)}"

hub_stats_tool = Tool(
    name="get_hub_stats",
    func=get_hub_stats,
    description="Fetches the most downloaded model from a specific author on the Hugging Face Hub."
)

print(hub_stats_tool.invoke("facebook")) # Example: Get the most downloaded model by Facebook


# Integrating Tools with Alfred
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HUGGINGFACEHUB_API_TOKEN:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found")

# ----------------------------
# Hugging Face LLM (NO TOOLS)
# ----------------------------
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

chat = ChatHuggingFace(llm=llm)

# ----------------------------
# LangGraph State
# ----------------------------
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# ----------------------------
# Assistant node
# ----------------------------
def assistant(state: AgentState):
    response = chat.invoke(state["messages"])
    return {"messages": [response]}

# ----------------------------
# Build Graph
# ----------------------------
builder = StateGraph(AgentState)
builder.add_node("assistant", assistant)
builder.add_edge(START, "assistant")
builder.add_edge("assistant", END)

alfred = builder.compile()

# ----------------------------
# Run
# ----------------------------
messages = [
    HumanMessage(content="Who is Facebook and what is their most popular model?")
]

result = alfred.invoke({"messages": messages})

print("\nAlfred's Response:\n")
print(result["messages"][-1].content)
