import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from tool import DuckDuckGoSearchRun, weather_info_tool, hub_stats_tool
from guest_stories import guest_info_tool

import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    st.error("Set HUGGINGFACEHUB_API_TOKEN in .env")
    st.stop()

# Sidebar settings
with st.sidebar:
    st.title("Settings")
    temp = st.slider("Temperature", 0.0, 1.0, 0.3)
    max_tokens = st.slider("Max Tokens", 128, 1024, 512)
    st.markdown("---")

# Initialize model
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    huggingfacehub_api_token=HF_TOKEN,
    temperature=temp,
    max_new_tokens=max_tokens,
)

chat = ChatHuggingFace(llm=llm)

# Tools
search_tool = DuckDuckGoSearchRun()
tools = [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool]

# LangGraph state
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def assistant(state: AgentState):
    msg = state["messages"][-1].content
    
    # e.g., get guest info
    guest_context = guest_info_tool.func(msg)

    prompt = f"""
You are Alfred, a helpful assistant.
Guest context:
{guest_context}

Question:
{msg}
"""
    resp = chat.invoke([HumanMessage(content=prompt)])
    return {"messages": state["messages"] + [resp]}

builder = StateGraph(AgentState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
alfred = builder.compile()

st.title("Alfred - Interactive RAG Chat")

# interactive chat
if "history" not in st.session_state:
    st.session_state.history = []

# Chat input
if query := st.chat_input("Ask Alfred..."):
    # add user
    st.session_state.history.append(HumanMessage(content=query))

    with st.spinner("Alfred is thinking..."):
        result = alfred.invoke({"messages": st.session_state.history})

    st.session_state.history = result["messages"]

# render conversation
for m in st.session_state.history:
    if isinstance(m, HumanMessage):
        st.chat_message("user").write(m.content)
    else:
        st.chat_message("assistant").write(m.content)
