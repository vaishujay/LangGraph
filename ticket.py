import streamlit as st
from typing_extensions import TypedDict
from typing import List, Optional, Literal

from langgraph.graph import StateGraph, START, END

# ---------------------------
# State Definition
# ---------------------------
class TicketState(TypedDict):
    ticket_id: str
    issue: str
    priority: Optional[str]
    assigned_to: Optional[str]
    history: List[str]

# ---------------------------
# LangGraph Nodes
# ---------------------------
def receive_ticket(state: TicketState):
    return {
        "ticket_id": state["ticket_id"],
        "issue": state["issue"],
        "priority": None,
        "assigned_to": None,
        "history": state["history"] + ["Ticket received"]
    }

def senior_support(state: TicketState):
    return {
        "ticket_id": state["ticket_id"],
        "issue": state["issue"],
        "priority": "High",
        "assigned_to": "Senior Support Team",
        "history": state["history"] + ["Assigned to Senior Support"]
    }

def junior_support(state: TicketState):
    return {
        "ticket_id": state["ticket_id"],
        "issue": state["issue"],
        "priority": "Normal",
        "assigned_to": "Junior Support Team",
        "history": state["history"] + ["Assigned to Junior Support"]
    }

# ---------------------------
# Decision Logic
# ---------------------------
def classify_ticket(state: TicketState) -> Literal["senior", "junior"]:
    urgent_keywords = ["payment", "crash", "down", "security", "failed"]

    if any(word in state["issue"].lower() for word in urgent_keywords):
        return "senior"
    return "junior"

# ---------------------------
# Build LangGraph
# ---------------------------
def build_graph():
    graph = StateGraph(TicketState)

    graph.add_node("receive", receive_ticket)
    graph.add_node("senior", senior_support)
    graph.add_node("junior", junior_support)

    graph.add_edge(START, "receive")
    graph.add_conditional_edges("receive", classify_ticket)
    graph.add_edge("senior", END)
    graph.add_edge("junior", END)

    return graph.compile()

graph_builder = build_graph()

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Customer Support Triage", layout="centered")

st.title("Customer Support Ticket Triage System")
st.write("Automatically routes tickets using **LangGraph**")

# Initialize session state
if "result" not in st.session_state:
    st.session_state.result = None

ticket_id = st.text_input("Ticket ID", placeholder="TCK-101")
issue = st.text_area("Describe the Issue", placeholder="Payment failed and app crashed")

if st.button("Submit Ticket"):
    if not ticket_id or not issue:
        st.warning("Please fill all fields")
    else:
        st.session_state.result = graph_builder.invoke({
            "ticket_id": ticket_id,
            "issue": issue,
            "priority": None,
            "assigned_to": None,
            "history": []
        })
        st.success("Ticket processed successfully")

# ---------------------------
# Display Result
# ---------------------------
if st.session_state.result:
    result = st.session_state.result

    st.subheader("Ticket Details")
    st.write(f"**Ticket ID:** {result['ticket_id']}")
    st.write(f"**Priority:** {result['priority']}")
    st.write(f"**Assigned To:** {result['assigned_to']}")

    st.subheader("Decision History")
    for step in result["history"]:
        st.write(f"- {step}")

# ---------------------------
# Graph Visualization
# ---------------------------
st.subheader("Workflow Graph")

try:
    st.image(graph_builder.get_graph().draw_mermaid_png())
except Exception:
    st.info("Graph visualization requires IPython support.")
