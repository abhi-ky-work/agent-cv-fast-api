from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from app.agent.data_loader import retriever
from app.agent.crew import execute_crew_query

class AgentState(TypedDict):
    query: str
    context: str
    response: str

def retrieve_node(state: AgentState):
    query = state["query"]
    # Retrieve documents from the vector store
    # Since retriever might be None if no docs exist, handle gracefully
    if retriever:
        docs = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)
    else:
        context = "No CV or context documents found."
    return {"context": context}

def generate_node(state: AgentState):
    query = state["query"]
    context = state["context"]
    # Execute CrewAI for generation
    response = execute_crew_query(query, context)
    return {"response": response}

workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

agent_graph = workflow.compile()
