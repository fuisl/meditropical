from langgraph.graph import END, START, StateGraph

from diagnosis_agent.nodes import (
    get_user_input,
    rag_retrieve,
    model_answer_and_reasoning,
    evaluate_diagnosis,
)
from diagnosis_agent.state import GraphState


def build_workflow():
    """
    Build and compile the LangGraph workflow. The nodes are intentionally minimal
    so you can fill them with actual LLM / retriever logic.
    """
    graph = StateGraph(GraphState)

    # register nodes (node name, callable)
    graph.add_node("get_user_input", get_user_input)
    graph.add_node("rag_retrieve", rag_retrieve)
    graph.add_node("model_diagnosis", model_answer_and_reasoning)
    graph.add_node("evaluate_diagnosis", evaluate_diagnosis)

    # wire edges: START -> get_user_input -> RAG -> Diagnose -> Evaluate -> END
    graph.add_edge(START, "get_user_input")
    graph.add_edge("get_user_input", "rag_retrieve")
    graph.add_edge("rag_retrieve", "model_diagnosis")
    graph.add_edge("model_diagnosis", "evaluate_diagnosis")
    graph.add_edge("evaluate_diagnosis", END)

    compiled = graph.compile()
    return compiled


graph = build_workflow()
