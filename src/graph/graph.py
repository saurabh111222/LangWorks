from src.graph.state import GraphState
from langgraph.graph import StateGraph, START, END
from src.graph.nodes import (
    chat_node,
    extract_and_store_memory,
    memory_context,
    summarise,
)
from src.graph.edges import should_summarise

def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node('chat', chat_node)
    graph.add_node('extract_memories', extract_and_store_memory)
    graph.add_node('memory_context', memory_context)
    graph.add_node('summarise', summarise)

    graph.add_edge(START, 'extract_memories')
    graph.add_edge('extract_memories', 'memory_context')
    graph.add_edge('memory_context', 'chat')
    graph.add_conditional_edges('chat', should_summarise, {True: 'summarise', False: END})
    graph.add_edge('summarise', END)

    return graph