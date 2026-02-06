from langgraph.graph import MessagesState

class GraphState(MessagesState):
    summary: str
    memory_context: str