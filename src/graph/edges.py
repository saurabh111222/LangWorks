from src.graph.state import GraphState

SUMMARY_THRESHOLD = 10

def should_summarise(state: GraphState):
    messages_count = len(state['messages'])
    if messages_count >= SUMMARY_THRESHOLD:
        return True
    else:
        return False