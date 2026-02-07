import os
import sys

# Ensure 'src' is in the python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from src.graph.graph import build_graph

# Initialize FastAPI
app = FastAPI(title="LangWorks RAG Server")

# --- Global State ---
# We initialize the checkpointer and graph ONCE.
# The checkpointer (InMemorySaver) handles thread isolation automatically.
checkpointer = InMemorySaver()
graph_builder = build_graph()
compiled_graph = graph_builder.compile(checkpointer=checkpointer)

# --- Pydantic Models for Input/Output ---
class ChatRequest(BaseModel):
    user_id: str
    thread_id: str
    message: str

class ChatResponse(BaseModel):
    bot_response: str
    summary: str | None = None

# --- Endpoints ---

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Handles chat interactions.
    - Uses 'user_id' for Qdrant collection separation.
    - Uses 'thread_id' for LangGraph conversation state/history.
    """
    
    # Config tells LangGraph which thread/user this is for
    config = {
        "configurable": {
            "thread_id": request.thread_id,
            "user_id": request.user_id
        }
    }

    try:
        # Await the graph execution (Non-blocking!)
        # This reuses the logic from src/graph/nodes.py
        result = await compiled_graph.ainvoke(
            {"messages": [HumanMessage(content=request.message)]}, 
            config=config
        )
        
        # Extract the last message from the bot
        last_message = result['messages'][-1].content
        summary = result.get('summary')

        return ChatResponse(
            bot_response=last_message,
            summary=summary
        )

    except Exception as e:
        # Log error in production
        print(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run with: python src/interfaces/server.py
    uvicorn.run(app, host="0.0.0.0", port=8000)