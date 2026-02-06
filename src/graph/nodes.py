from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from pydantic import Field, BaseModel
from uuid import uuid4
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
# from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings
from src.graph.state import GraphState
from src.graph.utils.chains import get_response_chain
from src.core.prompts import MEMORY_ANALYSIS_PROMPT
import os

llm = ChatOpenAI(model='gpt-4.1-mini')
SUMMARY_THRESHOLD = 10
SUMMARISE_MESSAGES = 5
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")


qdrant_client = QdrantClient(
    url=QDRANT_URL, 
    api_key=QDRANT_API_KEY,
)

embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=1536
        )


def memory_context(state: GraphState, config: RunnableConfig):
    user_id = config['configurable']['user_id']
    collection_name = f"collection_{user_id}"
    recent_context = " ".join([f"- {m.content}" for m in state["messages"][-3:]])
    
    ######## memory module
    top_revalent_memories =  qdrant_client.query_points(
        collection_name=collection_name,
        query=embeddings.embed_query(recent_context),
        limit=3
    )
    top_revalent_memories_blob = '\n'.join([f'- {mem.payload.get('text', '')}' for mem in top_revalent_memories.points[:2]]) #high score --> low score
    print(f"top_revalent_memories_blob: {top_revalent_memories_blob}")

    return {'memory_context': top_revalent_memories_blob}
    ########

async def chat_node(state: GraphState, config: RunnableConfig):
    messages = state['messages']
    summary = state.get("summary", None)
    memory_context = state.get('memory_context', '')

    chain = get_response_chain(summary, memory_context)
    response = await chain.ainvoke(
        {
            "messages": messages,
            # "memory_context": memory_context
        },
        config=config
    )
    return {'messages': [response]}

async def summarise(state: GraphState):
    """SHORT TERM MEMORY ---> IF messages >= 10 --> Delete starting 5 messages and create a summery of it and plug it with 5 recent messages"""
    existing_summary = state.get('summary')

    if not existing_summary:
        prompt = "Summarise the given conversation.\n"
    else:
        prompt = f"""Extend existing summary using the given conversation\n
                existing_summary: {existing_summary} \n
                """
    messages_to_summarise = state['messages'][:-SUMMARISE_MESSAGES]
    summarise_prompt = messages_to_summarise + [HumanMessage(content=prompt)]
    new_summary = await llm.ainvoke(summarise_prompt)
    print(f"new_summary: {new_summary.content}")

    return {'summary': new_summary.content, 'messages': [RemoveMessage(id=msg.id) for msg in messages_to_summarise]}

class memories(BaseModel):
    is_important: bool = Field(..., description='whether its an important memory or not')
    memories: list[str] = Field(..., description='List of all possible long term atmoic memories.')

structured_llm = llm.with_structured_output(memories)

# Memory spesific node - inside functions can be async
async def extract_and_store_memory(state: GraphState, config: RunnableConfig):
    # LLM extract memory from the user query
    
    user_query = state['messages'][-1].content
    memory_analysis_prompt_formatted = MEMORY_ANALYSIS_PROMPT.format(message=user_query)
    model_analysis = await structured_llm.ainvoke(memory_analysis_prompt_formatted)
    analysed_memories = []
    user_id = config['configurable']['user_id']
    collection_name = f"collection_{user_id}"

    if model_analysis.is_important and model_analysis.memories:
        for memory in model_analysis.memories:
            analysed_memories.append(memory)
    
    print(f"Analysed memories: {analysed_memories}")

    ######## memory module
    # Check if user specific collection already exists - create if not exists - collection_<user_id>
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    memories_to_store = []
    for mem in analysed_memories:
        related_memory = qdrant_client.query_points(
            collection_name=collection_name,
            query=embeddings.embed_query(mem),
            limit=1
        )
        print(f"related_memory: {related_memory}")
        # isinstance(related_memory.points[0], ScoredPoint)
        if (related_memory.points and related_memory.points[0].score < 0.9) or len(related_memory.points)==0:
            memories_to_store.append(mem)  

    if not memories_to_store:
        return {}

    points = [PointStruct(
    id=str(uuid4()),
    vector=embeddings.embed_query(memory),
    payload={
        "text": memory,
        # **metadata,
    }) for memory in memories_to_store]

    qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )
    print(f"memories_to_store: {memories_to_store}")
    ########
    
    return {}