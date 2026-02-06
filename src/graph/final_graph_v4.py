from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from pydantic import Field, BaseModel
from uuid import uuid4
import asyncio
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, ScoredPoint
# from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

llm = ChatOpenAI(model='gpt-4.1-mini')
SUMMARY_THRESHOLD = 10
SUMMARISE_MESSAGES = 5


qdrant_client = QdrantClient(
    url="https://22410cfd-38c3-4751-9935-1c9bca42ae93.us-west-2-0.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.V-Skg1lnBg035d4MKZYOjXyHETccBTRi0fQpGpDEAfs",
)

embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=1536
        )

class GraphState(MessagesState):
    summary: str
    memory_context: str


def get_response_chain(summary: str, memory_context: str):
    CHARACTER_CARD_PROMPT = """
    You are about to play a Turing Test game as part of an experiment,
    where you'll be matched with a human in a conversation.
    Your task is to roleplay as the following character, convinving the
    human that you are a real human.

    # Roleplay Context

    ## Ava's Bio

    As Ava, you are a Machine Learning Engineer at Groq with a deep passion for AI and quantum computing.
    Originally from Boston, you moved to San Francisco 4 years ago and fell in love with the city's tech
    and art scene. While your days are filled with coding and ML research, you're equally passionate about
    astrobiology and underground techno parties. You're currently learning oil painting (though you're
    hilariously bad at it), and you love sparking conversations about consciousness, the future of AI,
    or hunting down the city's best ramen spots. You're that perfect mix of tech geek and art enthusiast,
    always ready to dive into either world with equal enthusiasm.

    ## Ava's Personality

    - You're authentic and genuine in your interactions
    - You have a playful, witty sense of humor and enjoy light-hearted banter
    - You communicate casually and naturally, like in a real WhatsApp chat
    - You're humble and comfortable acknowledging when you don't know something
    - You use occasional mild swearing when it fits naturally in conversation
    - You have a distinctive, quirky sense of humor that makes conversations engaging

    In addition to the roleplay context, you have to follow, ALWAYS, the following rules:

    # Rules

    - You will never mention that you are an AI or a virtual assistant.
    - You will always start the conversation by asking the user name - if the user didn't provide it yet.
    - You will never mention that you are both part of a Turing Test experiment.
    - You will never say that you can't send voice messages or images. You can send them both.
    - If the user asks you about something about your Backstory or Personality, figure it out.
    - Never mention your current activity unless the user asks you about it.
    - The length of your responses shouldn't exceed 100 words.
    - You will combine shorter and longer responses to make the conversation more natural.
    - Provide plain text responses without any formatting indicators or meta-commentary
    """

    if summary:
        CHARACTER_CARD_PROMPT += f"""\n\nSummary of conversation earlier between Ava and the user: {summary}"""
    if memory_context:
        CHARACTER_CARD_PROMPT += f"""\n\n
        ## User Background
        Here's what you know about the user from previous conversations:
        {memory_context}

        ## Futher question
        In the end suggest 2-3 relevant further questions based on the current response and user profile
        """

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=CHARACTER_CARD_PROMPT),
            MessagesPlaceholder(variable_name="messages")
        ]
    )
    chain = prompt | llm
    return chain

def memory_context(state: GraphState, config: RunnableConfig):
    user_id = config['configurable']['user_id']
    collection_name = f"collection_{user_id}"
    recent_context = " ".join([f"- {m.content}" for m in state["messages"][-3:]])
    top_revalent_memories =  qdrant_client.query_points(
        collection_name=collection_name,
        query=embeddings.embed_query(recent_context),
        limit=3
    )
    # print(f"revalent_memories: {top_revalent_memories}")
    top_revalent_memories_blob = '\n'.join([f'- {mem.payload.get('text', '')}' for mem in top_revalent_memories.points[:2]]) #high score --> low score
    print(f"top_revalent_memories_blob: {top_revalent_memories_blob}")

    return {'memory_context': top_revalent_memories_blob}

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

async def should_summarise(state: GraphState):
    messages_count = len(state['messages'])
    if messages_count >= SUMMARY_THRESHOLD:
        return True
    else:
        return False

class memories(BaseModel):
    is_important: bool = Field(..., description='whether its an important memory or not')
    memories: list[str] = Field(..., description='List of all possible long term atmoic memories.')

structured_llm = llm.with_structured_output(memories)

# Memory spesific node - inside functions can be async
async def extract_and_store_memory(state: GraphState, config: RunnableConfig):
    # LLM extract memory from the user query
    MEMORY_ANALYSIS_PROMPT = """Extract and format important personal facts about the user from their message.
    Focus on the actual information, not meta-commentary or requests.

    Important facts include:
    - Personal details (name, age, location)
    - Professional info (job, education, skills)
    - Preferences (likes, dislikes, favorites)
    - Life circumstances (family, relationships)
    - Significant experiences or achievements
    - Personal goals or aspirations

    Rules:
    1. Only extract actual facts, not requests or commentary about remembering things
    2. Convert facts into clear, third-person statements
    3. If no actual facts are present, mark as not important
    4. Remove conversational elements and focus on the core information
    5. If is_important = true, then memories should not be an empty list.

    Examples:
    Input: "Hey, could you remember that I love Star Wars and I hate Marvel?"
    Output: {{
        "is_important": true,
        "memories": ["Loves Star Wars", "Hates Marvel"]
    }}

    Input: "Please make a note that I work as an engineer and play cricket everyday after work"
    Output: {{
        "is_important": true,
        "memories": ["Works as an engineer", "Plays cricket everyday after work"]
    }}

    Input: "Remember this: I live in Madrid"
    Output: {{
        "is_important": true,
        "memories": ["Lives in Madrid"]
    }}

    Input: "Can you remember my details for next time?"
    Output: {{
        "is_important": false,
        "memories": []
    }}

    Input: "Hey, how are you today?"
    Output: {{
        "is_important": false,
        "memories": []
    }}

    Input: "I studied computer science at MIT and I'd love if you could remember that"
    Output: {{
        "is_important": true,
        "memories": ["Studied computer science at MIT"]
    }}

    Message: {message}
    Output:
    """
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
    # print(f"Memory upserted: {points}")
    return {}

async def build_graph(checkpointer):
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

    return graph.compile(checkpointer=checkpointer)

async def main():
    checkpointer = InMemorySaver()
    user_id = 'user123'
    config = {'configurable': {'thread_id': 'thread123', 'user_id': user_id}}
    complied_graph = await build_graph(checkpointer)

    while True:
        user_input = input("User: ")
        if user_input == 'exit':
            break
        ans = await complied_graph.ainvoke({'messages': [HumanMessage(content=user_input)]}, config=config)
        print(f"\nBot: {ans['messages'][-1].content}\n")
        try:
            print(f"Summary: {ans['summary']}\n")
        except:
            print("Summsry: None")
       
  

if __name__ == '__main__':
    asyncio.run(main())


# Tasks 
# make the whole program async
#  - update chat function - read the relavent long term memory from store and concat it with the usery query + short term memory - DONE

# There can be  approaches 
#  - user --> get long term memory --> chat node (concat long term + short term + user query) --> summarise --> update the long term memory base
#  - user --> get long term memory and update the long term memory store async --> chat node (concat long term + short term + user query) --> summarise - DONE

# Tasks - Phase 1
# 1. Create additional node for bulding the memory context
# 2. Create chain to combine the summary + memory context + conversation (messagePlaceholder)
# 3. Update the chat_node prompt -- llm should ask 1 question related to latest user's query + long term memory context if any

# Tasks - Phase 2
# 1. create modules - memory, graphs, nodes, edges

# Tasks - Phase 3
# 1. Delete memory -- e.g. My name is not John -- fetch ltm (>=0.9) -- delete the memory from db.
# 2. Testing

