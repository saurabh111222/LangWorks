from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from src.core.prompts import CHARACTER_CARD_PROMPT

llm = ChatOpenAI(model='gpt-4.1-mini')


def get_response_chain(summary: str, memory_context: str):
    system_message = CHARACTER_CARD_PROMPT

    if summary:
        system_message += f"""\n\nSummary of conversation earlier between Ava and the user: {summary}"""
    if memory_context:
        system_message += f"""\n\n
        ## User Background
        Here's what you know about the user from previous conversations:
        {memory_context}

        ## Futher question
        In the end suggest 2-3 relevant further questions based on the current response and user profile
        """

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_message),
            MessagesPlaceholder(variable_name="messages")
        ]
    )
    chain = prompt | llm
    return chain