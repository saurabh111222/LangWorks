import sys
import os
# Add parent directory (LangWorks) to path so 'src' module can be found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

from dotenv import load_dotenv
load_dotenv()

from langgraph.checkpoint.memory import InMemorySaver
from src.graph import graph_builder
from langchain_core.messages import HumanMessage
import asyncio

async def main():
    checkpointer = InMemorySaver()
    user_id = 'user123'
    config = {'configurable': {'thread_id': 'thread123', 'user_id': user_id}}
    # graph_builder = build_graph()
    complied_graph = graph_builder.compile(checkpointer=checkpointer)

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



# print(sys.path)
# '/Users/saurabhkumar/Desktop/AIEng/LangWorks', '/Users/saurabhkumar/Desktop/AIEng/LangWorks/src/interfaces', '/Library/Frameworks/Python.framework/Versions/3.12/lib/python312.zip', '/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12', '/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload', '/Users/saurabhkumar/Library/Python/3.12/lib/python/site-packages', '/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages'