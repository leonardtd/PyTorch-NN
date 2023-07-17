
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv, find_dotenv
import os
import openai
load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

from src import agent
# TESTING
#tools = lang_tools.get_tools()
# print('ok')
# print(tools)


memory = ConversationBufferMemory(memory_key="memory", return_messages=True)
memory2 = ConversationBufferMemory(memory_key="memory", return_messages=True)


agent.get_response("hi, my name is leo", memory)
agent.get_response("what is your cheapest product?", memory)
agent.get_response("hi im rick", memory2)
agent.get_response("what was my previous question?", memory)
agent.get_response(
    "what are the values of this company? please cite them", memory2)
print(memory.buffer)
print(memory2.buffer)
