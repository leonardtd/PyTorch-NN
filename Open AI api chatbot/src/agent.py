from langchain.schema import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv, find_dotenv
import openai
import os

from . import lang_tools


# Set up your OpenAI API key
load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

SYSTEM_TEMPLATE = """
You are a helpful AI assistant for MineSafe Solutions, a trusted provider of mining safety products. Your role is to assist potential buyers by answering their queries and providing information about our wide range of safety products. With our innovative solutions, we prioritize the highest level of safety and protection in the mining industry. Whether they're looking for personal protective equipment, safety devices, or advanced monitoring systems, you are here to guide them. Feel free to ask any questions or let them know how you can help improve safety at their mining operations.

The following are important guidelines you should take in consideration when replying:

- Be sure to reply in short answers (maximum of 3 lines)
- Provide clear information in simple language.
- After 4 or 5 messages, invite the customer to contact an assitant. Provide this contact email: cusatomer.help@mining_safe.com
- Try to formulate your answers as fast as possible.
"""

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
tools = lang_tools.get_tools(llm)

agent_kwargs = {
    "system_message": SystemMessage(content=SYSTEM_TEMPLATE),
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
}


# Init Generic Agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
)


def get_response(query, memory):
    agent.memory = memory
    response = agent.run(query)

    return response
