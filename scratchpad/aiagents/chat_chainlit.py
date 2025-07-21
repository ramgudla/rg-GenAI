import chainlit as cl
import os, getpass
env_path = r'<env_file_path>'
from dotenv import load_dotenv
import json
#from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
load_dotenv(env_path)
from datetime import datetime
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
# Import things that are needed generically for tools
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_community.tools import TavilySearchResults

travily_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
)

class City(BaseModel):
    city: str = Field(description="City")

def get_current_weather(city: str) -> int:
    temparation = {'delhi':30,
                   'mumbai':20,
                   'chennai':40}
    return temparation[city.lower()]


weather = StructuredTool.from_function(
    func=get_current_weather,
    name="Get_Weather",
    description="Get the current temperature from a city, in Fahrenheit",
    args_schema=City,
    return_direct=False,
)

class DifferenceInput(BaseModel):
    minuend: int = Field(
        description="The number from which another number is to be subtracted"
    )
    subtrahend: int = Field(description="The number to be subtracted")


def get_difference(minuend: int, subtrahend: int) -> int:
    return minuend - subtrahend

difference = StructuredTool.from_function(
    func=get_difference,
    name="Difference",
    description="Get the difference between two numbers",
    args_schema=DifferenceInput,
    return_direct=False,
)

# llm = AzureChatOpenAI(temperature=0,
#                         api_key=os.getenv('AZURE_OPENAI_API_KEY'),
#                         azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
#                         openai_api_version=os.getenv('AZURE_OPENAI_VERSION'),
#                         azure_deployment=os.getenv('AZURE_GPT4O_MODEL'),
#                         streaming=True
#                         )

from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
llm = ChatOCIGenAI(
            model_id="cohere.command-r-08-2024",
            service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
            compartment_id="ocid1.compartment.oc1..xxxx",
            auth_type="API_KEY",
            auth_profile="xxxx",
            model_kwargs={"temperature": 0, "top_p": 0.75, "max_tokens": 512}
        )

tools_weather = [weather, difference, travily_tool]
llm_with_tools = llm.bind_tools(tools_weather)

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# Node
async def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


def build_graph():
    # start graph

    # Graph
    builder = StateGraph(MessagesState)

    # Define nodes: these do the work
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools_weather))

    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    # initialize state
    #state = MessagesState(messages=[])

    react_graph = builder.compile()

    # display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))
    return react_graph

@cl.on_chat_start
async def on_chat_start():
    react_graph =  build_graph()
    # save graph and state to the user session
    cl.user_session.set("graph", react_graph)
    #cl.user_session.set("state", state)


@cl.on_message
async def on_message(message: cl.Message):
    # Retrieve the graph and state from the user session
    graph: Runnable = cl.user_session.get("graph")
    state = cl.user_session.get("state")

    # Append the new message to the state
    #state["messages"] += [HumanMessage(content=message.content)]
    state = MessagesState(messages=[HumanMessage(content=message.content)])

    # Stream the response to the UI
    ui_message = cl.Message(content="")
    await ui_message.send()
    current_step = {}
    async for event in graph.astream_events(state, version="v1"):
        if event['event'] == 'on_tool_start':
            current_step[event['run_id']] = cl.Step(name=event['name'], type="tool")
            current_step[event['run_id']].show_input = True
            content_json = json.dumps(event['data']['input'])
            current_step[event['run_id']].input = content_json
            await current_step[event['run_id']].send()

        if event['event'] == 'on_tool_end':
            content_json = event['data']['output'].content
            current_step[event['run_id']].output = content_json
            await current_step[event['run_id']].update()

        if event["event"] == "on_chat_model_stream" and event["name"] == "ChatOCIGenAI":
            content = event["data"]["chunk"].content or ""
            await ui_message.stream_token(token=content)
    await ui_message.update()

if __name__ == "__main__":
    import asyncio
    react_graph =  build_graph()
    messages = [HumanMessage(content="Where is it warmest: Chennai, Delhi and Mumbai? And by how much is it warmer than the other cities?")]
    messages = asyncio.run(react_graph.ainvoke({"messages": messages}))
    print(messages)
    