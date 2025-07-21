# Using with LangGraph StateGraph

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

# from langchain_openai import ChatOpenAI
# model = ChatOpenAI(model="gpt-4o")

from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
model = ChatOCIGenAI(
            model_id="cohere.command-r-08-2024",
            service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
            compartment_id="ocid1.compartment.oc1..xxxx",
            auth_type="API_KEY",
            auth_profile="xxxx",
            model_kwargs={"temperature": 0, "top_p": 0.75, "max_tokens": 512}
        )

async def my_async_function():
    async with MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                # Make sure to update to the full absolute path to your mcp_server_math.py file
                "args": ["mcp_server_math.py"],
                "transport": "stdio",
            },
            "weather": {
                # make sure you start your weather server on port 8000; $python mcp_server_weather.py
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            }
        }
    ) as client:
        tools = client.get_tools()
        def call_model(state: MessagesState):
            response = model.bind_tools(tools).invoke(state["messages"])
            return {"messages": response}

        builder = StateGraph(MessagesState)
        builder.add_node(call_model)
        builder.add_node(ToolNode(tools))
        builder.add_edge(START, "call_model")
        builder.add_conditional_edges(
            "call_model",
            tools_condition,
        )
        builder.add_edge("tools", "call_model")
        graph = builder.compile()
        math_response = await graph.ainvoke({"messages": "what's (3 + 5) x 12?"})
        print(math_response)
        weather_response = await graph.ainvoke({"messages": "what is the weather in nyc?"})
        print(weather_response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(my_async_function())
