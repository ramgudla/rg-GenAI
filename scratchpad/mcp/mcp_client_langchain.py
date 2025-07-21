#Using LangChain With Model Context Protocol (MCP)

# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
import asyncio

#from langchain_openai import ChatOpenAI
#model = ChatOpenAI(model="gpt-4o")

from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
model = ChatOCIGenAI(
            model_id="cohere.command-r-08-2024",
            service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
            compartment_id="ocid1.compartment.oc1..xxxx",
            auth_type="API_KEY",
            auth_profile="xxxx",
            model_kwargs={"temperature": 0, "top_p": 0.75, "max_tokens": 512}
        )

server_params = StdioServerParameters(
    command="python",
    args=["mcp_server_math.py"],
)

async def run_agent():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            # Create and run the agent
            agent = create_react_agent(model, tools)
            agent_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
            return agent_response

# Run the async function
if __name__ == "__main__":
    result = asyncio.run(run_agent())
    print(result)
