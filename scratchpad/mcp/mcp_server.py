from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Local Agent Helper")

@mcp.tool()
def ls(directory: str) -> str:
    "List the contents of a directory."
    import os
    return "\n".join(os.listdir(directory))

@mcp.tool()
def cat(file: str) -> str:
    "Display the file content."
    try:
        with open(file, "r") as f:
            return f.read()
    except:
        return ""

@mcp.tool()
def echo(message: str, file: str) -> str:
    "Write text to a file."
    try:
        with open(file, "w") as f:
            f.write(message)
            return "success"
    except:
        return "failed"

# Entry point to run the server
if __name__ == "__main__":
    mcp.run()
