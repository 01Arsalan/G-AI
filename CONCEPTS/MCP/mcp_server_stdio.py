from fastmcp import FastMCP
mcp = FastMCP(name="MCP STDIO")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Adds two integers together.
    args:
        a: The first integer.
        b: The second integer.
    returns:
        The sum of the two integers."""
    return a + b

@mcp.tool(
        name="multiply_tool",
        description="Multiplies two integers together."
)
def multiply(a: int, b: int) -> int:
    """Multiplies two integers together.
    args:
        a: The first integer.
        b: The second integer.
    returns:
        The product of the two integers."""
    return a * b



if __name__ == "__main__":
    # mcp.run()
    # stdio transport so clients can talk via subprocess stdin/stdout 
    mcp.run(transport="stdio")


#find mcp.json in .vscode/ for client config details

# to test server manually, run: npx @modelcontextprotocal 