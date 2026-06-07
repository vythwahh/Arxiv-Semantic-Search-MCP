import asyncio
import logging
import anthropic
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from embedder import ArxivEmbedder
from hybrid_search import HybridSearch
 
from rag_engine import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ArxivMCPServer")

embedder = ArxivEmbedder()
hybrid_search = HybridSearch(alpha=0.7)
llm_client = anthropic.Anthropic()

pipeline = RAGPipeline(
    embedder=embedder,
    hybrid_search=hybrid_search,
    llm_client=llm_client,
    top_k=5,
    max_context_papers=3
)

server = Server("arxiv-semantic-search")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """
    Registers available tools for AI Agents connecting via MCP protocol.
    """
    return [
        types.Tool(
            name="index_arxiv",
            description="Fetch and index arXiv papers on a given topic for semantic search.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Topic or keywords to search arXiv for (e.g., 'transformer attention mechanism')"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of papers to fetch (default: 100)",
                        "default": 100
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="search_papers",
            description="Search indexed arXiv papers using hybrid semantic + keyword search.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="rag_search",
            description="Search arXiv papers and generate a synthesized answer using RAG pipeline.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Research question to answer from indexed papers"
                    }
                },
                "required": ["query"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """
    Handles tool calls from AI Agents via MCP protocol.
    Routes each tool name to its corresponding pipeline method.
    """
    try:
        if name == "index_arxiv":
            query = arguments["query"]
            max_results = arguments.get("max_results", 100)

            # Run the indexing in a separate thread to avoid blocking the event loop
            n_indexed = await asyncio.to_thread(
                pipeline.index, query=query, max_results=max_results
            )

            return [types.TextContent(
                type="text",
                text=f"Successfully indexed {n_indexed} arXiv papers for topic: '{query}'"
            )]

        elif name == "search_papers":
            query = arguments["query"]
            top_k = arguments.get("top_k", 5)

            # Avoid blocking the event loop by running the retrieval in a separate thread
            results = await asyncio.to_thread(pipeline.retrieve, query=query)

            if not results:
                return [types.TextContent(
                    type="text",
                    text="No results found. Please run index_arxiv first."
                )]

            # Get top-k results and format output
            sliced_results = results[:top_k]

            output = f"Top {len(sliced_results)} results for: '{query}'\n\n"
            for rank, r in enumerate(sliced_results, start=1):
                output += (
                    f"[Rank {rank}] Score: {r.score:.4f}\n"
                    f"Title: {r.paper.title}\n"
                    f"Authors: {', '.join(r.paper.authors[:3])}\n"
                    f"URL: {r.paper.url}\n"
                    f"Published: {r.paper.published}\n\n"
                )

            return [types.TextContent(type="text", text=output)]

        elif name == "rag_search":
            query = arguments["query"]
            
            # Implement pipeline RAG
            answer, results = await asyncio.to_thread(
                pipeline.search_and_generate, query=query
            )

            sources = "\n".join([
                f"- [Source {rank}] {r.paper.title} ({r.paper.url})"
                for rank, r in enumerate(results, start=1)
            ])

            output = (
                f"Answer:\n{answer}\n\n"
                f"Sources:\n{sources}"
            )

            return [types.TextContent(type="text", text=output)]

        else:
            return [types.TextContent(
                type="text",
                text=f"Unknown tool: {name}"
            )]

    except Exception as e:
        logger.error(f"Tool execution failed for '{name}': {e}")
        return [types.TextContent(
            type="text",
            text=f"[Error] Tool '{name}' failed: {str(e)}"
        )]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    logger.info("Starting ArXiv Semantic Search MCP Server...")
    asyncio.run(main())