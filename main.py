"""
Main entry point for the AI agent application.
This script initializes and runs the agent that:
1. Understands user queries
2. Retrieves information from MongoDB
3. Returns human-friendly responses
"""

import os
import asyncio
from dotenv import load_dotenv
from core.graph.agent_graph import create_agent_graph
from config.config import config


async def run_agent():
    """Initialize and run the agent system asynchronously."""
    # Load environment variables
    load_dotenv()
    
    # Create the agent graph
    agent_graph = create_agent_graph()
    
    # Example query
    user_query = "Show me all documents from the users collection"
    
    # Run the agent with the query using the async API
    result = await agent_graph.ainvoke({"query": user_query})
    
    # Display the result
    print(f"User Query: {user_query}")
    print(f"Agent Response: {result['response']}")


def main():
    """Initialize and run the agent system."""
    # Run the async function in an event loop
    asyncio.run(run_agent())


if __name__ == "__main__":
    main() 