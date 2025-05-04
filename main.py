"""
Main entry point for the AI agent application.
This script initializes and runs the agent that:
1. Understands user queries
2. Classifies intent (general conversation vs. database query)
3. Retrieves information from MongoDB for business inquiries
4. Returns human-friendly responses
"""

import os
import asyncio
from dotenv import load_dotenv
from core.graph.agent_graph import create_agent_graph
from config.config import config
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage


async def run_agent():
    """Initialize and run the agent system asynchronously."""
    # Load environment variables
    # print(config.mongodb.database)
    # input("does db load")
    load_dotenv()
    
    # Create the agent graph
    agent_graph = create_agent_graph()

    try:
        # Save the graph visualization to a file
        graph_image = agent_graph.get_graph().draw_mermaid_png()
        with open("agent_graph.png", "wb") as f:
            f.write(graph_image)
        print("Graph visualization saved to agent_graph.png")
    except Exception as e:
        # This requires some extra dependencies and is optional
        print(f"Could not generate graph visualization: {str(e)}")
    
    # Run the agent with each example query
    while True:
        print("\n" + "="*80)
        query = input("USER: ")
        
        # Initialize message state with the user query using LangChain message objects
        initial_state = {
            "messages": [
                HumanMessage(content=query)
            ]
        }
        
        # Run the agent with the query using the async API
        result = await agent_graph.ainvoke(initial_state)
        
        # Display the result
        print("\nAgent Processing:")
        for message in result["messages"]:
            if isinstance(message, HumanMessage):
                continue  # Skip displaying the human message again
            elif isinstance(message, AIMessage):
                print(f"ASSISTANT: {message.content}")
            elif isinstance(message, ToolMessage):
                print(f"[{message.tool_call_id}]: {message.content}")


def main():
    """Initialize and run the agent system."""
    print("Starting AI Agent System...")
    # Run the async function in an event loop
    asyncio.run(run_agent())


if __name__ == "__main__":
    main() 