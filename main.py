"""
Main entry point for the MongoDB AI agent application.
This script implements a ChatBot that:
1. Connects to MongoDB
2. Registers MongoDB tools
3. Uses OpenAI API with LangChain to process queries and execute MongoDB operations
"""

import os
import asyncio
from typing import Dict, List, Any
from dotenv import load_dotenv

from agents.tools.registry import ToolRegistry
from mongodb.mongodb_setup import setup_mongodb
from core.chatbot.mongo_chatbot import MongoDBChatBot


async def run_chat_bot():
    """Initialize and run the chatbot."""
    # Load environment variables
    load_dotenv()
    
    # Get required configuration
    mongodb_url = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/test")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        return
    
    # Connect to MongoDB
    print(f"Connecting to MongoDB at {mongodb_url}...")
    try:
        db_client = await setup_mongodb(mongodb_url)
        print("Connected to MongoDB successfully!")
        
        # Set the global db reference in the module
        import sys
        from mongodb import client as mongodb_client_module
        mongodb_client_module.db = db_client.db
        
        print(f"Global db reference set: {mongodb_client_module.db is not None}")
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        return
    
    # Initialize tool registry and chatbot
    tool_registry = ToolRegistry()
    chat_bot = MongoDBChatBot(
        mongodb_client=db_client,
        tool_registry=tool_registry,
        openai_api_key=openai_api_key
    )
    
    # Run the chatbot interaction loop
    print("\nMongoDB AI Assistant Ready!")
    print("Type 'exit' to quit")
    
    while True:
        print("\n" + "="*80)
        query = input("USER: ")
        
        if query.lower() in ["exit", "quit", "bye"]:
            print("Shutting down chatbot...")
            break
        
        # Process the query and get response
        response = await chat_bot.process_query(query)
        print(f"\nASSISTANT: {response}")
    
    # Clean up
    await db_client.close()
    print("MongoDB connection closed")


def main():
    """Entry point for the application."""
    print("Starting MongoDB AI Assistant...")
    asyncio.run(run_chat_bot())


if __name__ == "__main__":
    main() 