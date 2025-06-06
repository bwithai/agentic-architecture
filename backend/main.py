"""
Main entry point for the MongoDB AI agent application.
This script implements a ChatBot that:
1. Connects to MongoDB
2. Registers MongoDB tools
3. Uses OpenAI API with LangChain to process queries and execute MongoDB operations
"""

import os
import asyncio
import sys
from typing import Dict, List, Any
from dotenv import load_dotenv

from app.agents.tools.registry import ToolRegistry
from app.mongodb.mongodb_setup import setup_mongodb
# from core.chatbot.mongo_chatbot import MongoDBChatBot
from app.agents.specialized.mongodb_agent import MongoDBChatBot

# Create a class to redirect specific debug prints to a different stream
class DebugRedirector:
    def __init__(self, original_stdout, enable_debug=False):
        self.original_stdout = original_stdout
        self.debug_log_file = None
        self.enable_debug = enable_debug
        
    def write(self, message):
        # Only filter debug messages if debug is enabled
        if self.enable_debug:
            # Check if this is a debug message we want to filter
            debug_prefixes = [
                "Getting final response",
                "Response confidence score",
                "AI wants to call",
                "Calling tool",
                "With arguments",
                "Tool result",
                "Thinking process",
                "Classified intent",
                "FALLBACK TRIGGERED",
                "Direct execution failed"
            ]
            
            # If it's a debug message we want to filter, write to debug log
            if any(message.strip().startswith(prefix) for prefix in debug_prefixes):
                # Write to debug log file
                if self.debug_log_file is None:
                    self.debug_log_file = open("debug.log", "a", encoding="utf-8")
                self.debug_log_file.write(message)
                self.debug_log_file.flush()
            else:
                # Otherwise, write to original stdout
                self.original_stdout.write(message)
        else:
            # If debug is disabled, write everything to original stdout
            self.original_stdout.write(message)
    
    def flush(self):
        self.original_stdout.flush()
        if self.debug_log_file:
            self.debug_log_file.flush()
            
    def close(self):
        if self.debug_log_file:
            self.debug_log_file.close()
    
    def enable_debug_logging(self):
        """Enable debug logging to file."""
        self.enable_debug = True
    
    def disable_debug_logging(self):
        """Disable debug logging to file."""
        self.enable_debug = False


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
        from mongodb import client as mongodb_client_module
        mongodb_client_module.db = db_client.db
        
        print(f"Global db reference set: {mongodb_client_module.db is not None}")
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        return
    
    # Initialize tool registry and chatbot
    tool_registry = ToolRegistry(db_client)
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
        query = input("You: ")
        
        if query.lower() in ["exit", "quit", "bye"]:
            print("Shutting down chatbot...")
            break
        
        # Process the query and get response
        response = await chat_bot.process_query(query)
        print(f"\nAI: {response}")
    
    # Clean up
    await db_client.close()
    print("MongoDB connection closed")


def main():
    """Entry point for the application."""
    print("Starting MongoDB AI Assistant...")
    
    # Redirect stdout but keep debug logging disabled by default
    original_stdout = sys.stdout
    debug_redirector = DebugRedirector(original_stdout, enable_debug=False)
    sys.stdout = debug_redirector
    
    try:
        asyncio.run(run_chat_bot())
    except (KeyboardInterrupt, EOFError):
        # User intentionally stopped the script, no need for debug logging
        print("\nApplication stopped by user.")
    except Exception as e:
        # Enable debug logging only when there's an unexpected exception
        debug_redirector.enable_debug_logging()
        print(f"\nUnexpected error occurred: {e}")
        print("Debug logging has been enabled and saved to debug.log")
        
        # Log the exception details to debug file
        import traceback
        if debug_redirector.debug_log_file is None:
            debug_redirector.debug_log_file = open("debug.log", "a", encoding="utf-8")
        debug_redirector.debug_log_file.write(f"\n{'='*50}\n")
        debug_redirector.debug_log_file.write(f"Exception occurred: {e}\n")
        debug_redirector.debug_log_file.write(f"Traceback:\n{traceback.format_exc()}\n")
        debug_redirector.debug_log_file.write(f"{'='*50}\n")
        debug_redirector.debug_log_file.flush()
    finally:
        # Restore original stdout
        if isinstance(sys.stdout, DebugRedirector):
            sys.stdout.close()
        sys.stdout = original_stdout


if __name__ == "__main__":
    main() 