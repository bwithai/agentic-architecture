"""
Main entry point for the MongoDB AI agent application.
This script implements a ChatBot that:
1. Connects to MongoDB
2. Registers MongoDB tools
3. Uses OpenAI API with LangChain to process queries and execute MongoDB operations
"""

import os
import asyncio
import json
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import Tool
from pydantic.v1 import BaseModel, Field, create_model

from mongodb.client import MongoDBClient
from agents.tools.registry import ToolRegistry
from agents.utils.serialization_utils import serialize_mongodb_doc, mongodb_json_dumps


async def setup_mongodb(db_url: str) -> MongoDBClient:
    """Connect to MongoDB and return the client."""
    db_client = MongoDBClient()
    await db_client.connect(db_url)
    
    # Ensure the global db reference is set
    from mongodb import client as mongodb_client_module
    if mongodb_client_module.db is None:
        mongodb_client_module.db = db_client.db
        
    return db_client


class MongoDBChatBot:
    """A chatbot that interacts with MongoDB using LangChain."""
    
    def __init__(self, mongodb_client: MongoDBClient, tool_registry: ToolRegistry, openai_api_key: str):
        """Initialize the chatbot with MongoDB and tools."""
        self.mongodb_client = mongodb_client
        self.tool_registry = tool_registry
        self.conversation_history = []
        
        # Initialize the LangChain ChatModel with tools
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=openai_api_key,
            streaming=True
        )
        
        # Define system message
        self.system_message = SystemMessage(content="""You are a MongoDB database expert assistant connected directly to a live MongoDB database.
You have access to tools that allow you to query and modify the database.
When a user asks about data in the database, ALWAYS use the appropriate tools to fetch the data WITHOUT asking for confirmation first.

IMPORTANT: You should be proactive and use your MongoDB tools immediately to fulfill user requests.
For example, if the user asks "list 3 users", don't just say "I'll list 3 users for you" - immediately use the find tool on the users collection.

When processing user queries:
1. First, start by explaining your thought process with "I'm thinking:" followed by a brief explanation of your approach
2. Then execute the appropriate MongoDB tools
3. Finally, present the results in a clear, organized format

Always try to understand what the user is asking for and use the appropriate MongoDB tools to fulfill their request.
When appropriate, format data as tables and provide brief explanations about the data.
Be precise and helpful in your database operations.
""")
        
        # Create LangChain tools from MongoDB tools
        self.langchain_tools = self._create_langchain_tools()
    
    def _create_langchain_tools(self):
        """Create LangChain-compatible tools from MongoDB tools."""
        langchain_tools = []
        mongodb_tools = self.tool_registry.get_all_tools()
        
        for mongo_tool in mongodb_tools:
            # Create a function that will execute this tool
            async def run_tool(params: Dict[str, Any], tool=mongo_tool):
                # Set the global db reference
                from mongodb import client as mongodb_client_module
                if mongodb_client_module.db is None:
                    mongodb_client_module.db = self.mongodb_client.db
                
                # Execute the tool
                try:
                    # Ensure any MongoDB special types in tool arguments are serialized
                    serialized_params = serialize_mongodb_doc(params)
                    
                    # Call the MongoDB tool with serialized parameters
                    result = await tool.execute(serialized_params)
                    
                    # Extract the result text
                    if result and result.content and len(result.content) > 0:
                        return result.content[0].get("text", "")
                    else:
                        return "Operation completed but returned no content."
                except Exception as e:
                    return f"Error executing {tool.name}: {str(e)}"
            
            # Create a synchronous version to use with LangChain
            def sync_run_tool(params: Dict[str, Any], _run_tool=run_tool):
                # Get or create an event loop
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Already in an event loop, use asyncio.run_coroutine_threadsafe
                        import threading
                        import concurrent.futures
                        
                        # Create a new loop in a new thread
                        executor = concurrent.futures.ThreadPoolExecutor()
                        future = executor.submit(asyncio.run, _run_tool(params))
                        return future.result()
                    else:
                        # No running event loop, use run_until_complete
                        return loop.run_until_complete(_run_tool(params))
                except RuntimeError:
                    # No event loop exists, create one
                    return asyncio.run(_run_tool(params))
            
            # Create a named function for each tool
            func_name = f"run_{mongo_tool.name.replace('-', '_')}"
            tool_func = sync_run_tool
            tool_func.__name__ = func_name
            
            # Get properties and schema definitions
            properties = {}
            required = []
            
            if mongo_tool.input_schema and mongo_tool.input_schema.get("properties"):
                for name, prop in mongo_tool.input_schema.get("properties", {}).items():
                    properties[name] = {
                        "type": prop.get("type", "string"),
                        "description": prop.get("description", "")
                    }
                    
                    if name in mongo_tool.input_schema.get("required", []):
                        required.append(name)
            
            # Create a valid schema with type:object
            schema = {
                "type": "object",
                "properties": properties,
                "required": required
            }
            
            # Handle empty properties case - provide a dummy property if needed
            if not properties:
                schema["properties"] = {
                    "dummy": {
                        "type": "string",
                        "description": "Placeholder parameter (not used)"
                    }
                }
            
            # Create a Tool object
            tool = Tool(
                name=mongo_tool.name,
                description=mongo_tool.description,
                func=tool_func,
                args_schema=schema
            )
            
            langchain_tools.append(tool)
        
        return langchain_tools
    
    def _create_args_schema(self, schema: Dict[str, Any]):
        """Create a schema dictionary from a JSON schema."""
        if not schema or not schema.get("properties"):
            # Return empty schema with a dummy property
            return {
                "type": "object",
                "properties": {
                    "dummy": {
                        "type": "string",
                        "description": "Placeholder parameter (not used)"
                    }
                },
                "required": []
            }
        
        properties = {}
        required = []
        
        for name, prop in schema.get("properties", {}).items():
            properties[name] = {
                "type": prop.get("type", "string"),
                "description": prop.get("description", "")
            }
            
            if name in schema.get("required", []):
                required.append(name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
    
    async def process_query(self, query: str) -> str:
        """Process a user query and return a response."""
        # Add user message to conversation history
        human_message = HumanMessage(content=query)
        self.conversation_history.append(human_message)
        
        # Prepare the messages for LangChain
        messages = [self.system_message] + self.conversation_history
        
        try:
            # Get the LLM with tools
            llm_with_tools = self.llm.bind_tools(self.langchain_tools)
            
            # Serialize any MongoDB objects in messages before sending to OpenAI
            serialized_messages = self._serialize_conversation(messages)
            
            # Track if we're showing thinking output
            thinking_shown = False
            
            # Get the initial AI response with potential tool calls
            ai_message = await llm_with_tools.ainvoke(serialized_messages)
            
            # Check if the response contains a thinking process
            if ai_message.content and "I'm thinking:" in ai_message.content:
                thinking_shown = True
                # Extract and format the thinking part
                thinking_parts = ai_message.content.split("I'm thinking:")
                if len(thinking_parts) > 1:
                    thinking_text = "I'm thinking:" + thinking_parts[1].split("\n\n")[0]
                    print(f"Thinking process: {thinking_text}")
            
            self.conversation_history.append(ai_message)
            
            # Check if the model wants to call tools
            if hasattr(ai_message, 'tool_calls') and ai_message.tool_calls:
                num_tools = len(ai_message.tool_calls)
                print(f"AI wants to call {num_tools} tools")
                
                tool_results = []
                
                # Process each tool call
                for tool_call in ai_message.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]
                    
                    print(f"Calling tool: {tool_name}")
                    # Use our custom JSON serializer to handle MongoDB types
                    print(f"With arguments: {mongodb_json_dumps(tool_args)}")
                    
                    # Find the matching tool
                    matching_tool = next((t for t in self.langchain_tools if t.name == tool_name), None)
                    
                    if matching_tool:
                        try:
                            # Direct async handling of MongoDB tools
                            try:
                                # Find the original MongoDB tool
                                mongodb_tool = next((t for t in self.tool_registry.get_all_tools() if t.name == tool_name), None)
                                
                                # Set global MongoDB reference if needed
                                from mongodb import client as mongodb_client_module
                                if mongodb_client_module.db is None:
                                    mongodb_client_module.db = self.mongodb_client.db
                                
                                # Direct async execution
                                if mongodb_tool:
                                    result = await mongodb_tool.execute(tool_args)
                                    if result and result.content and len(result.content) > 0:
                                        tool_result = result.content[0].get("text", "")
                                    else:
                                        tool_result = "Operation completed but returned no content."
                                else:
                                    # Fallback to using LangChain tool
                                    tool_result = matching_tool.func(tool_args)
                                    
                            except Exception as e:
                                print(f"Direct execution failed, falling back to tool func: {str(e)}")
                                # Fallback to the standard tool function
                                tool_result = matching_tool.func(tool_args)
                            
                            # Add the tool result to the conversation
                            tool_message = ToolMessage(
                                content=str(tool_result),
                                tool_call_id=tool_id
                            )
                            self.conversation_history.append(tool_message)
                            tool_results.append({
                                "tool_name": tool_name,
                                "result": tool_result
                            })
                            print(f"Tool result: {str(tool_result)[:100]}...")
                            
                        except Exception as e:
                            error_msg = f"Error executing tool {tool_name}: {str(e)}"
                            print(error_msg)
                            import traceback
                            traceback.print_exc()
                            
                            # Add error result to conversation
                            tool_message = ToolMessage(
                                content=f"Error: {str(e)}",
                                tool_call_id=tool_id
                            )
                            self.conversation_history.append(tool_message)
                    else:
                        # Tool not found
                        error_msg = f"Tool {tool_name} not found"
                        print(error_msg)
                        tool_message = ToolMessage(
                            content=f"Error: {error_msg}",
                            tool_call_id=tool_id
                        )
                        self.conversation_history.append(tool_message)
                
                # Get a final response based on the tool results
                print("Getting final response based on tool results...")
                serialized_messages = self._serialize_conversation([self.system_message] + self.conversation_history)
                
                # Add a specific instruction to structure the response with the thinking process
                if not thinking_shown:
                    serialized_messages.append(
                        HumanMessage(content="Please present your findings clearly. If you haven't already, start with a brief explanation of what you did to get these results.")
                    )
                
                final_response = await self.llm.ainvoke(serialized_messages)
                self.conversation_history.append(final_response)
                
                return final_response.content
            else:
                # No tool calls, just return the AI message content
                return ai_message.content
                
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return f"I encountered an error: {str(e)}"
    
    def _serialize_conversation(self, messages):
        """Serialize MongoDB objects in the conversation for proper JSON conversion."""
        serialized_conversation = []
        
        for msg in messages:
            # Handle messages with tool calls
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                # Create new message with serialized tool calls
                serialized_msg = AIMessage(
                    content=msg.content,
                    tool_calls=[
                        {
                            "name": tc["name"],
                            "id": tc["id"], 
                            "args": serialize_mongodb_doc(tc["args"])
                        }
                        for tc in msg.tool_calls
                    ]
                )
                serialized_conversation.append(serialized_msg)
            # Handle tool messages with tool_call_id
            elif hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                # Make sure any MongoDB objects in content are serialized
                if isinstance(msg.content, str):
                    # Keep as is
                    serialized_conversation.append(msg)
                else:
                    # Serialize non-string content
                    serialized_content = serialize_mongodb_doc(msg.content)
                    from langchain_core.messages import ToolMessage
                    serialized_msg = ToolMessage(
                        content=str(serialized_content),
                        tool_call_id=msg.tool_call_id
                    )
                    serialized_conversation.append(serialized_msg)
            else:
                # No tool calls, just add the original message
                serialized_conversation.append(msg)
                
        return serialized_conversation


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