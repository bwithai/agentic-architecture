"""
MongoDBChatBot implementation for AI agent application.
This class handles the interaction between the user, LangChain, and MongoDB tools.
"""

import os
import asyncio
from typing import Dict, List, Any, Optional
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import Tool

from mongodb.client import MongoDBClient
from agents.tools.registry import ToolRegistry
from agents.utils.serialization_utils import serialize_mongodb_doc, mongodb_json_dumps
from core.chatbot.fallback_handler import handle_fallback


class MongoDBChatBot:
    """A chatbot that interacts with MongoDB using LangChain."""
    
    def __init__(self, mongodb_client: MongoDBClient, tool_registry: ToolRegistry, openai_api_key: str):
        """Initialize the chatbot with MongoDB and tools."""
        self.mongodb_client = mongodb_client
        self.tool_registry = tool_registry
        self.conversation_history = []
        self.fallback_threshold = 0.6  # Confidence threshold below which to trigger fallback
        
        # Initialize the LangChain ChatModel with tools
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=openai_api_key,
            streaming=True
        )
        
        # Define system message
        self.system_message = SystemMessage(content="""You are a Medical expert assistant connected directly to a live MongoDB database.
You have access to tools that allow you to query and modify the database.
When a user asks about data in the database, ALWAYS use the appropriate tools to fetch the data WITHOUT asking for confirmation first.

IMPORTANT INSTRUCTIONS:
1. When answering questions about database content, you MUST use multiple tools sequentially if needed.
2. First determine what collections exist, then query the relevant collections for the specific data.
3. If the first tool execution doesn't provide complete information, call additional tools to get more details.
4. NEVER stop with just high-level information (like just listing collection names) when the user is clearly asking for specific data.
5. Always follow through with additional tool calls until you have the complete information the user requested.
6. If after using tools you still cannot find relevant information, BE HONEST and acknowledge that you don't have the information.

When processing user queries:
1. First, start by explaining your thought process with "I'm thinking:" followed by a brief explanation of your approach
2. Then execute the appropriate MongoDB tools, using multiple tools in sequence when needed
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
    
    async def _evaluate_response_quality(self, query: str, response: str) -> float:
        """
        Evaluate the quality of the response relative to the user query.
        Returns a confidence score (0-1) indicating how well the response addresses the query.
        """
        # Create a prompt to ask the LLM to evaluate the response
        evaluation_prompt = f"""
        You are strictly evaluating the quality of an AI assistant's response to a user query.
        
        User Query: {query}
        
        Assistant Response: {response}
        
        On a scale of 0 to 1, how well did the assistant's response answer the user's query?
        Consider the following factors:
        - Relevance: Did the response address what the user was asking about?
        - Completeness: Did the response provide all the information the user requested?
        - Accuracy: Is the information provided likely to be correct based on available data?
        - Clarity: Is the response clear and understandable?
        
        Return only a single decimal number between 0 and 1 representing your confidence score.
        If the response indicates the assistant couldn't find information or returned empty/null results, give a lower score.
        """
        
        # Create a simple evaluator model with lower temperature
        evaluator = ChatOpenAI(
            model="gpt-3.5-turbo",  # Using a cheaper model for evaluation
            temperature=0,
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Get the evaluation
        evaluation_response = await evaluator.ainvoke([HumanMessage(content=evaluation_prompt)])
        
        # Try to extract a confidence score (float between 0-1)
        try:
            # Strip any non-numeric characters and convert to float
            confidence_text = evaluation_response.content.strip()
            # Extract the first decimal number from the response
            confidence_match = re.search(r'0\.\d+|\d+\.\d+|0|1', confidence_text)
            if confidence_match:
                confidence = float(confidence_match.group())
                # Ensure it's between 0 and 1
                confidence = max(0.0, min(1.0, confidence))
                return confidence
            else:
                # Default moderate confidence if no number found
                print(f"Could not extract confidence score from: {confidence_text}")
                return 0.5
        except Exception as e:
            print(f"Error parsing confidence score: {e}, defaulting to 0.5")
            return 0.5

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
            
            # Process for multi-turn tool calls
            tools_executed = 0
            MAX_TOOL_CALLS = 5  # Reasonable limit to prevent infinite loops
            tool_findings = []  # Track the results of all tool calls
            
            while tools_executed < MAX_TOOL_CALLS:
                # Get the AI response with potential tool calls
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
                    needs_more_tools = False
                    empty_results_count = 0  # Track empty or error results
                    
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
                                
                                # Track empty or minimal results for fallback detection
                                if (not tool_result) or tool_result == "[]" or "no content" in tool_result.lower():
                                    empty_results_count += 1
                                
                                # Save tool result for final evaluation
                                tool_results.append({
                                    "tool_name": tool_name,
                                    "args": tool_args,
                                    "result": tool_result
                                })
                                tool_findings.append(tool_results[-1])
                                
                                print(f"Tool result: {str(tool_result)[:100]}...")
                                
                                # If this was a list_collections tool, likely need more tools
                                if tool_name == "list_collections" or tool_name == "get_collection_info":
                                    needs_more_tools = True
                                
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
                                empty_results_count += 1
                        else:
                            # Tool not found
                            error_msg = f"Tool {tool_name} not found"
                            print(error_msg)
                            tool_message = ToolMessage(
                                content=f"Error: {error_msg}",
                                tool_call_id=tool_id
                            )
                            self.conversation_history.append(tool_message)
                            empty_results_count += 1
                    
                    # Update the counter of tools executed
                    tools_executed += num_tools
                    
                    # Check if all tool results were empty or errors - potential fallback situation
                    if empty_results_count == num_tools:
                        needs_more_tools = True  # Try one more round of tools
                    
                    # Prompt AI to evaluate if more tools are needed
                    if needs_more_tools and tools_executed < MAX_TOOL_CALLS:
                        evaluation_prompt = HumanMessage(content="""
Based on the tools you've used so far, evaluate if you have fully answered the user's question.
Do you need to call additional tools to provide complete information? 
If you need to call more tools, do so directly without asking for confirmation.
""")
                        self.conversation_history.append(evaluation_prompt)
                        serialized_messages = self._serialize_conversation([self.system_message] + self.conversation_history)
                        # Continue the loop for another round of potential tool calls
                    else:
                        # No more tools needed or max tools reached
                        break
                else:
                    # No tool calls in this message, break the loop
                    break
            
            # Get a final response based on all the tool results
            print("Getting final response based on all tool results...")
            serialized_messages = self._serialize_conversation([self.system_message] + self.conversation_history)
            
            # Add a specific instruction for the final response
            if not thinking_shown:
                serialized_messages.append(
                    HumanMessage(content="Please present your findings clearly. If you haven't already, start with a brief explanation of what you did to get these results.")
                )
            
            final_response = await self.llm.ainvoke(serialized_messages)
            self.conversation_history.append(final_response)
            
            # Now check if we should engage the fallback mechanism
            # Conditions for fallback:
            # 1. All tools returned empty or error results
            # 2. The final response contains indicators of uncertainty
            # 3. The response quality is below threshold
            
            response_text = final_response.content
            
            # Check if response indicates no information found
            uncertainty_phrases = [
                "i don't have", 
                "i couldn't find", 
                "no information", 
                "not found", 
                "doesn't exist",
                "no data",
                "no results",
                "could not locate",
                "unable to find",
                "not available"
            ]
            
            uncertainty_detected = any(phrase in response_text.lower() for phrase in uncertainty_phrases)
            
            # Get a quality score for the response
            confidence_score = await self._evaluate_response_quality(query, response_text)
            print(f"Response confidence score: {confidence_score}")
            
            # Determine if we need to use the fallback
            need_fallback = confidence_score < self.fallback_threshold or uncertainty_detected
            
            if need_fallback:
                print(f"Fallback needed. Confidence: {confidence_score}, Uncertainty detected: {uncertainty_detected}")
                fallback_message = await handle_fallback(query, response_text, confidence_score)
                return fallback_message
            else:
                return response_text
                
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