import os
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage, trim_messages
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.conversation.memory import ConversationBufferMemory

from mongodb.client import MongoDBClient
from agents.tools.registry import ToolRegistry
from agents.base import base_agent as HelperAgent
from agents.utils.serialization_utils import serialize_mongodb_doc, mongodb_json_dumps
from core.moderation.fallback_handler import handle_fallback

# Define message intent types as string constants
INTENT_CASUAL_CONVERSATION = "casual_conversation"
INTENT_DATABASE_QUERY = "database_query"
INTENT_MIXED = "mixed"


class MongoDBChatBot:
    """A chatbot that interacts with MongoDB using LangChain."""
    
    def __init__(self, mongodb_client: MongoDBClient, tool_registry: ToolRegistry, openai_api_key: str):
        """Initialize the chatbot with MongoDB and tools."""
        self.mongodb_client = mongodb_client
        self.tool_registry = tool_registry
        self.fallback_threshold = 0.6  # Confidence threshold below which to trigger fallback
        
        # Initialize memory for conversation context
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize the LangChain ChatModel with tools
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo-0125",  # Latest GPT-3.5-turbo with 16k context
            temperature=0,
            api_key=openai_api_key,
            streaming=True
        )
        
        # Initialize message trimmer for token limit management
        # GPT-3.5-turbo-0125 has 16k context, we'll use ~12k for messages, ~4k for response
        self.message_trimmer = trim_messages(
            max_tokens=12000,  # Leave room for response and tool outputs
            strategy="last",   # Keep most recent messages
            token_counter=self.llm,  # Use actual model for accurate token counting
            include_system=True,     # Always keep system message
            allow_partial=False,     # Don't cut messages in half
            start_on="human"        # Start trimming from human messages
        )

        # Define system message
        self.system_message = SystemMessage(content="""You are a friendly, conversational Medical expert assistant connected directly to a live MongoDB database.
You have access to tools that allow you to query and modify the database.
When a user asks about data in the database, ALWAYS use the appropriate tools to fetch the data WITHOUT asking for confirmation first.

IMPORTANT INSTRUCTIONS:
1. When answering questions about database content, use multiple tools sequentially if needed.
2. First determine what collections exist, then query the relevant collections for the specific data.
3. If the first tool execution doesn't provide complete information, call additional tools to get more details.
4. Follow through with additional tool calls until you have the complete information the user requested.
5. If after using tools you still cannot find relevant information, be honest and acknowledge that you don't have the information.

CONVERSATION STYLE:
1. Be conversational, warm, and natural in your responses.
2. Avoid robotic phrases like "I'm thinking:" or debug statements in your final responses.
3. Present results in a clear, organized format with appropriate spacing.
4. For medical information, be precise but explain concepts in accessible language.

Always try to understand what the user is asking for and use the appropriate MongoDB tools to fulfill their request.
When appropriate, format data as tables and provide brief explanations about the data.
""")
        
        # Create LangChain tools from MongoDB tools
        self.langchain_tools = self.tool_registry._create_langchain_tools()
        
        # Create intent classifier
        self.intent_classifier = HelperAgent._create_intent_classifier(self.llm)
        
        # Create conversation chain
        self.casual_conversation_chain = HelperAgent._create_casual_conversation_chain(self.llm)
            
    async def _classify_intent(self, query: str) -> str:
        """Classify the intent of a user query"""
        intent_result = await self.intent_classifier.ainvoke({"query": query})
        intent_text = intent_result.content.strip().lower()
        
        # Validate the intent type
        if intent_text not in [INTENT_CASUAL_CONVERSATION, INTENT_DATABASE_QUERY, INTENT_MIXED]:
            # Default to database query if classification fails
            print(f"Intent classification failed, got: {intent_text}")
            return INTENT_DATABASE_QUERY
        
        return intent_text
        
    async def process_query(self, query: str) -> str:
        """Process a user query and return a response."""
        # Add user message to memory (single source of truth)
        self.memory.chat_memory.add_user_message(query)
        
        # Classify the intent of the query
        intent = await self._classify_intent(query)
        print(f"Classified intent: {intent}")
        
        # For casual conversation, use the specialized chain
        if intent == INTENT_CASUAL_CONVERSATION:
            print("Handling as casual conversation")
            response = await self.casual_conversation_chain.ainvoke({"query": query})
            response_text = response.content
            
            # Store AI response in memory
            self.memory.chat_memory.add_ai_message(response_text)
            
            return response_text
        
        # For database queries or mixed intent, use the original tool-based approach
        # Prepare the messages for LangChain with trimming
        all_messages = [self.system_message] + self.memory.chat_memory.messages
        
        try:
            # Apply message trimming to prevent token limit issues
            try:
                messages = self.message_trimmer.invoke(all_messages)
                print(f"Trimmed messages from {len(all_messages)} to {len(messages)}")
            except Exception as e:
                print(f"Message trimming failed: {e}, using fallback")
                # Fallback: keep system message + last 10 messages
                messages = [self.system_message] + self.memory.chat_memory.messages[-10:]
            
            # Get the LLM with tools
            llm_with_tools = self.llm.bind_tools(self.langchain_tools)
            
            # Serialize any MongoDB objects in messages before sending to OpenAI
            serialized_messages = HelperAgent._serialize_conversation(messages)
            
            # Track if we're showing thinking output
            thinking_shown = False
            
            # Process for multi-turn tool calls
            tools_executed = 0
            MAX_TOOL_CALLS = 5  # Reasonable limit to prevent infinite loops
            tool_findings = []  # Track the results of all tool calls
            current_messages = serialized_messages.copy()  # Track current conversation state
            
            while tools_executed < MAX_TOOL_CALLS:
                # Get the AI response with potential tool calls
                ai_message = await llm_with_tools.ainvoke(current_messages)
                
                # Check if the response contains a thinking process
                if ai_message.content and "I'm thinking:" in ai_message.content:
                    thinking_shown = True
                    # Extract and format the thinking part
                    thinking_parts = ai_message.content.split("I'm thinking:")
                    if len(thinking_parts) > 1:
                        thinking_text = "I'm thinking:" + thinking_parts[1].split("\n\n")[0]
                        print(f"Thinking process: {thinking_text}")
                
                current_messages.append(ai_message)
                
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
                                
                                # Add the tool result to the current conversation
                                tool_message = ToolMessage(
                                    content=str(tool_result),
                                    tool_call_id=tool_id
                                )
                                current_messages.append(tool_message)
                                
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
                                current_messages.append(tool_message)
                                empty_results_count += 1
                        else:
                            # Tool not found
                            error_msg = f"Tool {tool_name} not found"
                            print(error_msg)
                            tool_message = ToolMessage(
                                content=f"Error: {error_msg}",
                                tool_call_id=tool_id
                            )
                            current_messages.append(tool_message)
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
                        current_messages.append(evaluation_prompt)
                        # Continue the loop for another round of potential tool calls
                    else:
                        # No more tools needed or max tools reached
                        break
                else:
                    # No tool calls in this message, break the loop
                    break
            
            # Get a final response based on all the tool results
            print("Getting final response based on all tool results...")
            
            # Add a specific instruction for the final response
            if not thinking_shown:
                current_messages.append(
                    HumanMessage(content="Please present your findings clearly. If you haven't already, start with a brief explanation of what you did to get these results.")
                )
            
            final_response = await self.llm.ainvoke(current_messages)
            
            # Store the final AI response in memory
            self.memory.chat_memory.add_ai_message(final_response.content)
            
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
            confidence_score = await HelperAgent._evaluate_response_quality(self.llm, query, response_text)
            print(f"Response confidence score: {confidence_score}")
            
            # Determine if we need to use the fallback
            need_fallback = confidence_score < self.fallback_threshold or uncertainty_detected
            
            if need_fallback:
                print(f"Fallback needed. Confidence: {confidence_score}, Uncertainty detected: {uncertainty_detected}")
                fallback_message = await handle_fallback(query, response_text, confidence_score)
                # Store in memory
                self.memory.chat_memory.add_ai_message(fallback_message)
                return fallback_message
            else:
                # Store in memory for future conversations
                self.memory.chat_memory.add_ai_message(response_text)
                return response_text
                
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return f"I encountered an error: {str(e)}"