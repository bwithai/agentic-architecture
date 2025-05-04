"""
Agent Graph

This module defines the LangGraph implementation that connects the intent classifier
and query understanding agents into a coherent workflow.
"""

import json
from typing import Dict, Any
from datetime import datetime
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from agents.base.base_agent import AgentInput, AgentOutput
from agents.specialized.query_understanding_agent import QueryUnderstandingAgent, MongoDBQuery
from agents.specialized.intent_classifier_agent import IntentClassifierAgent
from agents.specialized.mongodb_agent import MongoDBAgent
from agents.specialized.response_formatting_agent import ResponseFormattingAgent
from config.config import config


# Custom JSON encoder to handle datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super(DateTimeEncoder, self).default(obj)


async def classify_intent(state: MessagesState) -> MessagesState:
    """
    Node function to classify if the input is general conversation or database query.
    Also detects language and handles translation.
    
    Args:
        state (MessagesState): The current state with message history
    Returns:
        MessagesState: The updated state
    """
    # Extract the latest user message
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if not user_messages:
        return state
    
    latest_message = user_messages[-1].content
    
    # Create and run the intent classifier agent
    agent = IntentClassifierAgent(verbose=config.agent.verbose)
    agent_input = AgentInput(query=latest_message)
    
    output = await agent.run(agent_input)
    
    # Add the intent classification to the state as a 'tool' message
    classification = output.data.get("classification", {})
    # if the enten_type is not Business Inquiry, then it is GENERAL_CONVERSATION
    intent_type = classification.get("intent_type", "GENERAL_CONVERSATION")

    # Store the language information in state
    language_info = output.data.get("language_info", {})
    if "query_result" not in state:
        state["query_result"] = {}
    state["query_result"]["language_info"] = language_info
    
    # Log language detection
    if language_info:
        lang_name = language_info.get("language_name", "Unknown")
        is_eng = language_info.get("is_english", True)
        state["messages"].append(
            ToolMessage(
                content=f"Detected language: {lang_name} (Translation needed: {not is_eng})",
                tool_call_id="language_detector"
            )
        )
    
    state["messages"].append(
        ToolMessage(
            content=f"Intent classified as: {intent_type}",
            tool_call_id="intent_classifier"
        )
    )
    
    # If it's general conversation, add the response directly
    if intent_type == "GENERAL_CONVERSATION" and output.response:
        state["messages"].append(
            AIMessage(content=output.response)
        )
    
    return state


async def understand_query(state: MessagesState) -> MessagesState:
    """
    Node function to convert user query to MongoDB query using database schema.
    
    Args:
        state (MessagesState): The current state
    Returns:
        MessagesState: The updated state with query results
    """
    # Extract the latest user message
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if not user_messages:
        state["messages"].append(
            ToolMessage(
                content="Error: No user query found",
                tool_call_id="query_understanding"
            )
        )
        return state
    
    latest_message = user_messages[-1].content
    
    # Get language information from previous node
    language_info = state.get("query_result", {}).get("language_info", {})
    
    # If the language is not English and we have a translated query, use it
    if language_info and not language_info.get("is_english", True):
        query_to_process = language_info.get("translated_query", latest_message)
    else:
        query_to_process = latest_message
    
    # Create and run the query understanding agent
    agent = QueryUnderstandingAgent(verbose=config.agent.verbose)
    agent_input = AgentInput(query=query_to_process)
    
    output = await agent.run(agent_input)
    
    # Store results in state for the format_response node
    if "query_result" not in state:
        state["query_result"] = {}
        
    state["query_result"].update({
        "original_query": latest_message,
        "status": output.status,
        "error": output.error
    })
    
    # Preserve language information
    if language_info and "language_info" not in state["query_result"]:
        state["query_result"]["language_info"] = language_info
    
    if output.status == "error":
        state["messages"].append(
            ToolMessage(
                content=f"Error: {output.error}",
                tool_call_id="query_understanding"
            )
        )
        return state
    
    # Update the state
    mongodb_query = output.data.get("mongodb_query")
    explanation = output.data.get("explanation", "")
    
    # Store the mongodb query in state for the format_response node
    state["query_result"]["mongodb_query"] = mongodb_query
    state["query_result"]["explanation"] = explanation
    
    # Convert to proper JSON string if it's a dict
    if isinstance(mongodb_query, dict):
        mongodb_query_str = json.dumps(mongodb_query)
    else:
        mongodb_query_str = str(mongodb_query)
    
    state["messages"].append(
        ToolMessage(
            content=f"MongoDB query: {mongodb_query_str}\nExplanation: {explanation}",
            tool_call_id="query_understanding"
        )
    )
    
    # Extract limits from the user query
    import re
    # Look for patterns like "2 users" or "3 products"
    limit_patterns = {
        "users": re.search(r'(\d+)\s+(user|users)', query_to_process.lower()),
        "products": re.search(r'(\d+)\s+(product|products)', query_to_process.lower())
    }
    
    limits = {}
    for collection, match in limit_patterns.items():
        if match:
            limits[collection] = int(match.group(1))
    
    # Now run the MongoDB agent to execute the query
    try:
        # Connect to MongoDB directly
        from pymongo import MongoClient
        
        # Initialize MongoDB client with proper error handling
        client = MongoClient(config.mongodb.uri, serverSelectionTimeoutMS=5000)
        # Test connection before proceeding
        client.admin.command('ping')
        
        db = client[config.mongodb.database]
        db_name = config.mongodb.database
        
        all_results = {}
        collections_to_query = []
        
        # Handle multi-collection queries
        if isinstance(mongodb_query, list):
            # The query is a list of separate collection queries
            collections_to_query = mongodb_query
        elif isinstance(mongodb_query, dict) and 'collection' in mongodb_query and 'query' in mongodb_query:
            # Single collection query
            collections_to_query = [mongodb_query]
        
        # Handle numeric limits in the query like "first 3 users"
        numeric_limit_match = re.search(r'(?:top|first|latest|recent)\s+(\d+)', query_to_process.lower())
        default_limit = int(numeric_limit_match.group(1)) if numeric_limit_match else 5
        
        # If we need to query both users and products but the agent didn't generate both queries
        if "users" in limits and "products" in limits and len(collections_to_query) < 2:
            # Check if we need to add a products query
            if not any(q.get("collection") == "products" for q in collections_to_query):
                collections_to_query.append({"collection": "products", "query": {}})
            # Check if we need to add a users query
            if not any(q.get("collection") == "users" for q in collections_to_query):
                collections_to_query.append({"collection": "users", "query": {}})
                
        # If no specific collections were identified but we mentioned users or products
        if not collections_to_query:
            if "user" in query_to_process.lower() or "customer" in query_to_process.lower():
                collections_to_query.append({"collection": "users", "query": {}})
            if "product" in query_to_process.lower() or "item" in query_to_process.lower():
                collections_to_query.append({"collection": "products", "query": {}})
                
        # Last fallback - if still no collections identified, check available collections
        if not collections_to_query:
            available_collections = db.list_collection_names()
            if available_collections:
                collections_to_query.append({
                    "collection": available_collections[0], 
                    "query": {}
                })
        
        for query_item in collections_to_query:
            collection_name = query_item["collection"]
            query_filter = query_item["query"]
            
            # Determine appropriate limit for this collection
            query_limit = default_limit
            
            # Apply user-specified limits if available
            if collection_name == "users" and "users" in limits:
                query_limit = limits["users"]
            elif collection_name == "products" and "products" in limits:
                query_limit = limits["products"]
            
            try:
                # Check if collection exists before querying
                if collection_name in db.list_collection_names():
                    # Execute the query
                    collection = db[collection_name]
                    cursor = collection.find(query_filter).limit(query_limit)
                    
                    # Convert cursor to list and handle ObjectIds and datetime objects
                    results = []
                    for doc in cursor:
                        # Convert ObjectId to string for JSON serialization
                        if "_id" in doc:
                            doc["_id"] = str(doc["_id"])
                        results.append(doc)
                    
                    all_results[collection_name] = results
                    
                    state["messages"].append(
                        ToolMessage(
                            content=f"Query executed successfully. Found {len(results)} results in collection '{collection_name}'.",
                            tool_call_id=f"mongodb_agent_{collection_name}"
                        )
                    )
                else:
                    # Collection doesn't exist
                    all_results[collection_name] = []
                    state["messages"].append(
                        ToolMessage(
                            content=f"Collection '{collection_name}' does not exist in database '{db_name}'.",
                            tool_call_id=f"mongodb_agent_{collection_name}"
                        )
                    )
            except Exception as coll_err:
                # Handle collection-specific errors
                state["messages"].append(
                    ToolMessage(
                        content=f"Error querying collection '{collection_name}': {str(coll_err)}",
                        tool_call_id=f"mongodb_agent_{collection_name}"
                    )
                )
                all_results[collection_name] = []
        
        # Store results in state for the format_response node
        state["query_result"]["multi_collection_results"] = all_results
        state["query_result"]["count"] = sum(len(results) for results in all_results.values())
        state["query_result"]["database_name"] = db_name
        state["query_result"]["collections_queried"] = list(all_results.keys())
        
        # Close MongoDB connection
        client.close()
        
    except Exception as e:
        # Handle errors
        error_msg = str(e)
        state["messages"].append(
            ToolMessage(
                content=f"Error executing MongoDB query: {error_msg}",
                tool_call_id="mongodb_agent"
            )
        )
        
        state["query_result"]["status"] = "error"
        state["query_result"]["error"] = error_msg
        state["query_result"]["result"] = []
        state["query_result"]["count"] = 0
    
    return state


async def format_response(state: MessagesState) -> MessagesState:
    """
    Node function to format the query results into a user-friendly response.
    
    Args:
        state (MessagesState): The current state
    Returns:
        MessagesState: The updated state
    """
    # Get query results from state
    query_result = state.get("query_result", {})
    original_query = query_result.get("original_query", "")
    
    # Create and run the response formatting agent
    formatter_agent = ResponseFormattingAgent(verbose=config.agent.verbose)
    formatter_input = AgentInput(
        query=original_query,
        context=query_result
    )
    
    formatter_output = await formatter_agent.run(formatter_input)
    
    # Add the formatted response to the state
    state["messages"].append(
        AIMessage(content=formatter_output.response)
    )
    
    return state


def determine_next_step(state: MessagesState) -> str:
    """
    Determine next step based on intent classification.
    
    Args:
        state (MessagesState): The current state
    Returns:
        str: Next node name
    """
    # Check if we already have an assistant response (for general conversation)
    assistant_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
    if assistant_messages:
        # For general conversation, end the flow
        return END
    
    # Check intent type
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage) and msg.tool_call_id == "intent_classifier":
            if "Intent classified as: GENERAL_CONVERSATION" in msg.content:
                return END
            else:
                # For business inquiries, continue to query understanding
                return "understand_query"
    
    # Default case
    return "understand_query"


def create_agent_graph() -> StateGraph:
    """
    Create and configure the agent graph.
    
    Returns:
        StateGraph: The configured agent graph
    """
    # Create a new graph with MessagesState
    graph = StateGraph(MessagesState)
    
    # Add nodes to the graph
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("understand_query", understand_query)
    graph.add_node("format_response", format_response)
    
    # Entry point
    graph.set_entry_point("classify_intent")
    
    # Route based on intent classification
    graph.add_conditional_edges(
        "classify_intent",
        determine_next_step,
        {
            "understand_query": "understand_query",
            END: END
        }
    )
    
    # After understand_query, go to format_response
    graph.add_edge("understand_query", "format_response")
    
    # After format_response, end the flow
    graph.add_edge("format_response", END)
    
    # Compile the graph
    return graph.compile()
