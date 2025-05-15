"""
Agent Graph

This module defines the LangGraph implementation that connects the intent classifier
and query understanding agents into a coherent workflow.
"""

import json
import re
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
        MessagesState: The updated state with language info and intent classification
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
    intent_type = classification.get("intent_type", "BUSINESS_INQUIRY")
    
    # Store the language information in a tool message for downstream nodes
    language_info = output.data.get("language_info", {})
    
    # Log language detection
    if language_info:
        lang_name = language_info.get("language_name", "Unknown")
        is_eng = language_info.get("is_english", True)
        
        # Store language info in a tool message
        language_info_message = ToolMessage(
            content=f"Detected language: {lang_name} (Translation needed: {not is_eng})",
            tool_call_id="language_detector"
        )
        # Add additional metadata to the message
        language_info_message.additional_kwargs = {"language_info": language_info}
        state["messages"].append(language_info_message)
    
    state["messages"].append(
        ToolMessage(
            content=f"Intent classified as: {intent_type}",
            tool_call_id="intent_classifier"
        )
    )
    
    # If it's general conversation, add the response directly
    # (The response is already translated back to the original language by the agent)
    if intent_type == "GENERAL_CONVERSATION" and output.response:
        state["messages"].append(
            AIMessage(content=output.response)
        )
    
    return state


def run_mongo_query(db, query_str):
    """
    Given a database `db` and a string like "users.find({'x':1}).limit(10)",
    parse it, execute it, and return the resulting cursor or list.
    """

    # 1) Extract collection name + full chain of methods
    m = re.match(r'^(?P<coll>\w+)\.(?P<chain>.+)$', query_str.strip())
    if not m:
        raise ValueError(f"Query must start with '<collection>.' but got: {query_str!r}")

    coll_name = m.group('coll')
    chain    = m.group('chain')

    coll = db[coll_name]

    # 2) Split chain into individual calls, e.g. ["find({'x':1})", "limit(10)"]
    calls = re.findall(r'(\w+\([^)]*\))', chain)

    # Start by executing the first call to get a cursor or result
    obj = coll
    for call in calls:
        # parse method name and raw args inside parentheses
        method_name, raw_args = re.match(r'(\w+)\((.*)\)', call).groups()

        # build a Python-friendly args list
        args = []
        kwargs = {}
        if raw_args.strip():
            # If it looks like JSON (keys in quotes, etc.), use json.loads
            # else fall back to literal_eval
            try:
                # wrap in [] so we can decode multiple comma‐separated args
                decoded = json.loads(f'[{raw_args}]')
                args = decoded
            except json.JSONDecodeError:
                # For simple Python literals: numbers, tuples, etc.
                from ast import literal_eval
                args = list(literal_eval(f'({raw_args},)'))

        # dispatch to pymongo object (Collection or Cursor)
        fn = getattr(obj, method_name)
        obj = fn(*args, **kwargs)

    return obj

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
    
    # Get language information from language_detector message
    language_info = {}
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage) and msg.tool_call_id == "language_detector":
            language_info = msg.additional_kwargs.get("language_info", {})
            break

    # If the language is not English and we have a translated query, use it
    if language_info and not language_info.get("is_english", True):
        query_to_process = language_info.get("translated_query", latest_message)
    else:
        query_to_process = latest_message
    
    # Create and run the query understanding agent
    agent = QueryUnderstandingAgent(verbose=config.agent.verbose)
    agent_input = AgentInput(query=query_to_process)
    
    output = await agent.run(agent_input)
    
    # Store results in a tool message for format_response node
    query_result = {
        "original_query": query_to_process,
        "status": output.status,
        "error": output.error,
        "mongodb_query": output.data.get("mongodb_query", "")
    }
    
    if output.status == "error":
        error_message = ToolMessage(
            content=f"Error: {output.error}",
            tool_call_id="query_understanding"
        )
        # Store additional metadata in the message
        error_message.additional_kwargs = {"query_result": query_result, "language_info": language_info}
        state["messages"].append(error_message)
        return state
    
    # Update the query result
    mongodb_query = output.data.get("mongodb_query")
    
    query_message = ToolMessage(
        content=f"mongodb_query: {mongodb_query}",
        tool_call_id="query_understanding"
    )
    # Store query result in the message
    query_message.additional_kwargs = {"query_result": query_result, "language_info": language_info}
    state["messages"].append(query_message)
    
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
        for key, qstr in mongodb_query.items():
            cursor = run_mongo_query(db, qstr)
            # cursor might be a Cursor (for find/aggregate) or the direct return value
            # of an operation (e.g. distinct())
            # If iterable, cast to list; otherwise, keep as-is.
            try:
                all_results[key] = list(cursor)
            except TypeError:
                all_results[key] = cursor

        # Print all results
        print("All results: ", all_results)
        input("1. Press Enter to continue...")
        print("MongoDB query: ", mongodb_query)
        input("2. Press Enter to continue...")
        
        # If no specific collections were identified but we mentioned users or products
        if isinstance(mongodb_query, dict):
            for key, value in mongodb_query.items():
                # value have query like this: "users.find({}).limit(10)"
                query = value.split(".")
                if len(query) == 3:
                    cursor = db[query[1]].query[2]
                    results = list(cursor)
                    print("Results: ", results)
                    input("2. Press Enter to continue...")
                
                db_message = ToolMessage(
                    content=f"Query executed successfully. Found {len(results)} results in collection '{key}'.",
                    tool_call_id=f"mongodb_agent_{key}"
                )
                state["messages"].append(db_message)
        else:
            # Collection doesn't exist or invalid query format  
            state["messages"].append(
                ToolMessage(
                    content=f"Invalid MongoDB query format: {mongodb_query}",
                    tool_call_id="mongodb_agent_error"
                )
            )
    except Exception as coll_err:
        # Handle collection-specific errors
        state["messages"].append(
            ToolMessage(
                content=f"Error executing MongoDB query: {str(coll_err)}",
                tool_call_id="mongodb_agent_error"
            )
        )
        
        # Update query result with data
        query_result["multi_collection_results"] = all_results
        query_result["count"] = sum(len(results) for results in all_results.values())
        query_result["database_name"] = db_name
        query_result["collections_queried"] = list(all_results.keys())
        
        # Store final results in a tool message
        result_message = ToolMessage(
            content=f"Query results retrieved from {len(all_results)} collections with {query_result['count']} total documents.",
            tool_call_id="mongodb_results"
        )
        # Store the full results in additional_kwargs
        result_message.additional_kwargs = {
            "query_result": query_result, 
            "language_info": language_info
        }
        state["messages"].append(result_message)
        
        # Close MongoDB connection
        client.close()
        
    except Exception as e:
        # Handle errors
        error_msg = str(e)
        error_message = ToolMessage(
            content=f"Error executing MongoDB query: {error_msg}",
            tool_call_id="mongodb_agent"
        )
        
        # Update query result with error info
        query_result["status"] = "error"
        query_result["error"] = error_msg
        query_result["result"] = []
        query_result["count"] = 0
        
        # Store error details in the message
        error_message.additional_kwargs = {"query_result": query_result, "language_info": language_info}
        state["messages"].append(error_message)
    
    return state


async def format_response(state: MessagesState) -> MessagesState:
    """
    Node function to format the query results into a user-friendly response.
    
    Args:
        state (MessagesState): The current state
    Returns:
        MessagesState: The updated state with formatted response
    """
    # Extract the latest user message
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    original_query = user_messages[-1].content if user_messages else ""
    
    # Get query_result and language_info from the most recent result message
    query_result = {}
    language_info = {}
    
    # Find the most recent message with query results
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage) and "query_result" in msg.additional_kwargs:
            query_result = msg.additional_kwargs.get("query_result", {})
            language_info = msg.additional_kwargs.get("language_info", {})
            break
    
    # Create and run the response formatting agent
    formatter_agent = ResponseFormattingAgent(verbose=config.agent.verbose)
    formatter_input = AgentInput(
        query=original_query,
        context={**query_result, "language_info": language_info}
    )
    
    # The formatter agent will handle translation back to the original language
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
