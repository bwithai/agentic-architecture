"""
Agent Graph

This module defines the LangGraph implementation that connects all the agents
together into a coherent workflow.
"""

import asyncio
from typing import Dict, Any, List, Tuple, Annotated, TypedDict
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

from core.state.agent_state import AgentState
from agents.base.base_agent import AgentInput, AgentOutput
from agents.specialized.query_understanding_agent import QueryUnderstandingAgent
from agents.specialized.mongodb_agent import MongoDBAgent
from agents.specialized.response_formatting_agent import ResponseFormattingAgent
from agents.specialized.query_classifier_agent import QueryClassifierAgent
from agents.specialized.product_availability_agent import ProductAvailabilityAgent
from config.config import config


# Define the node functions for the graph
async def classify_query(state: AgentState) -> AgentState:
    """
    Node function to classify the type of query.
    
    Args:
        state (AgentState): The current state
        
    Returns:
        AgentState: The updated state
    """
    agent = QueryClassifierAgent(verbose=config.agent.verbose)
    agent_input = AgentInput(query=state.query)
    
    output = await agent.run(agent_input)
    
    # Update the state
    if output.status == "success":
        classification = output.data.get("classification", {})
        state.query_type = classification.get("query_type", "general_db_query")
        state.product_id = classification.get("product_id")
        state.add_to_history("query_classifier", "success", classification)
    else:
        state.query_type = "general_db_query"  # Default to general query on failure
        state.add_to_history("query_classifier", "error", output.error)
    
    return state


async def understand_query(state: AgentState) -> AgentState:
    """
    Node function to understand the user's query.
    
    Args:
        state (AgentState): The current state
        
    Returns:
        AgentState: The updated state
    """
    agent = QueryUnderstandingAgent(verbose=config.agent.verbose)
    agent_input = AgentInput(query=state.query)
    
    output = await agent.run(agent_input)
    
    # Update the state
    if output.status == "success":
        state.mongodb_query = output.data.get("mongodb_query")
        state.add_to_history("query_understanding", "success", state.mongodb_query)
    else:
        state.status = "error"
        state.error = output.error
        state.add_to_history("query_understanding", "error", output.error)
    
    return state


async def execute_query(state: AgentState) -> AgentState:
    """
    Node function to execute the MongoDB query.
    
    Args:
        state (AgentState): The current state
        
    Returns:
        AgentState: The updated state
    """
    agent = MongoDBAgent(verbose=config.agent.verbose)
    agent_input = AgentInput(
        query=state.query, 
        context={"mongodb_query": state.mongodb_query}
    )
    
    output = await agent.run(agent_input)
    
    # Update the state
    if output.status == "success":
        state.query_results = output.data
        
        # Check if this is a multi-collection result
        if "multi_collection_results" in output.data:
            state.add_to_history("mongodb", "success", {
                "multi_collection": True,
                "count": output.data.get("count"),
                "collections": list(output.data.get("multi_collection_results", {}).keys())
            })
        else:
            state.add_to_history("mongodb", "success", {
                "count": output.data.get("count"),
                "has_results": bool(output.data.get("result"))
            })
    else:
        state.status = "error"
        state.error = output.error
        state.add_to_history("mongodb", "error", output.error)
    
    return state


async def check_product_availability(state: AgentState) -> AgentState:
    """
    Node function to check product availability.
    
    Args:
        state (AgentState): The current state
        
    Returns:
        AgentState: The updated state
    """
    agent = ProductAvailabilityAgent(verbose=config.agent.verbose)
    agent_input = AgentInput(
        query=state.query,
        context={"product_id": state.product_id}
    )
    
    output = await agent.run(agent_input)
    
    # Update the state based on the result
    if output.status == "escalated":
        state.status = "escalated"
        state.escalation_status = output.data.get("escalation")
        state.response = output.response
        state.add_to_history("product_availability", "escalated", output.data)
    elif output.status == "success":
        state.product_availability = output.data.get("availability")
        state.response = output.response
        state.add_to_history("product_availability", "success", state.product_availability)
    else:
        state.status = "error"
        state.error = output.error
        state.add_to_history("product_availability", "error", output.error)
    
    return state


async def format_response(state: AgentState) -> AgentState:
    """
    Node function to format the response.
    
    Args:
        state (AgentState): The current state
        
    Returns:
        AgentState: The updated state
    """
    # Skip if we already have a response (from product availability)
    if state.response and (state.status == "success" or state.status == "escalated"):
        state.add_to_history("response_formatter", "skipped", "Response already provided")
        return state
    
    agent = ResponseFormattingAgent(verbose=config.agent.verbose)
    
    # Create a context with all necessary information
    context = {
        "mongodb_query": state.mongodb_query,
        "status": state.status,
        "error": state.error
    }
    
    # Add the appropriate result data based on query results type
    if state.query_results:
        if "multi_collection_results" in state.query_results:
            context["multi_collection_results"] = state.query_results.get("multi_collection_results", {})
            context["count"] = state.query_results.get("count", 0)
        else:
            context["result"] = state.query_results.get("result", [])
            context["count"] = state.query_results.get("count", 0)
    
    agent_input = AgentInput(query=state.query, context=context)
    
    output = await agent.run(agent_input)
    
    # Update the state
    state.response = output.response
    state.status = "success" if state.error is None else "error"
    state.add_to_history("response_formatter", "complete", None)
    
    return state


def route_by_query_type(state: AgentState) -> str:
    """
    Conditional node function to route based on query type.
    
    Args:
        state (AgentState): The current state
        
    Returns:
        str: Next node name
    """
    if state.query_type == "product_availability":
        return "product_availability"
    return "general_query"


def route_after_understanding(state: AgentState) -> str:
    """
    Decide next step after understanding the query.
    
    Args:
        state (AgentState): The current state
        
    Returns:
        str: Next node name
    """
    if state.status == "error":
        return "format_response"
    return "execute_query"


def route_after_execution(state: AgentState) -> str:
    """
    Decide next step after executing the query.
    
    Args:
        state (AgentState): The current state
        
    Returns:
        str: Next node name
    """
    # Always go to format_response after execute_query, regardless of success/failure
    return "format_response"


def create_agent_graph() -> StateGraph:
    """
    Create and configure the agent graph.
    
    Returns:
        StateGraph: The configured agent graph
    """
    # Create a new graph
    graph = StateGraph(AgentState)
    
    # Add nodes to the graph
    graph.add_node("classify_query", classify_query)
    graph.add_node("understand_query", understand_query)
    graph.add_node("execute_query", execute_query)
    graph.add_node("format_response", format_response)
    graph.add_node("check_product_availability", check_product_availability)
    
    # Entry point goes to query classifier
    graph.set_entry_point("classify_query")
    
    # Route based on query classification
    graph.add_conditional_edges(
        "classify_query",
        route_by_query_type,
        {
            "product_availability": "check_product_availability",
            "general_query": "understand_query"
        }
    )
    
    # Product availability flow
    graph.add_edge("check_product_availability", END)
    
    # General query flow
    graph.add_conditional_edges(
        "understand_query",
        route_after_understanding,
        {
            "format_response": "format_response",
            "execute_query": "execute_query"
        }
    )
    
    graph.add_conditional_edges(
        "execute_query",
        route_after_execution,
        {
            "format_response": "format_response"
        }
    )
    
    graph.add_edge("format_response", END)
    
    # Compile the graph
    return graph.compile() 