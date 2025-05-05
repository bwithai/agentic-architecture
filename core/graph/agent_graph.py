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
from config.config import config


# Define the node functions for the graph
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
        state.add_to_history("mongodb", "success", {
            "count": output.data.get("count"),
            "has_results": bool(output.data.get("result"))
        })
    else:
        state.status = "error"
        state.error = output.error
        state.add_to_history("mongodb", "error", output.error)
    
    return state


async def format_response(state: AgentState) -> AgentState:
    """
    Node function to format the response.
    
    Args:
        state (AgentState): The current state
        
    Returns:
        AgentState: The updated state
    """
    agent = ResponseFormattingAgent(verbose=config.agent.verbose)
    
    # Create a context with all necessary information
    context = {
        "mongodb_query": state.mongodb_query,
        "result": state.query_results.get("result", []) if state.query_results else [],
        "count": state.query_results.get("count", 0) if state.query_results else 0,
        "status": state.status,
        "error": state.error
    }
    
    agent_input = AgentInput(query=state.query, context=context)
    
    output = await agent.run(agent_input)
    
    # Update the state
    state.response = output.response
    state.status = "success" if state.error is None else "error"
    state.add_to_history("response_formatter", "complete", None)
    
    return state


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
    graph.add_node("understand_query", understand_query)
    graph.add_node("execute_query", execute_query)
    graph.add_node("format_response", format_response)
    
    # Add conditional edges for routing
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
    
    # Set the entry point
    graph.set_entry_point("understand_query")
    
    # Compile the graph
    return graph.compile() 