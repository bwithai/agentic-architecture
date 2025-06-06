"""
Agent State Management

This module defines the state for the agent graph, which tracks information
as it flows through the different agents in the system.
"""

from typing import Dict, Any, List, Optional, TypedDict
from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """
    State model for the agent graph.
    
    This tracks information as it flows through the agents and maintains
    the ongoing context of the conversation.
    """
    
    # The user's original query
    query: str = Field(default="")
    
    # The type of query (general, product_availability, etc.)
    query_type: str = Field(default="general")
    
    # The structured MongoDB query derived from the natural language
    mongodb_query: Optional[Dict[str, Any]] = Field(default=None)
    
    # Results from the MongoDB query
    query_results: Optional[Dict[str, Any]] = Field(default=None)
    
    # Product-specific information
    product_id: Optional[str] = Field(default=None)
    product_availability: Optional[Dict[str, Any]] = Field(default=None)
    
    # Escalation information
    escalation_status: Optional[Dict[str, Any]] = Field(default=None)
    
    # Final response to present to the user
    response: str = Field(default="")
    
    # Overall status of the processing
    status: str = Field(default="pending")
    
    # Any error information
    error: Optional[str] = Field(default=None)
    
    # Execution history for debugging/logging
    execution_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    def add_to_history(self, agent_name: str, action: str, data: Any = None) -> None:
        """
        Add an entry to the execution history.
        
        Args:
            agent_name (str): Name of the agent
            action (str): Action taken by the agent
            data (Any): Any relevant data
        """
        self.execution_history.append({
            "agent": agent_name,
            "action": action,
            "data": data
        })
    
    def is_successful(self) -> bool:
        """
        Check if the state indicates successful processing.
        
        Returns:
            bool: True if successful, False otherwise
        """
        return self.status == "success" and self.error is None 