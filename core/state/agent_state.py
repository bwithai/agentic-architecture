"""
Agent State Management

This module defines the state for the agent graph, which tracks information
as it flows through the different agents in the system.
"""

from typing import Dict, Any, List, Optional, TypedDict
from pydantic import BaseModel, Field
from datetime import datetime


class UserLanguagePreference(BaseModel):
    """User language preference model"""
    language_code: str = Field(default="en", description="ISO language code")
    language_name: str = Field(default="English", description="Human-readable language name")
    confidence: float = Field(default=1.0, description="Confidence score for language detection")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="When this preference was last updated")
    
    def update(self, language_code: str, language_name: str, confidence: float) -> None:
        """Update the language preference"""
        self.language_code = language_code
        self.language_name = language_name
        self.confidence = confidence
        self.last_updated = datetime.utcnow()


class AgentState(BaseModel):
    """
    State model for the agent graph.
    
    This tracks information as it flows through the agents and maintains
    the ongoing context of the conversation.
    """
    
    # User information
    user_id: Optional[str] = Field(None, description="Unique identifier for the user")
    user_name: Optional[str] = Field(None, description="Name of the user")
    
    # Language preferences
    language_preference: UserLanguagePreference = Field(
        default_factory=UserLanguagePreference,
        description="User language preference"
    )
    
    # Session information
    session_id: str = Field(..., description="Unique identifier for the session")
    session_start: datetime = Field(default_factory=datetime.utcnow, description="When the session started")
    last_interaction: datetime = Field(default_factory=datetime.utcnow, description="When the last interaction occurred")
    
    # Conversation history
    conversation_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of the conversation"
    )
    
    # Application state
    current_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current application state"
    )
    
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
    
    def update_language_preference(self, language_code: str, language_name: str, confidence: float) -> None:
        """Update the user's language preference"""
        self.language_preference.update(language_code, language_name, confidence)
    
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
    
    def add_to_conversation_history(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role (str): Role of the message sender (user, assistant, system)
            content (str): Content of the message
        """
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow()
        })
        self.last_interaction = datetime.utcnow()
    
    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
    
    def get_recent_conversation_history(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get the n most recent messages from the conversation history.
        
        Args:
            n (int): Number of messages to return
            
        Returns:
            List[Dict[str, Any]]: List of recent messages
        """
        return self.conversation_history[-n:] if self.conversation_history else [] 