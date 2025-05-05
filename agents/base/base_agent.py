"""
Base agent module that provides the foundation for all specialized agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class AgentInput(BaseModel):
    """Base model for agent inputs."""
    query: str = Field(description="The user query or instruction")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context information")


class AgentOutput(BaseModel):
    """Base model for agent outputs."""
    response: str = Field(description="The agent's response to the query")
    data: Dict[str, Any] = Field(default_factory=dict, description="Any data produced by the agent")
    status: str = Field(default="success", description="Status of the agent's execution")
    error: Optional[str] = Field(default=None, description="Error message if any")


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    All specialized agents should inherit from this class and implement the required methods.
    """
    def __init__(self, name: str = "base_agent", verbose: bool = False):
        """
        Initialize the base agent.
        
        Args:
            name (str): Name of the agent
            verbose (bool): Whether to output verbose logs
        """
        self.name = name
        self.verbose = verbose
    
    @abstractmethod
    async def run(self, inputs: AgentInput) -> AgentOutput:
        """
        Main method to run the agent.
        
        Args:
            inputs (AgentInput): The inputs for the agent
            
        Returns:
            AgentOutput: The outputs from the agent
        """
        pass
    
    def log(self, message: str) -> None:
        """
        Log a message if verbose mode is enabled.
        
        Args:
            message (str): The message to log
        """
        if self.verbose:
            print(f"[{self.name}] {message}")
            
    @abstractmethod
    def get_description(self) -> str:
        """
        Get a description of what this agent does.
        
        Returns:
            str: Description of the agent
        """
        pass 