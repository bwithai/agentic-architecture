from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional, TypedDict, Union


class ErrorCode(Enum):
    """Error codes similar to MCP error codes."""
    InvalidRequest = "invalid_request"
    InternalError = "internal_error"
    Unauthorized = "unauthorized"
    RateLimitExceeded = "rate_limit_exceeded"
    ServiceUnavailable = "service_unavailable"


class ChatBotError(Exception):
    """Custom error class for chat bot errors."""
    
    def __init__(self, code: ErrorCode, message: str):
        self.code = code
        self.message = message
        super().__init__(f"{code.value}: {message}")


class ContentItem(TypedDict):
    """Represents a single content item in a tool response."""
    type: str
    text: str


class ToolResponse:
    """Response format for tool execution."""
    
    def __init__(
        self, 
        content: List[ContentItem], 
        is_error: bool = False, 
        meta: Optional[Dict[str, Any]] = None
    ):
        self.content = content
        self.is_error = is_error
        self._meta = meta
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "content": self.content,
            "isError": self.is_error
        }
        if self._meta:
            result["_meta"] = self._meta
        return result


class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the tool."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does."""
        pass
    
    @property
    @abstractmethod
    def input_schema(self) -> Dict[str, Any]:
        """
        JSON Schema for the tool's input parameters.
        Should be a dict with structure:
        {
            "type": "object",
            "properties": {...},
            "required": [...]
        }
        """
        pass
    
    @abstractmethod
    async def execute(self, params: Dict[str, Any]) -> ToolResponse:
        """
        Execute the tool with the given parameters.
        
        Args:
            params: Dictionary of parameters for the tool
            
        Returns:
            A ToolResponse object
        """
        pass
    
    def validate_collection(self, collection: Any) -> str:
        """
        Validate that the given value is a valid collection name.
        
        Args:
            collection: Value to validate
            
        Returns:
            Validated collection name as string
            
        Raises:
            ChatBotError: If validation fails
        """
        if not isinstance(collection, str):
            raise ChatBotError(
                ErrorCode.InvalidRequest,
                f"Collection name must be a string, got {type(collection).__name__}"
            )
        return collection
    
    def validate_object(self, value: Any, name: str) -> Dict[str, Any]:
        """
        Validate that the given value is a dictionary.
        
        Args:
            value: Value to validate
            name: Name of the value for error messages
            
        Returns:
            Validated dictionary
            
        Raises:
            ChatBotError: If validation fails
        """
        if not value or not isinstance(value, dict):
            raise ChatBotError(
                ErrorCode.InvalidRequest, 
                f"{name} must be an object"
            )
        return value
    
    def handle_error(self, error: Exception) -> ToolResponse:
        """
        Convert an exception to a ToolResponse.
        
        Args:
            error: Exception to convert
            
        Returns:
            ToolResponse representing the error
        """
        return ToolResponse(
            content=[{
                "type": "text",
                "text": str(error)
            }],
            is_error=True
        )
