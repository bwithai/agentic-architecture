from typing import Dict, List, Optional, Any

from agents.tools.base.tool import BaseTool, ChatBotError, ErrorCode
from agents.tools.collection.list_collections import ListCollectionsTool
from agents.tools.documents.delete_one import DeleteOneTool
from agents.tools.documents.find import FindTool
from agents.tools.documents.insert_one import InsertOneTool
from agents.tools.documents.update_one import UpdateOneTool
from agents.tools.indexes.create_index import CreateIndexTool
from agents.tools.indexes.drop_index import DropIndexTool
from agents.tools.indexes.list_indexes import ListIndexesTool


class ToolRegistry:
    """Registry for managing and accessing MongoDB tools."""
    
    def __init__(self):
        """Initialize the tool registry with all available tools."""
        self._tools: Dict[str, BaseTool] = {}
        
        # Register all available tools
        self.register_tool(ListCollectionsTool())
        self.register_tool(FindTool())
        self.register_tool(InsertOneTool())
        self.register_tool(UpdateOneTool())
        self.register_tool(DeleteOneTool())
        self.register_tool(CreateIndexTool())
        self.register_tool(DropIndexTool())
        self.register_tool(ListIndexesTool())
    
    def register_tool(self, tool: BaseTool) -> None:
        """
        Register a tool with the registry.
        
        Args:
            tool: The tool to register
        """
        self._tools[tool.name] = tool
    
    def get_tool(self, name: str) -> BaseTool:
        """
        Get a tool by name.
        
        Args:
            name: Name of the tool to retrieve
            
        Returns:
            The requested tool
            
        Raises:
            ChatBotError: If the tool doesn't exist
        """
        tool = self._tools.get(name)
        if not tool:
            raise ChatBotError(
                ErrorCode.InvalidRequest, 
                f"Unknown tool: {name}"
            )
        return tool
    
    def get_all_tools(self) -> List[BaseTool]:
        """
        Get all registered tools.
        
        Returns:
            List of all registered tools
        """
        return list(self._tools.values())
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Get schemas for all tools in a format suitable for OpenAI API.
        
        Returns:
            List of tool schemas
        """
        schemas = []
        for tool in self.get_all_tools():
            input_schema = tool.input_schema
            schema = {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": input_schema.get("properties", {}),
                }
            }
            
            # Add required fields if present
            if "required" in input_schema:
                schema["parameters"]["required"] = input_schema["required"]
                
            schemas.append(schema)
            
        return schemas
