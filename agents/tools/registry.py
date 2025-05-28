from typing import Dict, List, Optional, Any
import asyncio
from langchain_core.tools import Tool
from mongodb.client import MongoDBClient

from agents.utils.serialization_utils import serialize_mongodb_doc, mongodb_json_dumps
from agents.tools.base.tool import BaseTool, ChatBotError, ErrorCode
from agents.tools.collection.list_collections import ListCollectionsTool
from agents.tools.documents.delete_one import DeleteOneTool
from agents.tools.documents.find import FindTool
from agents.tools.documents.insert_one import InsertOneTool
from agents.tools.documents.update_one import UpdateOneTool
from agents.tools.documents.count import CountTool
from agents.tools.indexes.create_index import CreateIndexTool
from agents.tools.indexes.drop_index import DropIndexTool
from agents.tools.indexes.list_indexes import ListIndexesTool
from agents.tools.patient.create_patient_profile import CreatePatientProfileTool


class ToolRegistry:
    """Registry for managing and accessing MongoDB tools."""
    
    def __init__(self, mongodb_client: MongoDBClient):
        """Initialize the tool registry with all available tools."""
        self._tools: Dict[str, BaseTool] = {}
        self.mongodb_client = mongodb_client
        
        # Register all available tools
        self.register_tool(ListCollectionsTool())
        self.register_tool(FindTool(mongodb_client))
        self.register_tool(InsertOneTool(mongodb_client))
        self.register_tool(UpdateOneTool(mongodb_client))
        self.register_tool(DeleteOneTool(mongodb_client))
        self.register_tool(CountTool(mongodb_client))
        self.register_tool(CreateIndexTool())
        self.register_tool(DropIndexTool())
        self.register_tool(ListIndexesTool())
        self.register_tool(CreatePatientProfileTool(mongodb_client))
    
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
    
    def _create_langchain_tools(self):
        """Create LangChain-compatible tools from MongoDB tools."""
        langchain_tools = []
        mongodb_tools = self.get_all_tools()
        
        for mongo_tool in mongodb_tools:
            # Create a function that will execute this tool
            async def run_tool(params: Dict[str, Any], tool=mongo_tool):
                # Set the global db reference
                from mongodb import client as mongodb_client_module
                if mongodb_client_module.db is None:
                    mongodb_client_module.db = self.mongodb_client.db
                
                # Execute the tool
                try:
                    # Ensure any MongoDB special types in tool arguments are serialized
                    serialized_params = serialize_mongodb_doc(params)
                    
                    # Call the MongoDB tool with serialized parameters
                    result = await tool.execute(serialized_params)
                    
                    # Extract the result text
                    if result and result.content and len(result.content) > 0:
                        return result.content[0].get("text", "")
                    else:
                        return "Operation completed but returned no content."
                except Exception as e:
                    return f"Error executing {tool.name}: {str(e)}"
            
            # Create a synchronous version to use with LangChain
            def sync_run_tool(params: Dict[str, Any], _run_tool=run_tool):
                # Get or create an event loop
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Already in an event loop, use asyncio.run_coroutine_threadsafe
                        import threading
                        import concurrent.futures
                        
                        # Create a new loop in a new thread
                        executor = concurrent.futures.ThreadPoolExecutor()
                        future = executor.submit(asyncio.run, _run_tool(params))
                        return future.result()
                    else:
                        # No running event loop, use run_until_complete
                        return loop.run_until_complete(_run_tool(params))
                except RuntimeError:
                    # No event loop exists, create one
                    return asyncio.run(_run_tool(params))
            
            # Create a named function for each tool
            func_name = f"run_{mongo_tool.name.replace('-', '_')}"
            tool_func = sync_run_tool
            tool_func.__name__ = func_name
            
            # Get properties and schema definitions
            properties = {}
            required = []
            
            if mongo_tool.input_schema and mongo_tool.input_schema.get("properties"):
                import copy
                # Use the entire properties section from the tool's schema
                properties = copy.deepcopy(mongo_tool.input_schema.get("properties", {}))
                required = mongo_tool.input_schema.get("required", [])
            
            # Create a valid schema with type:object
            schema = {
                "type": "object",
                "properties": properties,
                "required": required
            }
            
            # Handle empty properties case - provide a dummy property if needed
            if not properties:
                schema["properties"] = {
                    "dummy": {
                        "type": "string",
                        "description": "Placeholder parameter (not used)"
                    }
                }
            
            # Create a Tool object
            tool = Tool(
                name=mongo_tool.name,
                description=mongo_tool.description,
                func=tool_func,
                args_schema=schema
            )
            
            langchain_tools.append(tool)
        
        return langchain_tools
