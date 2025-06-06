import json
from typing import Dict, Any, Union

from app.mongodb.client import MongoDBClient
from app.agents.tools.base.tool import BaseTool, ToolResponse


class CreateIndexTool(BaseTool):
    """Tool to create a new index on a MongoDB collection."""
    
    @property
    def name(self) -> str:
        return "create_index"
    
    @property
    def description(self) -> str:
        return "Create a new index on a collection"
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": "Name of the collection"
                },
                "indexSpec": {
                    "type": "object",
                    "description": "Index specification (e.g., { field: 1 } for ascending index)"
                }
            },
            "required": ["collection", "indexSpec"]
        }
    
    async def execute(self, params: Dict[str, Any]) -> ToolResponse:
        """
        Create a new index on a MongoDB collection.
        
        Args:
            params: Dictionary containing:
                - collection: Name of the collection
                - indexSpec: Index specification object (field name to direction mapping)
                
        Returns:
            ToolResponse with the created index name
        """
        try:
            # Validate parameters
            collection = self.validate_collection(params.get("collection"))
            
            # Get index specification, ensuring it's a dictionary
            index_spec = params.get("indexSpec", {})
            if not isinstance(index_spec, dict):
                raise ValueError("indexSpec must be an object")
            
            # Get MongoDB client and create index
            index_name = MongoDBClient.db[collection].create_index(
                list(index_spec.items())  # Convert dict to list of tuples for pymongo
            )
            
            # Return success response
            return ToolResponse(
                content=[{
                    "type": "text",
                    "text": json.dumps({"indexName": index_name}, indent=2)
                }],
                is_error=False
            )
            
        except Exception as error:
            return self.handle_error(error)
