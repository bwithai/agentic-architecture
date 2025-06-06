import json
from typing import Dict, Any

from app.mongodb.client import MongoDBClient
from app.agents.tools.base.tool import BaseTool, ToolResponse, ChatBotError, ErrorCode


class DropIndexTool(BaseTool):
    """Tool to drop an index from a MongoDB collection."""
    
    @property
    def name(self) -> str:
        return "drop_index"
    
    @property
    def description(self) -> str:
        return "Drop an index from a collection"
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": "Name of the collection"
                },
                "indexName": {
                    "type": "string",
                    "description": "Name of the index to drop"
                }
            },
            "required": ["collection", "indexName"]
        }
    
    async def execute(self, params: Dict[str, Any]) -> ToolResponse:
        """
        Drop an index from a MongoDB collection.
        
        Args:
            params: Dictionary containing:
                - collection: Name of the collection
                - indexName: Name of the index to drop
                
        Returns:
            ToolResponse with the result of dropping the index
        """
        try:
            # Validate parameters
            collection = self.validate_collection(params.get("collection"))
            
            # Validate index name
            index_name = params.get("indexName")
            if not isinstance(index_name, str):
                raise ChatBotError(
                    ErrorCode.InvalidRequest,
                    "Index name must be a string"
                )
            
            # Get MongoDB client and drop the index
            result = MongoDBClient.db[collection].drop_index(index_name)
            
            # Return success response
            return ToolResponse(
                content=[{
                    "type": "text",
                    "text": json.dumps(result, indent=2)
                }],
                is_error=False
            )
            
        except Exception as error:
            return self.handle_error(error)
