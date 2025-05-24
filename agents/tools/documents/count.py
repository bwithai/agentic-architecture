import json
from typing import Dict, Any

from mongodb.client import db
from agents.tools.base.tool import BaseTool, ToolResponse


class CountTool(BaseTool):
    """Tool to count documents in a MongoDB collection."""
    
    @property
    def name(self) -> str:
        return "count"
    
    @property
    def description(self) -> str:
        return "Count documents in a collection using MongoDB query syntax"
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": "Name of the collection to count documents in"
                },
                "filter": {
                    "type": "object",
                    "description": "MongoDB query filter",
                    "default": {}
                }
            },
            "required": ["collection"]
        }
    
    async def execute(self, params: Dict[str, Any]) -> ToolResponse:
        """
        Count documents in a MongoDB collection.
        
        Args:
            params: Dictionary containing:
                - collection: Name of the collection
                - filter: (Optional) Query filter
                
        Returns:
            ToolResponse with the count result
        """
        try:
            # Validate collection name
            collection = self.validate_collection(params.get("collection"))
            
            # Get optional parameters with defaults
            filter_obj = params.get("filter", {})
            
            # Get MongoDB db directly from the module
            from mongodb.client import db
            
            if db is None:
                return ToolResponse(
                    content=[{
                        "type": "text",
                        "text": "Error: MongoDB database connection is not initialized."
                    }],
                    is_error=True
                )
            
            # Execute count
            count = db[collection].count_documents(filter_obj)
            
            # Return success response
            return ToolResponse(
                content=[{
                    "type": "text",
                    "text": json.dumps({"count": count}, indent=2)
                }],
                is_error=False
            )
            
        except Exception as error:
            print(f"Count error: {str(error)}")
            return self.handle_error(error) 