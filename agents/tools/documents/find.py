import json
from typing import Dict, Any, Optional, List

from mongodb.client import db
from agents.tools.base.tool import BaseTool, ToolResponse
from agents.utils.serialization_utils import mongodb_json_dumps, serialize_mongodb_doc


class FindTool(BaseTool):
    """Tool to query documents from a MongoDB collection."""
    
    @property
    def name(self) -> str:
        return "find"
    
    @property
    def description(self) -> str:
        return "Query documents in a collection using MongoDB query syntax"
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": "Name of the collection to query"
                },
                "filter": {
                    "type": "object",
                    "description": "MongoDB query filter",
                    "default": {}
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum documents to return",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 1000
                },
                "projection": {
                    "type": "object",
                    "description": "Fields to include/exclude",
                    "default": {}
                }
            },
            "required": ["collection"]
        }
    
    async def execute(self, params: Dict[str, Any]) -> ToolResponse:
        """
        Query documents from a MongoDB collection.
        
        Args:
            params: Dictionary containing:
                - collection: Name of the collection
                - filter: (Optional) Query filter
                - limit: (Optional) Maximum documents to return
                - projection: (Optional) Fields to include/exclude
                
        Returns:
            ToolResponse with the query results
        """
        try:
            # Validate collection name
            collection = self.validate_collection(params.get("collection"))
            
            # Get optional parameters with defaults
            filter_obj = params.get("filter", {})
            limit = min(params.get("limit", 10), 1000)
            projection = params.get("projection", {})
            
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
            
            # Execute query with parameters
            results = list(
                db[collection]
                .find(filter_obj, projection)
                .limit(limit)
            )
            
            # Use our custom serialization utility for MongoDB types
            serialized_results = serialize_mongodb_doc(results)
            
            # Return success response
            return ToolResponse(
                content=[{
                    "type": "text",
                    "text": mongodb_json_dumps(serialized_results)
                }],
                is_error=False
            )
            
        except Exception as error:
            print(f"Find error: {str(error)}")
            return self.handle_error(error)
