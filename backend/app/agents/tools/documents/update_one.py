import json
from typing import Dict, Any

from app.mongodb.client import MongoDBClient
from app.agents.tools.base.tool import BaseTool, ToolResponse


class UpdateOneTool(BaseTool):
    """Tool to update a single document in a MongoDB collection."""

    def __init__(self, mongodb_client: MongoDBClient):
        self.mongodb_client = mongodb_client
    
    @property
    def name(self) -> str:
        return "update_one"
    
    @property
    def description(self) -> str:
        return "Update a single document in a collection"
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": "Name of the collection"
                },
                "filter": {
                    "type": "object",
                    "description": "Filter to identify document"
                },
                "update": {
                    "type": "object",
                    "description": "Update operations to apply"
                }
            },
            "required": ["collection", "filter", "update"]
        }
    
    async def execute(self, params: Dict[str, Any]) -> ToolResponse:
        """
        Update a single document in a MongoDB collection.
        
        Args:
            params: Dictionary containing:
                - collection: Name of the collection
                - filter: Filter to identify document
                - update: Update operations to apply
                
        Returns:
            ToolResponse with the update result
        """
        try:
            # Validate parameters
            collection = self.validate_collection(params.get("collection"))
            filter_obj = self.validate_object(params.get("filter"), "Filter")
            update = self.validate_object(params.get("update"), "Update")
            
            # Get MongoDB db directly from the module
            if MongoDBClient.db is None:
                return ToolResponse(
                    content=[{
                        "type": "text",
                        "text": "Error: MongoDB database connection is not initialized."
                    }],
                    is_error=True
                )
            
            # Perform update
            result = MongoDBClient.db[collection].update_one(filter_obj, update)
            
            # Prepare the response
            response_data = {
                "matched": result.matched_count,
                "modified": result.modified_count,
                "upsertedId": str(result.upserted_id) if result.upserted_id else None
            }
            
            # Return success response
            return ToolResponse(
                content=[{
                    "type": "text",
                    "text": json.dumps(response_data, indent=2)
                }],
                is_error=False
            )
            
        except Exception as error:
            print(f"Update error: {str(error)}")
            return self.handle_error(error)
