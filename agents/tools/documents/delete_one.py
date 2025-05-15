import json
from typing import Dict, Any

from mongodb.client import db
from agents.tools.base.tool import BaseTool, ToolResponse


class DeleteOneTool(BaseTool):
    """Tool to delete a single document from a MongoDB collection."""
    
    @property
    def name(self) -> str:
        return "delete_one"
    
    @property
    def description(self) -> str:
        return "Delete a single document from a collection"
    
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
                }
            },
            "required": ["collection", "filter"]
        }
    
    async def execute(self, params: Dict[str, Any]) -> ToolResponse:
        """
        Delete a single document from a collection based on filter criteria.
        
        Args:
            params: Dictionary containing:
                - collection: Name of the collection
                - filter: Filter to identify document to delete
                
        Returns:
            ToolResponse with the deletion result
        """
        try:
            # Validate parameters
            collection = self.validate_collection(params.get("collection"))
            filter_obj = self.validate_object(params.get("filter"), "Filter")
            
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
            
            # Perform deletion
            result = db[collection].delete_one(filter_obj)
            
            # Return success response
            return ToolResponse(
                content=[{
                    "type": "text",
                    "text": json.dumps({"deleted": result.deleted_count}, indent=2)
                }],
                is_error=False
            )
            
        except Exception as error:
            print(f"Delete error: {str(error)}")
            return self.handle_error(error)
