import json
from typing import Dict, Any

from app.mongodb.client import db
from app.agents.tools.base.tool import BaseTool, ToolResponse
from app.agents.utils.serialization_utils import mongodb_json_dumps, serialize_mongodb_doc


class ListCollectionsTool(BaseTool):
    """Tool to list all available collections in the database."""
    
    @property
    def name(self) -> str:
        return "list_collections"
    
    @property
    def description(self) -> str:
        return "List all available collections in the database"
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {}
        }
    
    async def execute(self, params: Dict[str, Any]) -> ToolResponse:
        """
        List all collections in the connected MongoDB database.
        
        Args:
            params: Empty dictionary (no parameters needed)
            
        Returns:
            ToolResponse with the list of collections
        """
        try:
            # Import db directly from the module
            from app.mongodb.client import db
            
            if db is None:
                return ToolResponse(
                    content=[{
                        "type": "text",
                        "text": "Error: MongoDB database connection is not initialized."
                    }],
                    is_error=True
                )
            
            # Get collections from the database
            collections = list(db.list_collection_names())
            
            # Format the collections into a simpler structure
            formatted_collections = [
                {
                    "name": collection
                }
                for collection in collections
            ]
            
            # Create the response using our custom serialization
            return ToolResponse(
                content=[{
                    "type": "text",
                    "text": mongodb_json_dumps(formatted_collections)
                }],
                is_error=False
            )
            
        except Exception as error:
            print(f"List collections error: {str(error)}")
            return self.handle_error(error)
