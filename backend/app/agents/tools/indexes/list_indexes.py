import json
from typing import Dict, Any

from app.mongodb.client import db
from app.agents.tools.base.tool import BaseTool, ToolResponse


class ListIndexesTool(BaseTool):
    """Tool to list all indexes for a MongoDB collection."""
    
    @property
    def name(self) -> str:
        return "indexes"
    
    @property
    def description(self) -> str:
        return "List indexes for a collection"
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": "Name of the collection"
                }
            },
            "required": ["collection"]
        }
    
    async def execute(self, params: Dict[str, Any]) -> ToolResponse:
        """
        List all indexes for a MongoDB collection.
        
        Args:
            params: Dictionary containing:
                - collection: Name of the collection
                
        Returns:
            ToolResponse with the list of indexes
        """
        try:
            # Validate collection name
            collection = self.validate_collection(params.get("collection"))
            
            # Get MongoDB db directly from the module
            from app.mongodb.client import db
            
            if db is None:
                return ToolResponse(
                    content=[{
                        "type": "text",
                        "text": "Error: MongoDB database connection is not initialized."
                    }],
                    is_error=True
                )
            
            # Retrieve indexes
            indexes = list(db[collection].list_indexes())
            
            # Convert MongoDB cursor objects to serializable dictionaries
            serializable_indexes = json.loads(json.dumps(indexes, default=str))
            
            # Return success response
            return ToolResponse(
                content=[{
                    "type": "text",
                    "text": json.dumps(serializable_indexes, indent=2)
                }],
                is_error=False
            )
            
        except Exception as error:
            print(f"List indexes error: {str(error)}")
            return self.handle_error(error)
