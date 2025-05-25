import json
from typing import Dict, Any

from mongodb.client import MongoDBClient
from agents.tools.base.tool import BaseTool, ToolResponse
from agents.utils.serialization_utils import mongodb_json_dumps, serialize_mongodb_doc


class InsertOneTool(BaseTool):
    """Tool to insert a single document into a MongoDB collection."""

    def __init__(self, mongodb_client: MongoDBClient):
        self.mongodb_client = mongodb_client
    
    @property
    def name(self) -> str:
        return "insert_one"
    
    @property
    def description(self) -> str:
        return "Insert a single document into a collection"
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": "Name of the collection to insert into"
                },
                "document": {
                    "type": "object",
                    "description": "Document to insert (must be a valid JSON object)"
                }
            },
            "required": ["collection", "document"]
        }
    
    async def execute(self, params: Dict[str, Any]) -> ToolResponse:
        """
        Insert a document into a collection.
        
        Args:
            params:
                - collection: Name of the collection
                - document: Document to insert
                
        Returns:
            ToolResponse with the insertion result
        """
        try:
            # Validate parameters
            collection = self.validate_collection(params.get("collection"))
            document = self.validate_object(params.get("document"), "Document")
            
            # Get MongoDB db directly from the module
            if self.mongodb_client.db is None:
                return ToolResponse(
                    content=[{
                        "type": "text",
                        "text": "Error: MongoDB database connection is not initialized."
                    }],
                    is_error=True
                )
            
            # Perform insertion
            result = self.mongodb_client.db[collection].insert_one(document)
            
            # Return success response with serialized ObjectId
            response_data = {
                "acknowledged": result.acknowledged,
                "insertedId": str(result.inserted_id)  # Convert ObjectId to string
            }
            
            return ToolResponse(
                content=[{
                    "type": "text",
                    "text": mongodb_json_dumps(response_data)
                }],
                is_error=False
            )
            
        except Exception as error:
            print(f"Insert error: {str(error)}")
            return self.handle_error(error)
