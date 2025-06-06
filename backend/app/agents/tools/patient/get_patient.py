from typing import Dict, Any
from bson import ObjectId
from bson.errors import InvalidId

from app.mongodb.client import MongoDBClient
from app.agents.tools.base.tool import BaseTool, ToolResponse, ChatBotError, ErrorCode
from app.agents.utils.serialization_utils import serialize_mongodb_doc


class GetPatientTool(BaseTool):
    """Tool to retrieve a patient by MongoDB ObjectId."""

    def __init__(self, mongodb_client: MongoDBClient):
        self.mongodb_client = mongodb_client
    
    @property
    def name(self) -> str:
        return "get_patient"
    
    @property
    def description(self) -> str:
        return "Retrieve a patient document from the patients collection by MongoDB ObjectId."
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "MongoDB ObjectId of the patient to retrieve (e.g., '6838a9a09e7ca8ddfcc6c1de')"
                }
            },
            "required": ["patient_id"]
        }

    async def execute(self, params: Dict[str, Any]) -> ToolResponse:
        """
        Retrieve a patient document by ObjectId.
        
        Args:
            params: Dictionary containing:
                - patient_id: MongoDB ObjectId as string
                
        Returns:
            ToolResponse with the patient document or error message
        """
        try:
            # Get and validate patient_id parameter
            patient_id_str = params.get("patient_id")
            if not patient_id_str:
                raise ChatBotError(
                    ErrorCode.InvalidRequest,
                    "patient_id is required"
                )
            
            # Convert string to ObjectId
            try:
                patient_id = ObjectId(patient_id_str)
            except InvalidId:
                raise ChatBotError(
                    ErrorCode.InvalidRequest,
                    f"Invalid ObjectId format: {patient_id_str}"
                )
            
            # Check MongoDB connection
            if self.mongodb_client.db is None:
                return ToolResponse(
                    content=[{
                        "type": "text",
                        "text": "Error: MongoDB database connection is not initialized."
                    }],
                    is_error=True
                )

            # Query the patients collection
            collection = self.mongodb_client.db["patients"]
            patient_doc = collection.find_one({"_id": patient_id})
            
            if not patient_doc:
                return ToolResponse(
                    content=[{
                        "type": "text",
                        "text": f"Patient not found with ID: {patient_id_str}"
                    }],
                    is_error=False
                )
            
            # Serialize the document to handle MongoDB types
            serialized_patient = serialize_mongodb_doc(patient_doc)
            
            # Format response data
            response_data = {
                "patient": serialized_patient,
                "metadata": {
                    "patient_id": patient_id_str,
                    "collection": "patients",
                    "found": True
                }
            }
            
            # Convert to JSON string for response
            import json
            response_json = json.dumps(response_data, indent=2, ensure_ascii=False)
            
            return ToolResponse(
                content=[{
                    "type": "text",
                    "text": response_json
                }],
                is_error=False,
                meta={
                    "operation": "get_patient",
                    "patient_id": patient_id_str,
                    "found": True
                }
            )
            
        except ChatBotError:
            # Re-raise custom errors
            raise
        except Exception as e:
            # Handle unexpected errors
            return self.handle_error(e) 