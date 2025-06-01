from typing import Dict, Any, List, Optional
from datetime import datetime

from mongodb.client import MongoDBClient
from agents.tools.base.tool import BaseTool, ToolResponse, ChatBotError, ErrorCode
from agents.utils.serialization_utils import serialize_mongodb_doc


class CreatePatientProfileTool(BaseTool):
    """Tool to create a new patient profile in the patients collection."""

    def __init__(self, mongodb_client: MongoDBClient):
        self.mongodb_client = mongodb_client
    
    @property
    def name(self) -> str:
        return "create_patient_profile"
    
    @property
    def description(self) -> str:
        return "Create a new patient profile with basic information and medical data."
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Patient's full name"
                },
                "age": {
                    "type": "integer",
                    "description": "Patient's age in years",
                    "minimum": 0,
                    "maximum": 150
                },
                "gender": {
                    "type": "string",
                    "description": "Patient's gender (Male/Female/Other)"
                },
                "symptoms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of symptoms",
                    "default": []
                },
                "medical_history": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of medical history items",
                    "default": []
                },
                "medications": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of current medications",
                    "default": []
                },
                "additional_info": {
                    "type": "object",
                    "description": "Additional medical information as key-value pairs",
                    "default": {}
                },
                "chat_history": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Chat history with medical expert",
                    "default": []
                }
            },
            "required": ["name"]
        }

    async def execute(self, params: Dict[str, Any]) -> ToolResponse:
        """
        Create a new patient profile.
        
        Args:
            params: Dictionary containing patient information
                
        Returns:
            ToolResponse with the created patient document
        """
        try:
            # Validate required fields
            name = params.get("name")
            if not name or not isinstance(name, str):
                raise ChatBotError(
                    ErrorCode.InvalidRequest,
                    "Patient name is required and must be a string"
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

            # Create patient document based on patient_template structure
            patient_document = {
                "name": name.strip(),
                "age": params.get("age"),
                "gender": params.get("gender", ""),
                "symptoms": params.get("symptoms", []),
                "medical_history": params.get("medical_history", []),
                "medications": params.get("medications", []),
                "additional_info": params.get("additional_info", {}),
                "chat_history": params.get("chat_history", []),
                "timestamp": datetime.now(),
                "completion_notified": False,
                "qa_pairs_count": 0,
                "extraction_performed": False
            }
            
            # Insert into patients collection
            collection = self.mongodb_client.db["patients"]
            result = collection.insert_one(patient_document)
            
            # Get the inserted document
            inserted_patient = collection.find_one({"_id": result.inserted_id})
            
            # Serialize the document to handle MongoDB types
            serialized_patient = serialize_mongodb_doc(inserted_patient)
            
            # Format response data
            response_data = {
                "patient": serialized_patient,
                "metadata": {
                    "patient_id": str(result.inserted_id),
                    "collection": "patients",
                    "operation": "create",
                    "timestamp": datetime.now().isoformat()
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
                    "operation": "create_patient_profile",
                    "patient_id": str(result.inserted_id),
                    "patient_name": name
                }
            )
            
        except ChatBotError:
            # Re-raise custom errors
            raise
        except Exception as e:
            # Handle unexpected errors
            return self.handle_error(e) 