import json
from datetime import datetime
from typing import Dict, Any

from mongodb.client import MongoDBClient
from agents.tools.base.tool import BaseTool, ToolResponse
from agents.utils.serialization_utils import mongodb_json_dumps


class CreatePatientProfileTool(BaseTool):
    """Tool to create a patient profile and insert it into MongoDB."""

    def __init__(self, mongodb_client: MongoDBClient):
        self.mongodb_client = mongodb_client
    
    @property
    def name(self) -> str:
        return "create_patient_profile"
    
    @property
    def description(self) -> str:
        return "Create a new patient profile with collected information and insert it into the patients collection"
    
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
                    "description": "Patient's age"
                },
                "gender": {
                    "type": "string",
                    "description": "Patient's gender (male/female/other)"
                },
                "contact_info": {
                    "type": "object",
                    "description": "Contact information including phone and email",
                    "properties": {
                        "phone": {"type": "string"},
                        "email": {"type": "string"}
                    }
                },
                "symptoms": {
                    "type": "array",
                    "description": "List of current symptoms",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string", "description": "Description of the symptom"},
                            "location": {"type": "string", "description": "Location of the symptom (optional)"},
                            "duration": {"type": "string", "description": "How long the symptom has been present"},
                            "severity": {"type": "string", "description": "Severity level of the symptom"}
                        },
                        "required": ["description"]
                    }
                },
                "medical_history": {
                    "type": "array",
                    "description": "List of past medical conditions",
                    "items": {
                        "type": "string",
                        "description": "A medical condition from patient's history"
                    }
                },
                "allergies": {
                    "type": "array",
                    "description": "List of known allergies",
                    "items": {
                        "type": "string",
                        "description": "An allergy the patient has"
                    }
                },
                "current_medications": {
                    "type": "array",
                    "description": "List of current medications",
                    "items": {
                        "type": "string",
                        "description": "A medication the patient is currently taking"
                    }
                },
                "chat_history": {
                    "type": "array",
                    "description": "Chat conversation history",
                    "items": {
                        "type": "object",
                        "properties": {
                            "user": {"type": "string", "description": "User message"},
                            "bot": {"type": "string", "description": "Bot response"}
                        },
                        "required": ["user", "bot"]
                    }
                }
            },
            "required": ["name", "age", "gender"]
        }
    
    async def execute(self, params: Dict[str, Any]) -> ToolResponse:
        """
        Create a patient profile and insert it into MongoDB.
        
        Args:
            params: Patient information including name, age, gender, etc.
                
        Returns:
            ToolResponse with the insertion result
        """
        try:
            # Validate required parameters
            name = params.get("name")
            if not name or not isinstance(name, str):
                raise ValueError("Name is required and must be a string")
            
            age = params.get("age")
            if age is None or not isinstance(age, int):
                raise ValueError("Age is required and must be an integer")
            
            gender = params.get("gender")
            if not gender or not isinstance(gender, str):
                raise ValueError("Gender is required and must be a string")
            
            # Create the patient document with default structure
            patient_document = {
                "name": name,
                "age": age,
                "gender": gender.lower(),
                "contact_info": params.get("contact_info", {}),
                "symptoms": params.get("symptoms", []),
                "medical_history": params.get("medical_history", []),
                "allergies": params.get("allergies", []),
                "current_medications": params.get("current_medications", []),
                "diagnosis": "",
                "suggested_treatment": [],
                "vitals": {},
                "chat_history": params.get("chat_history", []),
                "timestamp": datetime.utcnow()
            }
            
            # Get MongoDB db directly from the module
            if self.mongodb_client.db is None:
                return ToolResponse(
                    content=[{
                        "type": "text",
                        "text": "Error: MongoDB database connection is not initialized."
                    }],
                    is_error=True
                )
            
            # Insert the patient profile into the patients collection
            result = self.mongodb_client.db["patients"].insert_one(patient_document)
            
            # Return success response with patient ID
            response_data = {
                "success": True,
                "message": f"Patient profile created successfully for {name}",
                "patient_id": str(result.inserted_id),
                "patient_name": name,
                "patient_age": age,
                "patient_gender": gender
            }
            
            return ToolResponse(
                content=[{
                    "type": "text",
                    "text": mongodb_json_dumps(response_data)
                }],
                is_error=False
            )
            
        except Exception as error:
            print(f"Create patient profile error: {str(error)}")
            return self.handle_error(error) 