from typing import List, Dict, Any, Optional, Set, Union
from datetime import datetime
from pymongo.collection import Collection


class MongoFieldSchema:
    """Schema definition for a MongoDB field."""
    
    def __init__(
        self, 
        field: str, 
        type: str, 
        is_required: bool, 
        sub_fields: Optional[List['MongoFieldSchema']] = None
    ):
        self.field = field
        self.type = type
        self.is_required = is_required
        self.sub_fields = sub_fields
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dict representation."""
        result = {
            "field": self.field,
            "type": self.type,
            "isRequired": self.is_required
        }
        
        if self.sub_fields:
            result["subFields"] = [field.to_dict() for field in self.sub_fields]
            
        return result


class MongoCollectionSchema:
    """Schema definition for a MongoDB collection."""
    
    def __init__(
        self, 
        collection: str, 
        fields: List[MongoFieldSchema], 
        count: int,
        indexes: Optional[List[Any]] = None
    ):
        self.collection = collection
        self.fields = fields
        self.count = count
        self.indexes = indexes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dict representation."""
        result = {
            "collection": self.collection,
            "fields": [field.to_dict() for field in self.fields],
            "count": self.count
        }
        
        if self.indexes:
            result["indexes"] = self.indexes
            
        return result


def infer_schema_from_value(value: Any) -> str:
    """
    Determine the type of a value.
    
    Args:
        value: Any value to analyze
        
    Returns:
        String representation of the value's type
    """
    if value is None:
        return "null"
    if isinstance(value, list):
        return "array"
    if isinstance(value, datetime):
        return "date"
    if isinstance(value, dict) or (hasattr(value, "__dict__") and not callable(value)):
        return "object"
    return type(value).__name__


def infer_schema_from_document(
    doc: Dict[str, Any], 
    parent_path: str = ""
) -> List[MongoFieldSchema]:
    """
    Infer schema from a document, handling nested fields.
    
    Args:
        doc: Document to analyze
        parent_path: Parent field path for nested documents
        
    Returns:
        List of field schemas
    """
    schema = []

    for key, value in doc.items():
        field_path = f"{parent_path}.{key}" if parent_path else key
        field_type = infer_schema_from_value(value)
        
        field = MongoFieldSchema(
            field=field_path,
            type=field_type,
            is_required=True
        )

        if field_type == "object" and value is not None:
            field.sub_fields = infer_schema_from_document(value, field_path)
        elif field_type == "array" and isinstance(value, list) and len(value) > 0:
            array_type = infer_schema_from_value(value[0])
            if array_type == "object":
                field.sub_fields = infer_schema_from_document(
                    value[0], 
                    f"{field_path}[]"
                )
                
        schema.append(field)
        
    return schema


async def build_collection_schema(
    collection: Collection, 
    sample_size: int = 100
) -> MongoCollectionSchema:
    """
    Build a schema for a MongoDB collection by analyzing a sample of documents.
    
    Args:
        collection: MongoDB collection
        sample_size: Number of documents to sample
        
    Returns:
        Collection schema
    """
    # Get sample documents
    docs = list(collection.find({}).limit(sample_size))
    
    # Get collection stats
    count = collection.count_documents({})
    indexes = list(collection.list_indexes())

    # Track field types and required fields
    field_schemas: Dict[str, Set[str]] = {}
    required_fields: Set[str] = set()

    # Analyze each document
    for doc in docs:
        doc_schema = infer_schema_from_document(doc)
        for field in doc_schema:
            if field.field not in field_schemas:
                field_schemas[field.field] = set()
                
            field_schemas[field.field].add(field.type)
            required_fields.add(field.field)

    # Check which fields are not in all documents
    for doc in docs:
        doc_fields = set(doc.keys())
        fields_to_remove = set()
        
        for field in required_fields:
            if field.split(".")[0] not in doc_fields:
                fields_to_remove.add(field)
                
        required_fields -= fields_to_remove

    # Build the field schemas
    fields = []
    for field, types in field_schemas.items():
        field_type = next(iter(types)) if len(types) == 1 else "|".join(types)
        fields.append(
            MongoFieldSchema(
                field=field,
                type=field_type,
                is_required=(field in required_fields)
            )
        )

    # Add sub-fields information
    for doc in docs:
        doc_schema = infer_schema_from_document(doc)
        for field_schema in doc_schema:
            if field_schema.sub_fields:
                existing_field = next(
                    (f for f in fields if f.field == field_schema.field), 
                    None
                )
                if existing_field and not existing_field.sub_fields:
                    existing_field.sub_fields = field_schema.sub_fields

    return MongoCollectionSchema(
        collection=collection.name,
        fields=fields,
        count=count,
        indexes=indexes
    )
