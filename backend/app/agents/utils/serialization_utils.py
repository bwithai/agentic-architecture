"""
Utility functions for serializing MongoDB data types and other complex objects
to JSON-compatible formats.
"""

import json
from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List, Union
from bson import ObjectId, DBRef, Binary, Regex, Code, Timestamp


class MongoJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle MongoDB specific types and other
    complex objects like datetime."""
    
    def default(self, obj: Any) -> Any:
        if obj is None:
            return None
        elif isinstance(obj, ObjectId):
            # Convert ObjectId to string
            return str(obj)
        elif isinstance(obj, (datetime, date)):
            # Convert datetime/date to ISO format string
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            # Convert Decimal to float
            return float(obj)
        elif isinstance(obj, (Binary, bytes, bytearray)):
            # Convert binary data to base64 string
            import base64
            return base64.b64encode(obj).decode('utf-8')
        elif isinstance(obj, DBRef):
            # Convert DBRef to dict
            return {
                'collection': obj.collection,
                'id': str(obj.id),
                'database': obj.database
            }
        elif isinstance(obj, Regex):
            # Convert Regex to string representation
            return {'$regex': obj.pattern}
        elif isinstance(obj, Code):
            # Convert Code to string
            return str(obj)
        elif isinstance(obj, Timestamp):
            # Convert Timestamp to dict with time and inc values
            return {'$timestamp': {'t': obj.time, 'i': obj.inc}}
        
        # Let the base class default method handle other types or raise TypeError
        return super().default(obj)


def serialize_mongodb_doc(doc: Any) -> Any:
    """
    Recursively serialize a MongoDB document or list of documents to ensure
    all values are JSON serializable.
    
    Args:
        doc: MongoDB document (dict), list of documents, or any value to serialize
        
    Returns:
        A JSON-serializable version of the document/value
    """
    if doc is None:
        return None
    
    if isinstance(doc, list):
        return [serialize_mongodb_doc(item) for item in doc]
    
    if isinstance(doc, dict):
        result = {}
        for key, value in doc.items():
            result[key] = serialize_mongodb_doc(value)
        return result
    
    # Handle various MongoDB types
    if isinstance(doc, ObjectId):
        return str(doc)
    elif isinstance(doc, (datetime, date)):
        return doc.isoformat()
    elif isinstance(doc, Decimal):
        return float(doc)
    elif isinstance(doc, (Binary, bytes, bytearray)):
        import base64
        return base64.b64encode(doc).decode('utf-8')
    elif isinstance(doc, DBRef):
        return {
            'collection': doc.collection,
            'id': str(doc.id),
            'database': doc.database
        }
    elif isinstance(doc, Regex):
        return {'$regex': doc.pattern}
    elif isinstance(doc, Code):
        return str(doc)
    elif isinstance(doc, Timestamp):
        return {'$timestamp': {'t': doc.time, 'i': doc.inc}}
    
    # Return other types as is
    return doc


def mongodb_json_dumps(obj: Any, **kwargs) -> str:
    """
    Serialize an object to a JSON formatted string, properly handling
    MongoDB specific types like ObjectId.
    
    Args:
        obj: The object to serialize
        **kwargs: Additional arguments to pass to json.dumps
        
    Returns:
        JSON string representation
    """
    # Set indent=2 by default if not provided
    if 'indent' not in kwargs:
        kwargs['indent'] = 2
        
    return json.dumps(obj, cls=MongoJSONEncoder, **kwargs) 