"""
Product Tools

This module provides specialized tools for checking product availability and escalation paths
when information cannot be determined.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from config.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_product_availability(product_id: str) -> Dict[str, Any]:
    """
    Check a product's availability in the MongoDB database.
    
    Args:
        product_id (str): The ID of the product to check
        
    Returns:
        Dict[str, Any]: A dictionary containing availability information or error details
    """
    client = None
    
    try:
        # Connect to MongoDB
        logger.info(f"Checking availability for product ID: {product_id}")
        client = MongoClient(config.mongodb.uri)
        db = client[config.mongodb.database]
        
        # Check if product exists
        product = db.products.find_one({"_id": product_id})
        
        if not product:
            logger.warning(f"Product not found: {product_id}")
            return {
                "status": "not_found",
                "message": f"Product with ID {product_id} not found in database",
                "product_id": product_id,
                "timestamp": datetime.now()
            }
        
        # Extract availability information
        availability = {
            "status": "found",
            "product_id": product_id,
            "product_name": product.get("name", "Unknown"),
            "in_stock": product.get("in_stock", False),
            "quantity": product.get("quantity", 0),
            "available_date": product.get("available_date"),
            "timestamp": datetime.now()
        }
        
        logger.info(f"Product availability checked: {availability['product_name']} - In stock: {availability['in_stock']}")
        return availability
    
    except PyMongoError as e:
        logger.error(f"MongoDB error checking product {product_id}: {str(e)}")
        return {
            "status": "error",
            "message": f"Database error: {str(e)}",
            "product_id": product_id,
            "timestamp": datetime.now()
        }
    
    except Exception as e:
        logger.error(f"Unexpected error checking product {product_id}: {str(e)}")
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}",
            "product_id": product_id,
            "timestamp": datetime.now()
        }
    
    finally:
        if client:
            client.close()


def escalate_to_admin(request_details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Escalate a product inquiry to the admin team.
    
    In a production environment, this would connect to a ticketing system,
    email service, or admin notification channel.
    
    Args:
        request_details (Dict[str, Any]): Details of the request to escalate
        
    Returns:
        Dict[str, Any]: A dictionary containing the escalation result
    """
    try:
        # Log the escalation for now (in production, this would be a more robust solution)
        logger.info(f"ESCALATION TO ADMIN: {request_details}")
        
        # In a real system, this would be where the ticket is created or email is sent
        
        return {
            "status": "escalated",
            "message": "Request has been forwarded to the administrator team",
            "request_id": f"req-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "timestamp": datetime.now()
        }
    
    except Exception as e:
        logger.error(f"Error escalating to admin: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to escalate request: {str(e)}",
            "timestamp": datetime.now()
        }


def extract_product_id_from_query(query: str) -> Optional[str]:
    """
    Attempt to extract a product ID from a natural language query.
    This is a simple implementation - in production, you would use a more robust
    entity extraction approach.
    
    Args:
        query (str): The user's natural language query
        
    Returns:
        Optional[str]: The extracted product ID or None if not found
    """
    # Simple implementation - look for common patterns
    # In production, this would use a more sophisticated entity extraction technique
    
    import re
    
    # Look for patterns like "product ID: ABC123" or "product ABC123"
    patterns = [
        r"product\s+id[:\s]+([a-zA-Z0-9-_]+)",
        r"product[:\s]+([a-zA-Z0-9-_]+)",
        r"id[:\s]+([a-zA-Z0-9-_]+)",
        r"item[:\s]+([a-zA-Z0-9-_]+)",
        r"sku[:\s]+([a-zA-Z0-9-_]+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None 