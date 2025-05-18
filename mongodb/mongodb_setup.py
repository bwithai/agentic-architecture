"""
MongoDB setup module for the AI agent application.
This provides functions to connect to MongoDB and set up the database client.
"""

from mongodb.client import MongoDBClient

async def setup_mongodb(db_url: str) -> MongoDBClient:
    """
    Connect to MongoDB and return the client.
    Also ensures the global DB reference is set.
    
    Args:
        db_url: The MongoDB connection URL
        
    Returns:
        The MongoDB client instance
    """
    db_client = MongoDBClient()
    await db_client.connect(db_url)
    
    # Ensure the global db reference is set
    from mongodb import client as mongodb_client_module
    if mongodb_client_module.db is None:
        mongodb_client_module.db = db_client.db
        
    return db_client 