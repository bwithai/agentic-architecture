from pymongo import MongoClient
from urllib.parse import urlparse
from typing import Optional

# Global db reference that will be set when a client connects
db = None


class MongoDBClient:
    """A class to manage MongoDB connections using OOP principles."""

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize the MongoDB client.
        
        Args:
            database_url: MongoDB connection string (optional at initialization)
        """
        self.client = None
        self.db = None
        self.database_url = database_url
        if database_url:
            self.connect(database_url)
    
    async def connect(self, database_url: Optional[str] = None) -> None:
        """
        Connect to MongoDB and initialize the db variable.
        
        Args:
            database_url: MongoDB connection string (overrides the one provided at initialization)
        """
        if database_url:
            self.database_url = database_url
        
        if not self.database_url:
            raise ValueError("Database URL must be provided")
            
        try:
            self.client = MongoClient(self.database_url)
            resource_base_url = urlparse(self.database_url)
            
            # Safely extract database name from path
            path_parts = resource_base_url.path.strip('/').split("/")
            db_name = path_parts[0] if path_parts and path_parts[0] else "test"
            
            print(f"Connecting to database: {db_name}")
            self.db = self.client[db_name]
            
            # Set the global db reference
            global db
            db = self.db
        except Exception as error:
            print(f"MongoDB connection error: {error}")
            raise
    
    async def close(self) -> None:
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            
            # Clear the global db reference
            global db
            db = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        if not self.client and self.database_url:
            await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()