from pymongo import MongoClient
from urllib.parse import urlparse
from typing import Optional

# Global db reference that will be set when a client connects
db = None


class MongoDBClient:
    """A class to manage MongoDB connections using OOP principles."""

    def __init__(self, database_url: Optional[str] = None, database_name: Optional[str] = None):
        """
        Initialize the MongoDB client.
        
        Args:
            database_url: MongoDB connection string (optional at initialization)
            database_name: Database name (optional, will extract from URL if not provided)
        """
        self.client = None
        self.db = None
        self.database_url = database_url
        self.database_name = database_name
        
        # For backward compatibility, try sync connection if URL is provided
        if database_url:
            self.connect_sync(database_url, database_name)
    
    def connect_sync(self, database_url: Optional[str] = None, database_name: Optional[str] = None) -> None:
        """
        Synchronous connection to MongoDB.
        
        Args:
            database_url: MongoDB connection string
            database_name: Database name (optional)
        """
        if database_url:
            self.database_url = database_url
        if database_name:
            self.database_name = database_name
        
        if not self.database_url:
            raise ValueError("Database URL must be provided")
            
        try:
            self.client = MongoClient(self.database_url)
            
            # Determine database name
            if self.database_name:
                db_name = self.database_name
            else:
                # Extract from URL
                resource_base_url = urlparse(self.database_url)
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
    
    async def connect(self, database_url: Optional[str] = None, database_name: Optional[str] = None) -> None:
        """
        Async connection to MongoDB (wraps sync connection).
        
        Args:
            database_url: MongoDB connection string
            database_name: Database name (optional)
        """
        self.connect_sync(database_url, database_name)
    
    async def close(self) -> None:
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            
            # Clear the global db reference
            global db
            db = None
    
    def close_sync(self) -> None:
        """Synchronous close of MongoDB connection."""
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