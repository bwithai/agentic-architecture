"""
Script to check MongoDB databases and collections.
This will list all databases, collections, and document counts to help diagnose connection issues.
"""

import pymongo
import os
from dotenv import load_dotenv

# Load environment variables (if they exist)
load_dotenv()

def check_mongodb():
    """Connect to MongoDB and list all databases, collections, and document counts."""
    # Get the connection string (default to localhost if not in .env)
    mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    
    print(f"Connecting to MongoDB at: {mongodb_uri}")
    
    try:
        # Connect to MongoDB
        client = pymongo.MongoClient(mongodb_uri)
        
        # Check connection
        client.admin.command('ping')
        print("MongoDB connection successful!\n")
        
        # List all databases
        print("==== AVAILABLE DATABASES ====")
        db_names = client.list_database_names()
        for db_name in db_names:
            if db_name not in ['admin', 'config', 'local']:
                print(f"\nDatabase: {db_name}")
                
                # Get the database
                db = client[db_name]
                
                # List all collections in the database
                print("  Collections:")
                collections = db.list_collection_names()
                
                if not collections:
                    print("    No collections found")
                
                for collection_name in collections:
                    # Count documents in each collection
                    count = db[collection_name].count_documents({})
                    print(f"    - {collection_name}: {count} documents")
        
        print("\n==== CHECKING 'default_database' ====")
        default_db = client['default_database']
        print("Collections in 'default_database':")
        default_collections = default_db.list_collection_names()
        
        if not default_collections:
            print("  No collections found in 'default_database'")
        
        for collection_name in default_collections:
            count = default_db[collection_name].count_documents({})
            print(f"  - {collection_name}: {count} documents")
            
        # Check for 'users' collection in all databases
        print("\n==== LOOKING FOR 'users' COLLECTION IN ALL DATABASES ====")
        found = False
        for db_name in db_names:
            if db_name not in ['admin', 'config', 'local']:
                db = client[db_name]
                if 'users' in db.list_collection_names():
                    count = db['users'].count_documents({})
                    print(f"Found 'users' collection in database '{db_name}' with {count} documents")
                    found = True
        
        if not found:
            print("No 'users' collection found in any database")
            
    except Exception as e:
        print(f"MongoDB connection error: {str(e)}")
    finally:
        # Close the connection
        if 'client' in locals():
            client.close()
            print("\nMongoDB connection closed")

if __name__ == "__main__":
    check_mongodb() 