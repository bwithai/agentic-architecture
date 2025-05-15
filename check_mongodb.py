"""
Script to check MongoDB databases and collections.
This will list all databases, collections, and document counts to help diagnose connection issues.
"""

from typing import List
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


def get_db_schema():
    from config.config import config
    """Get the schema of the database."""
    client = pymongo.MongoClient(config.mongodb.uri, serverSelectionTimeoutMS=5000)
    db = client[config.mongodb.database]
    schema_lines: List[str] = [f"Database: {config.mongodb.database}", "Collections:"]
    
    for idx, coll_name in enumerate(db.list_collection_names(), start=1):
        schema_lines.append(f"\n{idx}. {coll_name}")
        sample = db[coll_name].find_one()
        
        if sample:
            field_details = []
            for key, val in sample.items():
                typ = type(val).__name__
                example = str(val)[:30] + "..." if len(str(val)) > 30 else str(val)
                
                # Check if the value is an array type
                if isinstance(val, (list, tuple)):
                    array_type = "list" if isinstance(val, list) else "tuple"
                    field_details.append(f"   - {key}: {typ} (Array type: {array_type}, Example: {example})")
                elif isinstance(val, dict):
                    field_details.append(f"   - {key}: {typ} (JSON object, Example: {example})")
                else:
                    field_details.append(f"   - {key}: {typ} (Example: {example})")
                    
            schema_lines.extend(field_details)
            
            # Get count of documents
            count = db[coll_name].count_documents({})
            schema_lines.append(f"   - Total documents: {count}")
        else:
            schema_lines.append("   - <empty collection>")

    return "\n".join(schema_lines)

if __name__ == "__main__":
    # check_mongodb() 
    print(get_db_schema())