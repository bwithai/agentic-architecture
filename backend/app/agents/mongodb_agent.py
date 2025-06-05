from pymongo import MongoClient
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Custom JSON encoder to handle MongoDB types
class MongoJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class MongoDBAgent:
    def __init__(self, host="localhost", port=27017, db_name="kami"):
        self.client = MongoClient(host=host, port=port)
        self.db = self.client[db_name]
        self.collections = {
            "users": self.db.users,
            "products": self.db.products
        }
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not found in environment variables.")
            print("Please set your OpenAI API key in an .env file or directly in the environment.")
            
        self.openai_client = OpenAI(api_key=api_key)
        
    def close(self):
        """Close the MongoDB connection"""
        if self.client:
            self.client.close()
    
    def parse_query_with_openai(self, query):
        """Use OpenAI to parse the user's query into structured format"""
        # Get a sample document from each collection to help the model understand the schema
        product_schema = {}
        user_schema = {}
        
        try:
            product_sample = self.db.products.find_one()
            if product_sample:
                # Convert ObjectId to string
                product_sample["_id"] = str(product_sample["_id"])
                # Convert datetime objects
                for key, value in product_sample.items():
                    if isinstance(value, datetime):
                        product_sample[key] = value.isoformat()
                product_schema = json.dumps(product_sample)
            
            user_sample = self.db.users.find_one()
            if user_sample:
                # Convert ObjectId to string
                user_sample["_id"] = str(user_sample["_id"])
                # Convert datetime objects
                for key, value in user_sample.items():
                    if isinstance(value, datetime):
                        user_sample[key] = value.isoformat()
                user_schema = json.dumps(user_sample)
        except Exception as e:
            print(f"Error fetching schema samples: {e}")
        
        prompt = f"""
        Parse the following database query into a structured format. The query is related to a MongoDB database 
        with 'users' and 'products' collections.
        
        PRODUCT COLLECTION SCHEMA SAMPLE:
        {product_schema}
        
        USER COLLECTION SCHEMA SAMPLE:
        {user_schema}
        
        Query: "{query}"
        
        Return a JSON object with the following structure:
        {{
            "collection": "users or products (which collection to query)",
            "operation": "count, list, find, find_one, aggregate, or unknown",
            "filters": {{
                "field_name": "value",
                "nested.field": "value"  // For nested fields like symptoms.symptom_name
            }},
            "projection": ["field1", "field2"],  // Fields to include in results, empty means all fields
            "limit": number or null,
            "explanation": "Brief explanation of what the query is asking for"
        }}
        
        RULES FOR PARSING:
        1. For questions about products or medications, use the products collection
        2. For questions about users, patients, or customers, use the users collection
        3. For questions about specific fields like quantity, price, etc., identify the correct field in the schema
        4. For nested fields like compositions or symptoms, use dot notation (e.g., "compositions.ingredient_name")
        5. For questions about descriptions or symptoms, look in the appropriate nested arrays
        6. If the query is ambiguous or doesn't specify filters, use empty filters and explain the ambiguity
        
        EXAMPLES:
        
        Example 1: "How many users are there?"
        {{
            "collection": "users",
            "operation": "count",
            "filters": {{}},
            "projection": [],
            "limit": null,
            "explanation": "Count total number of users in the database"
        }}
        
        Example 2: "Find products where name is iPhone"
        {{
            "collection": "products",
            "operation": "find",
            "filters": {{
                "product_name": "iPhone"
            }},
            "projection": [],
            "limit": 5,
            "explanation": "Find products with name containing iPhone"
        }}
        
        Example 3: "Show me product names for first 5 products"
        {{
            "collection": "products",
            "operation": "list",
            "filters": {{}},
            "projection": ["product_name"],
            "limit": 5,
            "explanation": "List the names of the first 5 products"
        }}
        
        Example 4: "What is the description for fatigue symptom"
        {{
            "collection": "products",
            "operation": "find",
            "filters": {{
                "symptoms.symptom_name": "fatigue"
            }},
            "projection": ["symptoms"],
            "limit": 5,
            "explanation": "Find products that have symptoms related to fatigue and show their descriptions"
        }}
        
        Example 5: "How much quantity of Molagit Tab is available"
        {{
            "collection": "products",
            "operation": "find",
            "filters": {{
                "product_name": "Molagit Tab"
            }},
            "projection": ["product_name", "compositions.quantity", "compositions.ingredient_name"],
            "limit": 1,
            "explanation": "Find the quantity of Molagit Tab medication available"
        }}

        Only return the JSON object, nothing else.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that parses database queries and understands MongoDB schema."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=1000
            )
            
            parsed_result = response.choices[0].message.content
            
            # Extract JSON from the response
            try:
                # Find JSON part in the response if there's extra text
                start_idx = parsed_result.find('{')
                end_idx = parsed_result.rfind('}') + 1
                if start_idx >= 0 and end_idx > 0:
                    json_str = parsed_result[start_idx:end_idx]
                    return json.loads(json_str)
                else:
                    return json.loads(parsed_result)
            except json.JSONDecodeError:
                print(f"Error parsing OpenAI response: {parsed_result}")
                # Return a default structure as fallback
                return {
                    "collection": "unknown",
                    "operation": "unknown",
                    "filters": {},
                    "projection": [],
                    "limit": None,
                    "explanation": "Failed to parse query"
                }
                
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            # Return a default structure as fallback
            return {
                "collection": "unknown",
                "operation": "unknown",
                "filters": {},
                "projection": [],
                "limit": None,
                "explanation": f"API error: {str(e)}"
            }
    
    def process_query(self, query):
        """Process natural language query using OpenAI and return relevant information"""
        # Parse the query using OpenAI
        parsed_query = self.parse_query_with_openai(query)
        
        collection_name = parsed_query.get("collection")
        operation = parsed_query.get("operation")
        filters = parsed_query.get("filters", {})
        projection = parsed_query.get("projection", [])
        limit = parsed_query.get("limit")
        explanation = parsed_query.get("explanation", "")
        
        # Default limit if not specified
        if not limit:
            limit = 10 if operation == "list" else 5
        
        # Handle unknown collection
        if collection_name not in self.collections:
            return "Please specify if you're asking about users or products."
        
        collection = self.collections[collection_name]
        
        # Create projection dictionary for MongoDB
        projection_dict = {}
        if projection:
            for field in projection:
                projection_dict[field] = 1
        
        # Preprocess filters for text search and nested fields
        mongo_filters = {}
        for field, value in filters.items():
            # Handle nested fields (dot notation)
            if isinstance(value, str):
                # Use case-insensitive regex for string fields
                mongo_filters[field] = {"$regex": value, "$options": "i"}
            else:
                mongo_filters[field] = value
        
        # Handle different types of operations
        if operation == "count":
            count = collection.count_documents(mongo_filters)
            return f"There are {count} {collection_name} matching your criteria in the database."
        
        elif operation in ["list", "find", "find_one"]:
            # Use projection if specified
            if projection_dict:
                if operation == "find_one":
                    items = [collection.find_one(mongo_filters, projection_dict)]
                    if items[0] is None:
                        items = []
                else:
                    items = list(collection.find(mongo_filters, projection_dict).limit(limit))
            else:
                if operation == "find_one":
                    items = [collection.find_one(mongo_filters)]
                    if items[0] is None:
                        items = []
                else:
                    items = list(collection.find(mongo_filters).limit(limit))
            
            # Convert ObjectId to string for JSON serialization
            for item in items:
                if item and "_id" in item:
                    item["_id"] = str(item["_id"])
            
            if not items:
                return f"No {collection_name} found matching your criteria."
            
            # Format response with explanation
            if explanation:
                result = f"{explanation}:\n\n"
                result += json.dumps(items, indent=2, cls=MongoJSONEncoder)
                return result
            else:
                # Use custom encoder to handle dates and other MongoDB types
                return json.dumps(items, indent=2, cls=MongoJSONEncoder)
        
        # Add support for basic aggregation
        elif operation == "aggregate":
            return "Aggregation queries are not yet supported in this version."
        
        else:
            return (
                f"I understand you're asking about {collection_name}, but I'm not sure what specifically.\n"
                f"Try queries like:\n"
                f"- 'How many {collection_name} are there?'\n"
                f"- 'List all {collection_name}'\n"
                f"- 'Find {collection_name} where name is X'\n"
                f"- 'Show me the product names for the first 5 products'\n"
                f"- 'What is the description for fatigue symptom'"
            )


def main():
    print("MongoDB Agent initialized. Type 'exit' to quit.")
    print("Set up your OpenAI API key in a .env file as OPENAI_API_KEY=your_key_here")
    agent = MongoDBAgent()
    
    try:
        while True:
            query = input("\nEnter your query: ")
            if query.lower() in ["exit", "quit", "q"]:
                break
            
            response = agent.process_query(query)
            print("\nResponse:")
            print(response)
    
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        agent.close()


if __name__ == "__main__":
    main() 