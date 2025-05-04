"""
Query Understanding Agent

This agent is responsible for:
1. Loading database schema information dynamically from MongoDB
2. Understanding user queries
3. Converting natural language queries into MongoDB queries
"""

from typing import Dict, Any, Optional, List, Union
from pydantic import Field, BaseModel as PydanticBaseModel
from agents.base.base_agent import BaseAgent, AgentInput, AgentOutput
from config.config import config
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pymongo import MongoClient
import json
import re


# Define the model classes needed by mongodb_agent.py
class MongoDBQuery(PydanticBaseModel):
    """Simple MongoDB query targeting a single collection"""
    collection: str = Field(..., description="The MongoDB collection name to query")
    query: Dict[str, Any] = Field(..., description="The MongoDB query filter")
    projection: Optional[Dict[str, Any]] = Field(None, description="Optional fields to include/exclude")
    sort: Optional[Dict[str, int]] = Field(None, description="Optional sort criteria")
    limit: Optional[int] = Field(None, description="Optional result limit")


class ComplexQuery(PydanticBaseModel):
    """Complex query that might span multiple collections or require aggregation"""
    queries: List[MongoDBQuery] = Field(..., description="List of MongoDB queries to execute")
    join_type: Optional[str] = Field(None, description="How to join results (union, intersection)")
    description: str = Field(..., description="Description of what the complex query does")


class MongoDBQuerySchema(PydanticBaseModel):
    """Schema for the MongoDB query output"""
    mongodb_query: Dict[str, Any] = Field(..., description="The MongoDB query as a JSON object with 'collection' and 'query' fields")
    explanation: str = Field(..., description="Brief explanation of what the query does")


class QueryUnderstandingAgent(BaseAgent):
    """
    Agent that understands user queries and converts them to MongoDB queries.
    """
    
    def __init__(self, model_name: str = None, verbose: bool = False):
        super().__init__(name="query_understanding", verbose=verbose)
        self.model_name = model_name or config.openai.model
        # Initialize DB client and load schema
        self.db_connected = False
        try:
            self.client = MongoClient(config.mongodb.uri, serverSelectionTimeoutMS=5000)
            # Verify connection with a quick check
            self.client.server_info()
            self.db = self.client[config.mongodb.database]
            self.db_schema = self._load_database_schema()
            self.db_connected = True
            self.log("Successfully connected to MongoDB")
        except Exception as e:
            self.log(f"Failed to connect to MongoDB: {str(e)}")
            self.db_schema = "Database connection not available"
        # Initialize LLM and prompt
        self._initialize_components()
        self.log("Initialized Query Understanding Agent with dynamic schema fetch")

    def _load_database_schema(self) -> str:
        """Dynamically build schema information by inspecting collections and sample docs"""
        schema_lines: List[str] = [f"Database: {config.mongodb.database}", "Collections:"]
        
        try:
            for idx, coll_name in enumerate(self.db.list_collection_names(), start=1):
                schema_lines.append(f"\n{idx}. {coll_name}")
                sample = self.db[coll_name].find_one()
                
                if sample:
                    field_details = []
                    for key, val in sample.items():
                        typ = type(val).__name__
                        example = str(val)[:30] + "..." if len(str(val)) > 30 else str(val)
                        field_details.append(f"   - {key}: {typ} (Example: {example})")
                    schema_lines.extend(field_details)
                    
                    # Get count of documents
                    count = self.db[coll_name].count_documents({})
                    schema_lines.append(f"   - Total documents: {count}")
                else:
                    schema_lines.append("   - <empty collection>")
                    
            # Add some examples of common MongoDB query patterns
            schema_lines.append("\nCommon MongoDB Query Patterns:")
            schema_lines.append("1. Find all documents in a collection: { 'collection': 'users', 'query': {}, 'limit': 10 }")
            schema_lines.append("2. Find by exact match: { 'collection': 'users', 'query': { 'username': 'john' } }")
            schema_lines.append("3. Find with multiple criteria: { 'collection': 'products', 'query': { 'price': { '$lt': 100 }, 'category': 'electronics' } }")
            schema_lines.append("4. Limit results: { 'collection': 'orders', 'query': {}, 'limit': 5 }")
            schema_lines.append("5. Sort results: { 'collection': 'products', 'query': {}, 'sort': { 'price': 1 } }")
            schema_lines.append("6. Search by text pattern: { 'collection': 'users', 'query': { 'name': { '$regex': 'john', '$options': 'i' } } }")
            schema_lines.append("7. Query for a range: { 'collection': 'products', 'query': { 'price': { '$gte': 10, '$lte': 50 } } }")
            
        except Exception as e:
            self.log(f"Error loading schema details: {str(e)}")
            schema_lines.append(f"\nError loading schema details: {str(e)}")
            
        return "\n".join(schema_lines)

    def _initialize_components(self):
        """Initialize the LLM and query understanding prompt"""
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0.2,
            api_key=config.openai.api_key
        )
        
        # We'll prepare the full system prompt during query time to avoid template variables
        self.system_prompt_template = """You are an expert AI assistant specialized in converting natural language queries to MongoDB queries.

You will receive a user's question and the current database schema. Your task is to:
1. Analyze what data the user is looking for
2. Identify which collection(s) are relevant
3. Construct the appropriate MongoDB query
4. Apply any filtering, sorting, or limiting the user specifies

IMPORTANT GUIDELINES:
- ALWAYS include a collection name in your query
- Use proper MongoDB operators ($eq, $gt, $lt, $in, etc.) when needed
- Add appropriate 'limit' parameters based on the user's request
- If the user asks for "first" or "top" N items, add a 'limit' parameter with that number
- Support regex searches for partial text matches using $regex
- Return MULTIPLE queries if the user asks for data from multiple collections
- Only use collections that actually exist in the database schema

MULTI-COLLECTION QUERIES:
If the user asks for data from multiple collections (e.g., "users and products"), return an ARRAY of multiple query objects, like:
[
  {"collection": "users", "query": {}, "limit": 3},
  {"collection": "products", "query": {}, "limit": 3}
]

OUTPUT FORMAT:
Return a JSON object with these fields:
- mongodb_query: Either a single query object or an array of query objects
- explanation: A brief explanation of what the query does

Here is the schema of the database:
{db_schema}

Now, create a MongoDB query for the user's question: {query}
"""
        
        # Create a simple parser that will extract just the JSON
        class MongoDBQueryParser(JsonOutputParser):
            def parse(self, text):
                # Try to extract JSON from text
                import re
                import json
                
                # Look for JSON pattern
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```|```\s*([\s\S]*?)\s*```|({[\s\S]*}|\[[\s\S]*\])', text)
                
                if json_match:
                    json_str = next(group for group in json_match.groups() if group)
                    try:
                        return json.loads(json_str)
                    except:
                        # If parsing the extracted JSON fails, try to find JSON object
                        try:
                            return json.loads(text)
                        except:
                            # Fall back to simple query
                            return {
                                "mongodb_query": {"collection": "users", "query": {}},
                                "explanation": "Fallback simple query for all users."
                            }
                else:
                    # Try to parse the whole text as JSON
                    try:
                        return json.loads(text)
                    except:
                        # Fall back to simple query
                        return {
                            "mongodb_query": {"collection": "users", "query": {}},
                            "explanation": "Fallback simple query for all users."
                        }
        
        self.query_parser = MongoDBQueryParser()
    
    async def run(self, inputs: AgentInput) -> AgentOutput:
        """
        Process user input and convert to MongoDB query.
        """
        query_text = inputs.query
        
        if not self.db_connected:
            return AgentOutput(
                response="Unable to process your query. The database connection is not available.",
                data={},
                status="error",
                error="MongoDB connection not established"
            )
            
        try:
            # Extract numeric limits from the query
            limit_patterns = {
                "users": re.search(r'(\d+)\s+(user|users)', query_text.lower()),
                "products": re.search(r'(\d+)\s+(product|products)', query_text.lower())
            }
            
            limits = {}
            for collection, match in limit_patterns.items():
                if match:
                    limits[collection] = int(match.group(1))
            
            # Also look for general limits
            general_limit_match = re.search(r'(?:top|first|latest|recent)\s+(\d+)', query_text.lower())
            default_limit = int(general_limit_match.group(1)) if general_limit_match else None
            
            # Create the complete prompt with the actual values substituted
            system_prompt = self.system_prompt_template.format(
                db_schema=self.db_schema,
                query=query_text
            )
            
            # Create a new prompt template directly with the expanded content
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "Please create the MongoDB query now.")
            ])
            
            # Create the chain
            chain = prompt | self.llm | self.query_parser
            
            # Get result from LLM - no variables to pass
            result = await chain.ainvoke({})
            
            # Make sure we have both required fields
            if "mongodb_query" not in result:
                raise ValueError("LLM response missing 'mongodb_query' field")
                
            if "explanation" not in result:
                result["explanation"] = "Generated MongoDB query based on user request."
            
            # Apply any detected limits if not already in query
            mongodb_query = result["mongodb_query"]
            
            # If query is an array (multi-collection)
            if isinstance(mongodb_query, list):
                for q in mongodb_query:
                    collection = q.get("collection", "")
                    if collection in limits and "limit" not in q:
                        q["limit"] = limits[collection]
                    elif default_limit and "limit" not in q:
                        q["limit"] = default_limit
            # If query is a single object
            elif isinstance(mongodb_query, dict):
                collection = mongodb_query.get("collection", "")
                if collection in limits and "limit" not in mongodb_query:
                    mongodb_query["limit"] = limits[collection]
                elif default_limit and "limit" not in mongodb_query:
                    mongodb_query["limit"] = default_limit
                    
            # Log outputs
            self.log(f"Generated MongoDB query: {json.dumps(mongodb_query)}")
            self.log(f"Explanation: {result['explanation']}")
            
            return AgentOutput(
                response=result["explanation"],
                data={
                    "mongodb_query": mongodb_query,
                    "explanation": result["explanation"]
                },
                status="success"
            )
        except Exception as e:
            self.log(f"Error generating MongoDB query: {str(e)}")
            return AgentOutput(
                response="Sorry, I couldn't understand the query. Could you rephrase?",
                data={},
                status="error",
                error=str(e)
            )

    def get_description(self) -> str:
        return "Converts natural language queries into MongoDB queries using live database schema."
