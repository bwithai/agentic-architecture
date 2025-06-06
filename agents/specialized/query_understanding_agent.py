"""
Query Understanding Agent

This agent is responsible for understanding user queries and converting them
into structured representations for database operations.
"""

from typing import Dict, Any, List, Optional, Type, Literal, ClassVar, Union
import json
import re
from datetime import datetime
import asyncio
from pydantic import BaseModel, Field, create_model, ConfigDict

from agents.base.base_agent import BaseAgent, AgentInput, AgentOutput
from config.config import config
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain
from langchain.tools import BaseTool, Tool
from langchain.schema.runnable import RunnablePassthrough
from pymongo import MongoClient
from pymongo.collection import Collection


class MongoDBQuery(BaseModel):
    """Structured representation of a MongoDB query."""
    collection: str = Field(description="The MongoDB collection to query")
    operation: str = Field(description="The operation type (find, aggregate, count, etc.)")
    filter: Dict[str, Any] = Field(default_factory=dict, description="The filter criteria")
    projection: Optional[Dict[str, Any]] = Field(default=None, description="Fields to include/exclude")
    sort: Optional[Dict[str, int]] = Field(default=None, description="Sort criteria")
    limit: Optional[int] = Field(default=None, description="Maximum number of results")
    skip: Optional[int] = Field(default=None, description="Number of documents to skip")


class ComplexQuery(BaseModel):
    """Container for multiple MongoDB queries."""
    is_multi_collection: bool = Field(default=False, description="Whether this is a multi-collection query")
    queries: List[MongoDBQuery] = Field(default_factory=list, description="List of individual queries")


class CollectionSchema(BaseModel):
    """Schema for a MongoDB collection"""
    fields: Dict[str, str] = Field(default_factory=dict, description="Field names and descriptions")
    name_fields: List[str] = Field(default_factory=list, description="Fields likely to contain names")
    sample_documents: Optional[List[Dict[str, Any]]] = Field(default=None, description="Sample documents for reference")
    field_types: Dict[str, str] = Field(default_factory=dict, description="Field data types")
    
    
class DatabaseSchema(BaseModel):
    """Complete MongoDB database schema"""
    collections: Dict[str, CollectionSchema] = Field(default_factory=dict, description="Collection schemas")
    updated_at: Optional[datetime] = Field(default=None, description="When schema was last updated")


class SchemaFetcherTool(BaseModel):
    """Model for the schema fetcher tool"""
    mongodb_uri: str = Field(default=config.mongodb.uri, description="MongoDB connection URI")
    database_name: str = Field(default=config.mongodb.database, description="MongoDB database name")
    schema_cache: Dict[str, Any] = Field(default_factory=dict, description="Cached schema information")
    cache_initialized: bool = Field(default=False, description="Whether the cache is initialized")
    sample_size: int = Field(default=10, description="Number of documents to sample per collection")
    verbose: bool = Field(default=False, description="Whether to output verbose logs")


# Create a standalone class that doesn't inherit from BaseTool
class MongoSchemaHelper:
    """Helper class for MongoDB schema operations"""
    
    def __init__(self, params: SchemaFetcherTool):
        self.params = params
        
    def log(self, message: str) -> None:
        """Log a message if verbose mode is enabled"""
        if self.params.verbose:
            print(f"[SchemaFetcher] {message}")
    
    def connect(self) -> Optional[MongoClient]:
        """Connect to MongoDB and return the client"""
        try:
            client = MongoClient(self.params.mongodb_uri)
            # Test connection with a quick ping
            client.admin.command('ping')
            self.log(f"Connected to MongoDB at {self.params.mongodb_uri}")
            return client
        except Exception as e:
            self.log(f"Error connecting to MongoDB: {str(e)}")
            return None
    
    def run(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Fetch the database schema by sampling collections.
        
        Args:
            force_refresh: Whether to force a refresh of the schema cache
        
        Returns:
            Dict: The database schema
        """
        if self.params.cache_initialized and not force_refresh:
            self.log("Returning cached schema")
            return self.params.schema_cache
        
        self.log(f"Fetching schema for database: {self.params.database_name}")
        schema = {}
        
        client = self.connect()
        if not client:
            self.log("No connection - using fallback schema")
            return self._get_fallback_schema()
            
        try:
            db = client[self.params.database_name]
            collections = db.list_collection_names()
            self.log(f"Found collections: {collections}")
            
            for collection_name in collections:
                collection = db[collection_name]
                schema[collection_name] = self._get_collection_schema(collection, collection_name)
                self.log(f"Added schema for collection: {collection_name}")
            
            # Update cache
            self.params.schema_cache = schema
            self.params.cache_initialized = True
            return schema
                
        except Exception as e:
            self.log(f"Error fetching schema: {str(e)}")
            return self._get_fallback_schema()
        finally:
            if client:
                client.close()
                self.log("Closed MongoDB connection")

    def _get_collection_schema(self, collection: Collection, collection_name: str) -> Dict[str, Any]:
        """Analyze a collection and return its schema"""
        try:
            # Sample documents to infer schema
            sample_docs = list(collection.find().limit(self.params.sample_size))
            self.log(f"Sampled {len(sample_docs)} documents from {collection_name}")
                
            if not sample_docs:
                return self._get_empty_collection_schema(collection_name)
                
            fields = {}
            name_fields = []
            field_types = {}
            
            # Analyze fields from sample documents
            for doc in sample_docs:
                for field, value in doc.items():
                    if field not in fields:
                        # Infer field description and type
                        description = self._infer_field_description(field, value)
                        fields[field] = description
                        field_types[field] = self._get_field_type(value)
                        
                        # Identify potential name fields
                        if self._is_likely_name_field(field):
                            name_fields.append(field)
            
            # Convert ObjectIds to strings for JSON serialization
            clean_samples = []
            for doc in sample_docs[:3]:  # Store just a few samples
                clean_doc = {}
                for k, v in doc.items():
                    clean_doc[k] = str(v) if hasattr(v, '__str__') and not isinstance(v, (str, int, float, bool, list, dict)) else v
                clean_samples.append(clean_doc)
            
            return {
                "fields": fields,
                "name_fields": list(set(name_fields)),
                "sample_documents": clean_samples,
                "field_types": field_types
            }
        except Exception as e:
            self.log(f"Error analyzing collection {collection_name}: {str(e)}")
            return self._get_empty_collection_schema(collection_name)

    def _get_empty_collection_schema(self, collection_name: str) -> Dict[str, Any]:
        """Return a default schema for an empty or inaccessible collection"""
        # Handle known collections with predefined schemas
        if collection_name == "users":
            return {
                "fields": {
                    "user_name": "The username or login name",
                    "name": "The user's full name",
                    "email": "User's email address",
                    "role": "User role in the system",
                    "location": "User's physical location",
                    "created_at": "When the user was created"
                },
                "name_fields": ["user_name", "name"],
                "field_types": {
                    "user_name": "string",
                    "name": "string",
                    "email": "string",
                    "role": "string",
                    "location": "string",
                    "created_at": "date"
                },
                "sample_documents": [
                    {"user_name": "jsmith", "name": "John Smith", "email": "john@example.com", "role": "user"},
                    {"user_name": "agarcia", "name": "Ana Garcia", "email": "ana@example.com", "role": "admin"}
                ]
            }
        elif collection_name == "products":
            return {
                "fields": {
                    "name": "Product name",
                    "description": "Product description",
                    "price": "Product price in the default currency",
                    "category": "Product category or classification",
                    "sku": "Stock Keeping Unit - unique product identifier"
                },
                "name_fields": ["name"],
                "field_types": {
                    "name": "string",
                    "description": "string",
                    "price": "number",
                    "category": "string",
                    "sku": "string"
                },
                "sample_documents": [
                    {"name": "Widget Pro", "price": 29.99, "category": "Widgets", "sku": "WDG-001"},
                    {"name": "Super Gadget", "price": 49.99, "category": "Gadgets", "sku": "GDG-100"}
                ]
            }
        else:
            # Generic empty schema
            return {
                "fields": {"_id": "Document ID"},
                "name_fields": [],
                "field_types": {"_id": "objectid"},
                "sample_documents": []
            }

    def _get_fallback_schema(self) -> Dict[str, Any]:
        """Return a fallback schema when DB connection fails"""
        self.log("Using fallback schema")
        return {
            "users": self._get_empty_collection_schema("users"),
            "products": self._get_empty_collection_schema("products"),
            "orders": {
                "fields": {
                    "order_id": "Unique order identifier",
                    "customer_id": "Reference to the customer who placed the order",
                    "products": "List of products in the order",
                    "total": "Total order amount",
                    "status": "Current order status",
                    "created_at": "When the order was created"
                },
                "name_fields": [],
                "field_types": {
                    "order_id": "string",
                    "customer_id": "string",
                    "products": "array",
                    "total": "number",
                    "status": "string",
                    "created_at": "date"
                },
                "sample_documents": [
                    {"order_id": "ORD-1001", "customer_id": "CUST-001", "total": 79.99, "status": "shipped"},
                    {"order_id": "ORD-1002", "customer_id": "CUST-002", "total": 29.99, "status": "pending"}
                ]
            }
        }
    
    def _infer_field_description(self, field_name: str, sample_value: Any) -> str:
        """Infer a description for a field based on its name and sample value"""
        field_lower = field_name.lower()
        
        # Common field patterns
        if field_name == '_id':
            return "MongoDB document unique identifier"
        elif 'name' in field_lower:
            if 'first' in field_lower:
                return "First name"
            elif 'last' in field_lower:
                return "Last name"
            elif 'user' in field_lower:
                return "Username used for login"
            else:
                return "Full name or display name"
        elif 'email' in field_lower:
            return "Email address"
        elif 'phone' in field_lower:
            return "Phone number"
        elif 'address' in field_lower:
            if 'line' in field_lower:
                return "Address line"
            elif 'city' in field_lower:
                return "City name"
            elif 'state' in field_lower or 'province' in field_lower:
                return "State or province"
            elif 'zip' in field_lower or 'postal' in field_lower:
                return "ZIP or postal code"
            elif 'country' in field_lower:
                return "Country name"
            else:
                return "Physical address information"
        elif 'price' in field_lower or 'cost' in field_lower:
            return f"Price or cost value in currency"
        elif 'date' in field_lower or field_lower.endswith('at') or field_lower.endswith('on'):
            if 'created' in field_lower or 'creation' in field_lower:
                return "Creation timestamp"
            elif 'updated' in field_lower or 'modified' in field_lower:
                return "Last update timestamp"
            elif 'birth' in field_lower:
                return "Birth date"
            else:
                return f"Date/time value for {field_name.replace('_', ' ')}"
        elif 'description' in field_lower:
            return "Descriptive text"
        elif 'image' in field_lower or 'photo' in field_lower or 'picture' in field_lower:
            return "Image file reference or URL"
        elif 'url' in field_lower or 'link' in field_lower:
            return "Web URL or link"
        elif 'status' in field_lower or 'state' in field_lower:
            return "Status indicator"
        elif 'type' in field_lower or 'category' in field_lower:
            return "Type or category classification"
        elif 'count' in field_lower or 'number' in field_lower or 'amount' in field_lower:
            return "Numeric count or quantity"
        elif 'id' in field_lower and field_lower != '_id':
            prefix = field_lower.split('_id')[0].strip('_')
            if prefix:
                return f"Unique identifier for {prefix}"
            return "Unique identifier"
        
        # Default based on value type
        return f"Field containing {self._get_field_type(sample_value)} data"
    
    def _get_field_type(self, value: Any) -> str:
        """Determine the type of a field value"""
        if value is None:
            return "null"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, dict):
            return "object"
        elif isinstance(value, datetime):
            return "date"
        else:
            # Try to get the class name for other types
            return value.__class__.__name__.lower()
    
    def _is_likely_name_field(self, field_name: str) -> bool:
        """Determine if a field is likely to contain a name"""
        name_indicators = ['name', 'username', 'user_name', 'full_name', 'firstName', 'lastName', 'title']
        field_lower = field_name.lower()
        
        return any(indicator in field_lower for indicator in name_indicators)
    
    def format_for_prompt(self) -> str:
        """Format the schema for inclusion in LLM prompts"""
        schema = self.run()
        
        if not schema:
            return "DATABASE SCHEMA: Unable to fetch current schema. Using generic field names."
            
        schema_text = "DATABASE SCHEMA INFORMATION (dynamically fetched from current database):\nCollections and their fields:\n"
        
        for collection, details in schema.items():
            schema_text += f"\n- {collection.upper()} COLLECTION:\n"
            fields = details.get("fields", {})
            field_types = details.get("field_types", {})
            
            for field, description in fields.items():
                field_type = field_types.get(field, "unknown")
                schema_text += f"  - {field} ({field_type}): {description}\n"
            
            if details.get("name_fields"):
                schema_text += f"  (Name fields: {', '.join(details['name_fields'])})\n"
                
            # Include a sample document if available
            samples = details.get("sample_documents", [])
            if samples:
                schema_text += f"  Sample document: {json.dumps(samples[0], indent=2)[:200]}...\n"
        
        return schema_text


# Now create a BaseTool implementation that delegates to the helper
class SchemaFetcher(BaseTool):
    """LangChain tool to dynamically fetch MongoDB schema"""
    
    name: str = "schema_fetcher"
    description: str = "Fetches schema information from MongoDB database"
    return_direct: bool = False
    args_schema: Type[BaseModel] = SchemaFetcherTool
    
    def __init__(self, mongodb_uri: str = None, database_name: str = None, verbose: bool = False):
        """Initialize the schema fetcher with connection details"""
        super().__init__()
        # Create the tool parameters
        params = SchemaFetcherTool(
            mongodb_uri=mongodb_uri or config.mongodb.uri,
            database_name=database_name or config.mongodb.database,
            verbose=verbose
        )
        # Create the helper that does the actual work
        self._helper = MongoSchemaHelper(params)
    
    def _run(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Run the schema fetcher"""
        return self._helper.run(force_refresh)
    
    async def _arun(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Async version of _run"""
        # This could be improved to be truly async with motor instead of pymongo
        return self._run(force_refresh)
    
    def format_for_prompt(self) -> str:
        """Format the schema for inclusion in LLM prompts"""
        return self._helper.format_for_prompt()


class QueryUnderstandingAgent(BaseAgent):
    """
    Agent that understands natural language queries and converts them
    to structured MongoDB queries.
    """
    
    def __init__(self, model_name: str = None, verbose: bool = False):
        """
        Initialize the query understanding agent.
        
        Args:
            model_name (str): Name of the LLM model to use
            verbose (bool): Whether to output verbose logs
        """
        super().__init__(name="query_understanding", verbose=verbose)
        
        model_name = model_name or config.openai.model
        
        self.llm = ChatOpenAI(
            api_key=config.openai.api_key,
            model=model_name,
            temperature=config.openai.temperature
        )
        
        # Initialize schema fetcher
        self.schema_fetcher = SchemaFetcher(verbose=verbose)
        
        # Initialize output parsers
        self.standard_parser = PydanticOutputParser(pydantic_object=MongoDBQuery)
        self.complex_parser = PydanticOutputParser(pydantic_object=ComplexQuery)
        
        # Create the prompt templates
        self._initialize_prompts()
        
        # Set up standard and complex chains
        self._setup_chains()
    
    def _initialize_prompts(self):
        """Initialize the prompt templates"""
        self.standard_system_prompt = (
            "You are an expert at understanding natural language queries about "
            "databases and converting them into MongoDB operations. "
            "Analyze the user's query and convert it into a structured MongoDB query object. "
            "The response should include collection name, operation type, "
            "filter criteria, and optionally projection, sort, limit, and skip parameters.\n\n"
            "{schema_info}\n\n"
            "IMPORTANT GUIDELINES:\n"
            "1. USE THE CORRECT FIELD NAMES from the schema above.\n"
            "2. For text searches, use case-insensitive regex with {{\"$regex\": \"value\", \"$options\": \"i\"}}.\n"
            "3. Make sure to determine the correct collection based on the query context.\n"
            "4. For partial matches, use regex patterns.\n"
            "5. For exact matches, use equality operators with the correct data types.\n\n"
            "Respond with a valid JSON object that conforms to the MongoDBQuery schema."
        )
        
        self.complex_system_prompt = (
            "You are an expert at understanding database queries that span multiple collections. "
            "Analyze if the query involves multiple MongoDB collections and break it into separate MongoDB query objects. "
            "\n\n{schema_info}\n\n"
            "Your response should be a JSON object that conforms to the ComplexQuery schema, "
            "with 'is_multi_collection' (boolean) and 'queries' (array of MongoDB query objects). "
            "Use the correct field names from the schema."
        )
        
        # Initialize prompts with empty schema (will be updated before use)
        self.standard_prompt = ChatPromptTemplate.from_messages([
            ("system", self.standard_system_prompt),
            ("human", "{query}")
        ])
        
        self.complex_prompt = ChatPromptTemplate.from_messages([
            ("system", self.complex_system_prompt),
            ("human", "{query}")
        ])
    
    def _setup_chains(self):
        """Set up the LangChain chains"""
        # Standard query chain
        self.standard_chain = (
            {"query": RunnablePassthrough(), "schema_info": lambda _: self.schema_fetcher.format_for_prompt()}
            | self.standard_prompt
            | self.llm
            | self.standard_parser
        )
        
        # Complex query chain
        self.complex_chain = (
            {"query": RunnablePassthrough(), "schema_info": lambda _: self.schema_fetcher.format_for_prompt()}
            | self.complex_prompt
            | self.llm
            | self.complex_parser
        )
        
        # Direct name query handling chain
        self.name_query_system_prompt = "Extract the name being searched for from the query."
        self.name_query_prompt = ChatPromptTemplate.from_messages([
            ("system", self.name_query_system_prompt),
            ("human", "Extract the name from: {query}"),
        ])
        self.name_query_chain = self.name_query_prompt | self.llm
    
    def _format_schema_for_prompt(self, schema):
        """Format the database schema for inclusion in the prompt."""
        return self.schema_fetcher.format_for_prompt()
    
    def _adjust_query_for_schema(self, query: MongoDBQuery, schema: Dict) -> None:
        """
        Adjust query based on the actual schema, mapping fields correctly.
        
        Args:
            query (MongoDBQuery): The MongoDB query to modify
            schema (Dict): The current database schema
        """
        collection = query.collection
        if not collection in schema or not query.filter:
            return
            
        collection_schema = schema[collection]
        valid_fields = set(collection_schema.get("fields", {}).keys())
        name_fields = set(collection_schema.get("name_fields", []))
        field_types = collection_schema.get("field_types", {})
        
        # Create a new filter with correct field mappings
        new_filter = {}
        
        for field, value in query.filter.items():
            # Handle common field name mismatches
            if field == "name" and "name" not in valid_fields and name_fields:
                # Use the first name field from the schema
                correct_field = next(iter(name_fields))
                self.log(f"Mapping 'name' to '{correct_field}' in {collection} collection")
                new_filter[correct_field] = value
            elif field not in valid_fields:
                # Try to find a matching field
                matches = [f for f in valid_fields if field.lower() in f.lower() or f.lower() in field.lower()]
                if matches:
                    correct_field = matches[0]  # Use the first match
                    self.log(f"Mapping '{field}' to '{correct_field}' in {collection} collection")
                    new_filter[correct_field] = value
                else:
                    # Keep the original field if no match found
                    new_filter[field] = value
            else:
                # Field exists in schema, keep as is
                new_filter[field] = value
                
            # Ensure text fields use case-insensitive regex
            if isinstance(new_filter[list(new_filter.keys())[-1]], str):
                field_name = list(new_filter.keys())[-1]
                text_value = new_filter[field_name]
                new_filter[field_name] = {"$regex": text_value, "$options": "i"}
                
            # Ensure proper type conversion for numeric fields
            field_name = list(new_filter.keys())[-1]
            field_type = field_types.get(field_name, "")
            if field_type in ["number", "integer", "float"] and isinstance(new_filter[field_name], str):
                try:
                    if field_type == "integer":
                        new_filter[field_name] = int(new_filter[field_name])
                    else:
                        new_filter[field_name] = float(new_filter[field_name])
                except ValueError:
                    # If conversion fails, leave as is (might be a regex)
                    pass
                
        # Replace the original filter with the adjusted one
        query.filter = new_filter
        self.log(f"Adjusted query filters based on dynamic schema: {query.filter}")
    
    def _ensure_regex_properly_formed(self, query: MongoDBQuery) -> None:
        """
        Ensure that regex patterns in the query are properly formed.
        
        Args:
            query (MongoDBQuery): The MongoDB query to check
        """
        if not query.filter:
            return
            
        # Check each filter field
        for field, value in list(query.filter.items()):
            # If it's a dict that looks like a regex pattern but isn't properly formed
            if isinstance(value, dict) and ('$regex' in value or '$options' in value):
                # Ensure both required components are present
                if '$regex' not in value:
                    # Missing regex pattern
                    if '$options' in value:
                        options = value['$options']
                        # Default to empty string if not properly formed
                        query.filter[field] = {'$regex': '', '$options': options}
                        self.log(f"Fixed malformed regex for {field}: added missing $regex pattern")
                
                if '$options' not in value and '$regex' in value:
                    # Missing options, add case-insensitive
                    regex_pattern = value['$regex']
                    query.filter[field] = {'$regex': regex_pattern, '$options': 'i'}
                    self.log(f"Fixed regex for {field}: added missing $options")
    
    async def _process_name_query(self, query: str) -> ComplexQuery:
        """
        Process a simple name query directly.
        
        Args:
            query (str): The user query
            
        Returns:
            ComplexQuery: A structured query object
        """
        # Use the LLM to extract the name
        name_extraction = await self.name_query_chain.ainvoke({"query": query})
        name_value = name_extraction.content.strip()
        
        self.log(f"Extracted name from query: {name_value}")
        
        # Get schema to find correct name field
        schema = self.schema_fetcher._run()
        user_schema = schema.get("users", {})
        name_fields = user_schema.get("name_fields", ["user_name", "name"])
        
        if name_fields:
            name_field = name_fields[0]  # Use first name field
            self.log(f"Using field '{name_field}' for name search")
            
            # Create direct query
            mongo_query = MongoDBQuery(
                collection="users",
                operation="find",
                filter={name_field: {"$regex": name_value, "$options": "i"}}
            )
            
            return ComplexQuery(
                is_multi_collection=False,
                queries=[mongo_query]
            )
        
        # Fallback if no name fields found
        return ComplexQuery(
            is_multi_collection=False,
            queries=[MongoDBQuery(
                collection="users",
                operation="find",
                filter={"name": {"$regex": name_value, "$options": "i"}}
            )]
        )
    
    async def run(self, inputs: AgentInput) -> AgentOutput:
        """
        Process a natural language query and convert it to a structured MongoDB query.
        
        Args:
            inputs (AgentInput): The agent inputs including the user query
            
        Returns:
            AgentOutput: The structured query and other outputs
        """
        self.log(f"Processing query: {inputs.query}")
        
        # Handle simple name queries directly
        name_match = re.search(r'(?:user|person|customer).*(?:name|called)\s+(\w+)', inputs.query, re.IGNORECASE)
        if name_match:
            try:
                complex_query = await self._process_name_query(inputs.query)
                name_value = name_match.group(1)
                
                return AgentOutput(
                    response=f"I understood your query to find a user named {name_value}.",
                    data={"mongodb_query": complex_query.model_dump()},
                    status="success"
                )
            except Exception as e:
                self.log(f"Error processing name query: {str(e)}")
                # Continue with standard processing if name query fails
        
        try:
            # First, check if this might be a complex query involving multiple collections
            try:
                complex_query = await self.complex_chain.ainvoke(inputs.query)
                
                # If it's a multi-collection query
                if complex_query.is_multi_collection and complex_query.queries:
                    self.log("Detected multi-collection query")
                    
                    # Ensure field names match the schema and queries use appropriate operators
                    schema = self.schema_fetcher._run()
                    for query in complex_query.queries:
                        self._adjust_query_for_schema(query, schema)
                        self._ensure_regex_properly_formed(query)
                    
                    collections = [q.collection for q in complex_query.queries]
                    self.log(f"Generated multi-collection query for: {', '.join(collections)}")
                    
                    return AgentOutput(
                        response=f"I understood your query about multiple collections: {', '.join(collections)}.",
                        data={"mongodb_query": complex_query.model_dump()},
                        status="success"
                    )
            except Exception as e:
                self.log(f"Error processing as complex query: {str(e)}")
            
            # If not a complex query or processing failed, try the standard approach
            self.log("Processing as standard query")
            
            try:
                mongo_query = await self.standard_chain.ainvoke(inputs.query)
                
                # Adjust query based on schema
                schema = self.schema_fetcher._run()
                self._adjust_query_for_schema(mongo_query, schema)
                self._ensure_regex_properly_formed(mongo_query)
                
                self.log(f"Generated MongoDB query for collection: {mongo_query.collection}")
                self.log(f"Query filters: {mongo_query.filter}")
                
                # Wrap in a ComplexQuery structure for consistent handling
                complex_query = ComplexQuery(
                    is_multi_collection=False,
                    queries=[mongo_query]
                )
                
                return AgentOutput(
                    response=f"I understood your query about the {mongo_query.collection} collection.",
                    data={"mongodb_query": complex_query.model_dump()},
                    status="success"
                )
            except Exception as e:
                self.log(f"Error processing as standard query: {str(e)}")
                raise
        
        except Exception as e:
            self.log(f"Error parsing query: {str(e)}")
            return AgentOutput(
                response="I'm sorry, I couldn't understand your query properly.",
                status="error",
                error=str(e)
            )
    
    def get_description(self) -> str:
        """
        Get a description of what this agent does.
        
        Returns:
            str: Description of the agent
        """
        return "Analyzes natural language queries and converts them into structured MongoDB operations." 