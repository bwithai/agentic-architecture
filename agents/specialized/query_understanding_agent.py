"""
Query Understanding Agent

This agent is responsible for:
1. Loading database schema information dynamically from MongoDB
2. Understanding user queries (already translated to English by the intent classifier)
3. Converting natural language queries into MongoDB operations
4. Executing MongoDB operations and returning results
"""

from typing import Dict, Any, Optional, List, Union
from pydantic import Field, BaseModel as PydanticBaseModel
from agents.base.base_agent import BaseAgent, AgentInput, AgentOutput
from config.config import config
from langchain_openai import ChatOpenAI
import json
from pymongo import MongoClient


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


class QueryUnderstandingAgent(BaseAgent):
    """
    Agent that understands user queries and converts them to MongoDB operations.
    This agent receives queries already translated to English by the intent classifier.
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
        # Store conversation history
        self.conversation_history = []

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
                        
                        # Check if the value is an array type
                        if isinstance(val, (list, tuple)):
                            array_type = "list" if isinstance(val, list) else "tuple"
                            field_details.append(f"   - {key}: {typ} (Array type: {array_type})")
                            
                            # If array contains objects/dicts, show their structure
                            if val and isinstance(val[0], dict):
                                field_details.append("     Array item structure:")
                                for sub_key, sub_val in val[0].items():
                                    sub_typ = type(sub_val).__name__
                                    field_details.append(f"       - {sub_key}: {sub_typ}")
                                    
                        else:
                            field_details.append(f"   - {key}: {typ}")
                            
                    schema_lines.extend(field_details)
                    
                    # Get count of documents
                    count = self.db[coll_name].count_documents({})
                    schema_lines.append(f"   - Total documents: {count}")
                else:
                    schema_lines.append("   - <empty collection>")
        except Exception as e:
            self.log(f"Error loading schema details: {str(e)}")
            schema_lines.append(f"\nError loading schema details: {str(e)}")
            
        return "\n".join(schema_lines)

    def _initialize_components(self):
        """Initialize the LLM and system prompt"""
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0.2,
            api_key=config.openai.api_key
        )
        
        # System prompt that instructs the model how to interact with MongoDB
        self.system_prompt = """
        You are a MongoDB assistant that helps users interact with their MongoDB database.
        You can translate natural language queries into MongoDB operations.
        
        You have access to the following MongoDB collections and their schema:
        {db_schema}

        You can perform these operations:
        - List collections
        - Query documents with filters
        - Insert documents
        - Update documents
        - Delete documents
        - Create and list indexes
        - Get collection schema
        
        When responding to user queries, you'll identify the appropriate MongoDB operation,
        collection, and parameters needed to fulfill the request. Always try to understand
        the user's intent and provide the most relevant data.
        
        Remember that queries have already been translated to English if they were originally
        in another language, so focus on understanding the query semantics.
        """
        
        # Define function descriptions to be used for function calling
        self.functions = [
            {
                "name": "set_database",
                "description": "Set the active MongoDB database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "db_name": {
                            "type": "string",
                            "description": "The name of the database to connect to"
                        }
                    },
                    "required": ["db_name"]
                }
            },
            {
                "name": "list_collections",
                "description": "List all collections in the current database",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "find_documents",
                "description": "Find documents in a collection based on a query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "The name of the collection to query"
                        },
                        "query": {
                            "type": "object",
                            "description": "MongoDB query filter in JSON format",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of documents to return"
                        },
                        "projection": {
                            "type": "object",
                            "description": "Fields to include or exclude from results"
                        }
                    },
                    "required": ["collection_name"]
                }
            },
            {
                "name": "insert_document",
                "description": "Insert a document into a collection",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "The name of the collection"
                        },
                        "document": {
                            "type": "object",
                            "description": "The document to insert"
                        }
                    },
                    "required": ["collection_name", "document"]
                }
            },
            {
                "name": "update_document",
                "description": "Update a document in a collection",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "The name of the collection"
                        },
                        "filter_query": {
                            "type": "object",
                            "description": "Query to find document to update"
                        },
                        "update_data": {
                            "type": "object",
                            "description": "New data to update document with"
                        }
                    },
                    "required": ["collection_name", "filter_query", "update_data"]
                }
            },
            {
                "name": "delete_document",
                "description": "Delete a document from a collection",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "The name of the collection"
                        },
                        "filter_query": {
                            "type": "object",
                            "description": "Query to find document to delete"
                        }
                    },
                    "required": ["collection_name", "filter_query"]
                }
            },
            {
                "name": "list_indexes",
                "description": "List indexes for a collection",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "The name of the collection"
                        }
                    },
                    "required": ["collection_name"]
                }
            },
            {
                "name": "create_index",
                "description": "Create an index on a collection",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "The name of the collection"
                        },
                        "field_name": {
                            "type": "string",
                            "description": "The field to create an index on"
                        },
                        "index_type": {
                            "type": "integer",
                            "description": "1 for ascending, -1 for descending",
                            "enum": [1, -1]
                        }
                    },
                    "required": ["collection_name", "field_name"]
                }
            },
            {
                "name": "get_collection_schema",
                "description": "Get the schema of a collection",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "The name of the collection"
                        },
                        "sample_size": {
                            "type": "integer",
                            "description": "Number of documents to sample for schema inference"
                        }
                    },
                    "required": ["collection_name"]
                }
            }
        ]

    def set_database(self, db_name):
        """Set the active database"""
        self.db = self.client[db_name]
        self.db_schema = self._load_database_schema()
        return f"Connected to database: {db_name}"
    
    def list_collections(self):
        """List all collections in the current database"""
        if not self.db:
            return "No database selected. Use 'set_database' first."
        
        collections = self.db.list_collection_names()
        return json.dumps(collections, indent=2)
    
    def find_documents(self, collection_name, query=None, limit=10, projection=None):
        """Find documents in a collection based on a query"""
        if not self.db:
            return "No database selected. Use 'set_database' first."
        
        if query is None:
            query = {}
            
        if projection is None:
            projection = {}
            
        collection = self.db[collection_name]
        results = list(collection.find(query, projection).limit(limit))
        
        # Convert ObjectId to string for JSON serialization
        for doc in results:
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])
                
        return json.dumps(results, indent=2, default=str)
    
    def insert_document(self, collection_name, document):
        """Insert a document into a collection"""
        if not self.db:
            return "No database selected. Use 'set_database' first."
            
        collection = self.db[collection_name]
        result = collection.insert_one(document)
        return f"Document inserted with ID: {result.inserted_id}"
    
    def update_document(self, collection_name, filter_query, update_data):
        """Update a document in a collection"""
        if not self.db:
            return "No database selected. Use 'set_database' first."
            
        collection = self.db[collection_name]
        result = collection.update_one(filter_query, {"$set": update_data})
        return f"Matched: {result.matched_count}, Modified: {result.modified_count}"
    
    def delete_document(self, collection_name, filter_query):
        """Delete a document from a collection"""
        if not self.db:
            return "No database selected. Use 'set_database' first."
            
        collection = self.db[collection_name]
        result = collection.delete_one(filter_query)
        return f"Deleted: {result.deleted_count} document(s)"
    
    def list_indexes(self, collection_name):
        """List indexes for a collection"""
        if not self.db:
            return "No database selected. Use 'set_database' first."
            
        collection = self.db[collection_name]
        indexes = list(collection.list_indexes())
        return json.dumps([idx for idx in indexes], indent=2, default=str)
    
    def create_index(self, collection_name, field_name, index_type=1):
        """Create an index on a collection"""
        if not self.db:
            return "No database selected. Use 'set_database' first."
            
        collection = self.db[collection_name]
        result = collection.create_index([(field_name, index_type)])
        return f"Index created: {result}"
    
    def get_collection_schema(self, collection_name, sample_size=10):
        """Infer the schema of a collection based on sample documents"""
        if not self.db:
            return "No database selected. Use 'set_database' first."
            
        collection = self.db[collection_name]
        sample_docs = list(collection.find().limit(sample_size))
        
        if not sample_docs:
            return "No documents found in collection."
            
        # Simple schema inference
        schema = {}
        for doc in sample_docs:
            for key, value in doc.items():
                if key not in schema:
                    schema[key] = []
                value_type = type(value).__name__
                if value_type not in schema[key]:
                    schema[key].append(value_type)
        
        return json.dumps(schema, indent=2)
    
    async def process_query(self, user_query, language_info=None):
        """
        Process a natural language query using OpenAI to determine the MongoDB operation
        
        Args:
            user_query (str): The user's query (already translated to English if needed)
            language_info (dict): Optional language information from intent classifier
        """
        # Add user query to conversation history for context
        self.conversation_history.append({"role": "user", "content": user_query})
        
        try:
            # Create system prompt with database schema
            filled_system_prompt = self.system_prompt.format(db_schema=self.db_schema)
            
            # Create message list with system prompt and conversation history
            messages = [{"role": "system", "content": filled_system_prompt}]
            messages.extend(self.conversation_history)
            
            # Get response from the model with function definitions
            from langchain_core.messages import SystemMessage, HumanMessage
            langchain_messages = [SystemMessage(content=filled_system_prompt)]
            
            # Add conversation history as LangChain message objects
            for msg in self.conversation_history:
                if msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                # Add other message types as needed (assistant, function, etc.)
            
            # Call the LLM using invoke (not async in newer LangChain versions)
            response = self.llm.invoke(
                langchain_messages,
                functions=self.functions,
                function_call="auto"
            )
            
            # Get the response
            function_name = None
            result = None
            mongo_query_command = None
            
            # Check if the model wants to call a function
            function_call = response.additional_kwargs.get('function_call')
            if function_call:
                function_name = function_call.get('name')
                function_args = json.loads(function_call.get('arguments', '{}'))
                
                # Store the MongoDB command representation
                if function_name == "find_documents":
                    collection = function_args.get("collection_name")
                    query = function_args.get("query", {})
                    limit = function_args.get("limit", 10)
                    projection = function_args.get("projection", {})
                    
                    # Create a MongoDB command string
                    limit_str = f".limit({limit})" if limit else ""
                    projection_str = f", {json.dumps(projection)}" if projection else ""
                    
                    mongo_query_command = {
                        f"{collection}": f"db.{collection}.find({json.dumps(query)}{projection_str}){limit_str}"
                    }
                # Add similar representations for other MongoDB operations
                elif function_name == "insert_document":
                    collection = function_args.get("collection_name")
                    document = function_args.get("document")
                    mongo_query_command = {
                        f"{collection}": f"db.{collection}.insertOne({json.dumps(document)})"
                    }
                elif function_name == "update_document":
                    collection = function_args.get("collection_name")
                    filter_query = function_args.get("filter_query")
                    update_data = function_args.get("update_data")
                    mongo_query_command = {
                        f"{collection}": f"db.{collection}.updateOne({json.dumps(filter_query)}, {{$set: {json.dumps(update_data)}}})"
                    }
                elif function_name == "delete_document":
                    collection = function_args.get("collection_name")
                    filter_query = function_args.get("filter_query")
                    mongo_query_command = {
                        f"{collection}": f"db.{collection}.deleteOne({json.dumps(filter_query)})"
                    }
                elif function_name == "list_collections":
                    mongo_query_command = {
                        "admin": "db.listCollections()"
                    }
                elif function_name == "list_indexes":
                    collection = function_args.get("collection_name")
                    mongo_query_command = {
                        f"{collection}": f"db.{collection}.getIndexes()"
                    }
                
                # Call the appropriate function
                if function_name == "set_database":
                    result = self.set_database(function_args.get("db_name"))
                elif function_name == "list_collections":
                    result = self.list_collections()
                elif function_name == "find_documents":
                    result = self.find_documents(
                        function_args.get("collection_name"),
                        function_args.get("query", {}),
                        function_args.get("limit", 10),
                        function_args.get("projection", {})
                    )
                elif function_name == "insert_document":
                    result = self.insert_document(
                        function_args.get("collection_name"),
                        function_args.get("document")
                    )
                elif function_name == "update_document":
                    result = self.update_document(
                        function_args.get("collection_name"),
                        function_args.get("filter_query"),
                        function_args.get("update_data")
                    )
                elif function_name == "delete_document":
                    result = self.delete_document(
                        function_args.get("collection_name"),
                        function_args.get("filter_query")
                    )
                elif function_name == "list_indexes":
                    result = self.list_indexes(function_args.get("collection_name"))
                elif function_name == "create_index":
                    result = self.create_index(
                        function_args.get("collection_name"),
                        function_args.get("field_name"),
                        function_args.get("index_type", 1)
                    )
                elif function_name == "get_collection_schema":
                    result = self.get_collection_schema(
                        function_args.get("collection_name"),
                        function_args.get("sample_size", 10)
                    )
                
                # Add function result to conversation history
                self.conversation_history.append({
                    "role": "function",
                    "name": function_name,
                    "content": result
                })
                
                # Create new LangChain messages for the second request
                from langchain_core.messages import FunctionMessage
                
                langchain_messages = [SystemMessage(content=filled_system_prompt)]
                for msg in self.conversation_history:
                    if msg["role"] == "user":
                        langchain_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "function":
                        langchain_messages.append(FunctionMessage(
                            name=msg["name"],
                            content=msg["content"]
                        ))
                
                # Get a new response from OpenAI to interpret the function result
                second_response = self.llm.invoke(langchain_messages)
                
                assistant_response = second_response.content
                self.conversation_history.append({"role": "assistant", "content": assistant_response})
                
                # Parse the result if it's a JSON string
                parsed_result = None
                try:
                    if isinstance(result, str) and (result.startswith('{') or result.startswith('[')):
                        parsed_result = json.loads(result)
                except json.JSONDecodeError:
                    parsed_result = result
                
                return assistant_response, parsed_result, function_name, mongo_query_command
            else:
                # If no function call, just return the response
                assistant_response = response.content
                self.conversation_history.append({"role": "assistant", "content": assistant_response})
                return assistant_response, None, None, None
        except Exception as e:
            self.log(f"Error processing query: {str(e)}")
            return f"Error processing query: {str(e)}", None, None, None
    
    async def run(self, inputs: AgentInput) -> AgentOutput:
        """
        Process user input and execute MongoDB operations.
        
        Args:
            inputs (AgentInput): The agent inputs including query and context
        """
        query_text = inputs.query
        language_info = inputs.context.get('language_info', {}) if inputs.context else {}
        
        if not self.db_connected:
            return AgentOutput(
                response="Unable to process your query. The database connection is not available.",
                data={
                    "language_info": language_info
                },
                status="error",
                error="MongoDB connection not established"
            )
            
        try:
            # Check if we have a translated query from intent classifier
            if language_info and not language_info.get('is_english', True):
                # Use the translated query instead of the original
                query_to_use = language_info.get('translated_query', query_text)
                self.log(f"Using translated query: {query_to_use}")
            else:
                query_to_use = query_text
            
            # Process the query
            response, result, function_name, mongodb_query = await self.process_query(
                query_to_use, language_info
            )
            
            # Calculate result count if available
            result_count = 0
            if isinstance(result, list):
                result_count = len(result)
            elif isinstance(result, dict) and 'count' in result:
                result_count = result['count']
            
            return AgentOutput(
                response=response,
                data={
                    "mongodb_result": result,
                    "function_called": function_name,
                    "mongodb_query": mongodb_query,
                    "original_query": query_text,
                    "result": result,
                    "count": result_count,
                    "status": "success",
                    "language_info": language_info
                },
                status="success"
            )
        except Exception as e:
            error_msg = str(e)
            self.log(f"Error processing MongoDB query: {error_msg}")
            return AgentOutput(
                response="Failed to process your MongoDB request",
                data={
                    "original_query": query_text,
                    "error": error_msg,
                    "status": "error",
                    "language_info": language_info
                },
                status="error",
                error=error_msg
            )

    def get_description(self) -> str:
        return "Converts natural language queries into MongoDB operations and executes them using live database." 