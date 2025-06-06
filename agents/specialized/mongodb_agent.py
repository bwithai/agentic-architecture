"""
MongoDB Agent

This agent is responsible for executing MongoDB queries and retrieving data.
"""

from typing import Dict, Any, List, Optional
import json
import asyncio
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from agents.base.base_agent import BaseAgent, AgentInput, AgentOutput
from config.config import config
from agents.specialized.query_understanding_agent import MongoDBQuery, ComplexQuery


class MongoDBAgent(BaseAgent):
    """
    Agent for executing queries against MongoDB and retrieving data.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the MongoDB agent.
        
        Args:
            verbose (bool): Whether to output verbose logs
        """
        super().__init__(name="mongodb", verbose=verbose)
        self.client = None
        self.db = None
    
    def connect(self):
        """
        Connect to MongoDB.
        """
        if not self.client:
            self.log(f"Connecting to MongoDB at {config.mongodb.uri}")
            self.client = MongoClient(config.mongodb.uri)
            self.db = self.client[config.mongodb.database]
            self.log(f"Connected to database {config.mongodb.database}")
    
    def disconnect(self):
        """
        Disconnect from MongoDB.
        """
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            self.log("Disconnected from MongoDB")
    
    async def execute_single_query(self, mongo_query: MongoDBQuery) -> Dict[str, Any]:
        """
        Execute a single MongoDB query.
        
        Args:
            mongo_query (MongoDBQuery): The MongoDB query to execute
            
        Returns:
            Dict[str, Any]: Query results and metadata
        """
        self.log(f"Executing query on collection: {mongo_query.collection}")
        
        # Get the collection
        collection = self.db[mongo_query.collection]
        
        # Execute the query based on operation type
        result = None
        if mongo_query.operation == "find":
            cursor = collection.find(
                filter=mongo_query.filter,
                projection=mongo_query.projection
            )
            
            if mongo_query.sort:
                cursor = cursor.sort(list(mongo_query.sort.items()))
            
            if mongo_query.skip:
                cursor = cursor.skip(mongo_query.skip)
            
            if mongo_query.limit:
                cursor = cursor.limit(mongo_query.limit)
            
            # Convert cursor to list of dictionaries
            result = list(cursor)
            
            # Convert ObjectId to string for JSON serialization
            for doc in result:
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])
        
        elif mongo_query.operation == "count":
            result = collection.count_documents(mongo_query.filter)
        
        elif mongo_query.operation == "aggregate":
            # For simplicity, we're assuming filter contains the aggregate pipeline
            pipeline = mongo_query.filter.get("pipeline", [])
            result = list(collection.aggregate(pipeline))
            
            # Convert ObjectId to string for JSON serialization
            for doc in result:
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])
        
        else:
            raise ValueError(f"Unsupported operation: {mongo_query.operation}")
        
        # Create a response based on the results
        count = len(result) if isinstance(result, list) else 1
        
        return {
            "collection": mongo_query.collection,
            "result": result,
            "count": count,
            "operation": mongo_query.operation
        }
    
    async def run(self, inputs: AgentInput) -> AgentOutput:
        """
        Execute a MongoDB query based on the input.
        
        Args:
            inputs (AgentInput): The agent inputs including MongoDB query information
            
        Returns:
            AgentOutput: The query results and status
        """
        try:
            self.connect()
            
            # Extract MongoDB query from context
            if "mongodb_query" not in inputs.context:
                return AgentOutput(
                    response="No MongoDB query information provided.",
                    status="error",
                    error="Missing MongoDB query in context"
                )
            
            query_data = inputs.context["mongodb_query"]
            
            # Handle both single and complex queries
            if "is_multi_collection" in query_data:
                # This is a ComplexQuery object
                complex_query = ComplexQuery(**query_data)
                
                if complex_query.is_multi_collection:
                    self.log(f"Processing multi-collection query with {len(complex_query.queries)} collections")
                    
                    # Execute each query and collect results
                    all_results = {}
                    total_count = 0
                    
                    for query in complex_query.queries:
                        try:
                            result = await self.execute_single_query(query)
                            collection_name = query.collection
                            all_results[collection_name] = result
                            total_count += result["count"]
                        except Exception as e:
                            self.log(f"Error executing query for {query.collection}: {str(e)}")
                            all_results[query.collection] = {"error": str(e)}
                    
                    # Create a combined result structure
                    collections = list(all_results.keys())
                    response = f"Queries executed across {len(collections)} collections: {', '.join(collections)}. Found {total_count} total results."
                    
                    return AgentOutput(
                        response=response,
                        data={"multi_collection_results": all_results, "count": total_count},
                        status="success"
                    )
                else:
                    # Single query wrapped in a complex query structure
                    mongo_query = complex_query.queries[0] if complex_query.queries else None
                    if not mongo_query:
                        return AgentOutput(
                            response="No valid MongoDB query found.",
                            status="error",
                            error="Empty queries list in complex query"
                        )
            else:
                # Legacy format (direct MongoDBQuery)
                mongo_query = MongoDBQuery(**query_data)
            
            # Execute the single query
            result_data = await self.execute_single_query(mongo_query)
            
            # Create a response
            response = f"Query executed successfully on {mongo_query.collection}. Found {result_data['count']} result(s)."
            
            return AgentOutput(
                response=response,
                data={"result": result_data["result"], "count": result_data["count"]},
                status="success"
            )
        
        except PyMongoError as e:
            self.log(f"MongoDB error: {str(e)}")
            return AgentOutput(
                response=f"Error executing MongoDB query: {str(e)}",
                status="error",
                error=str(e)
            )
        
        except Exception as e:
            self.log(f"Error: {str(e)}")
            return AgentOutput(
                response=f"An error occurred: {str(e)}",
                status="error",
                error=str(e)
            )
        
        finally:
            self.disconnect()
    
    def get_description(self) -> str:
        """
        Get a description of what this agent does.
        
        Returns:
            str: Description of the agent
        """
        return "Executes queries against MongoDB and retrieves data based on structured query information." 