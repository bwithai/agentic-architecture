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
from agents.specialized.query_understanding_agent import MongoDBQuery


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
            mongo_query = MongoDBQuery(**query_data)
            
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
                return AgentOutput(
                    response=f"Unsupported operation: {mongo_query.operation}",
                    status="error",
                    error=f"Operation {mongo_query.operation} not implemented"
                )
            
            # Create a response based on the results
            count = len(result) if isinstance(result, list) else 1
            response = f"Query executed successfully. Found {count} result(s)."
            
            return AgentOutput(
                response=response,
                data={"result": result, "count": count},
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