"""
Query Understanding Agent

This agent is responsible for understanding user queries and converting them
into structured representations for database operations.
"""

from typing import Dict, Any, List, Optional
import json
from pydantic import BaseModel, Field

from agents.base.base_agent import BaseAgent, AgentInput, AgentOutput
from config.config import config
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


class MongoDBQuery(BaseModel):
    """Structured representation of a MongoDB query."""
    collection: str = Field(description="The MongoDB collection to query")
    operation: str = Field(description="The operation type (find, aggregate, count, etc.)")
    filter: Dict[str, Any] = Field(default_factory=dict, description="The filter criteria")
    projection: Optional[Dict[str, Any]] = Field(default=None, description="Fields to include/exclude")
    sort: Optional[Dict[str, int]] = Field(default=None, description="Sort criteria")
    limit: Optional[int] = Field(default=None, description="Maximum number of results")
    skip: Optional[int] = Field(default=None, description="Number of documents to skip")


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
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at understanding natural language queries about databases and converting them into MongoDB operations.
            
Your task is to analyze a user's query about MongoDB data and convert it into a structured MongoDB query object.

The query object should include:
- collection: Which collection to query
- operation: What MongoDB operation to perform (find, aggregate, count, etc.)
- filter: The criteria to filter documents
- projection: (Optional) Which fields to include or exclude
- sort: (Optional) How to sort the results
- limit: (Optional) Maximum number of results
- skip: (Optional) Number of documents to skip

Respond ONLY with a valid JSON object containing these fields.
"""),
            ("human", "{query}")
        ])
    
    async def run(self, inputs: AgentInput) -> AgentOutput:
        """
        Process a natural language query and convert it to a structured MongoDB query.
        
        Args:
            inputs (AgentInput): The agent inputs including the user query
            
        Returns:
            AgentOutput: The structured query and other outputs
        """
        self.log(f"Processing query: {inputs.query}")
        
        chain = self.prompt | self.llm
        result = await chain.ainvoke({"query": inputs.query})
        
        try:
            # Extract the JSON from the model's response
            response_text = result.content
            
            # Try to find JSON-like content in the response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                structured_query = json.loads(json_str)
                
                # Validate the structured query
                mongo_query = MongoDBQuery(**structured_query)
                
                self.log(f"Generated MongoDB query: {mongo_query.model_dump_json()}")
                
                return AgentOutput(
                    response=f"I understood your query about the {mongo_query.collection} collection.",
                    data={"mongodb_query": mongo_query.model_dump()},
                    status="success"
                )
            else:
                raise ValueError("No valid JSON found in the response")
        
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