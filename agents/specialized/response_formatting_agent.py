"""
Response Formatting Agent

This agent is responsible for formatting raw data into human-friendly responses.
"""

from typing import Dict, Any, List, Optional
import json
from datetime import datetime
from pydantic import BaseModel, Field

from agents.base.base_agent import BaseAgent, AgentInput, AgentOutput
from config.config import config
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


# Custom JSON encoder to handle MongoDB types like datetime
class MongoJSONEncoder(json.JSONEncoder):
    """Custom JSON Encoder that can handle MongoDB types like datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class ResponseFormattingAgent(BaseAgent):
    """
    Agent for formatting raw MongoDB query results into human-friendly responses.
    """
    
    def __init__(self, model_name: str = None, verbose: bool = False):
        """
        Initialize the response formatting agent.
        
        Args:
            model_name (str): Name of the LLM model to use
            verbose (bool): Whether to output verbose logs
        """
        super().__init__(name="response_formatter", verbose=verbose)
        
        model_name = model_name or config.openai.model
        
        self.llm = ChatOpenAI(
            api_key=config.openai.api_key,
            model=model_name,
            temperature=config.openai.temperature
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at converting database query results into natural, human-friendly responses.

Your task is to take the original user query and the raw results from a MongoDB query, and create a helpful, well-formatted response.

Guidelines:
1. Summarize the key information from the results
2. Format the data in a readable way (using bullet points, tables, etc. when appropriate)
3. Highlight important patterns or insights
4. Be conversational and friendly
5. If the results are empty, explain this clearly
6. If there was an error, explain it in simple terms

Here's the data you'll receive:
- original_query: The user's original natural language query
- mongodb_query: The structured MongoDB query that was executed
- result: The raw results from MongoDB
- count: The number of results
- status: Whether the query was successful or had an error
- error: Any error message (if status is "error")

Respond with a complete, well-formatted, and helpful message that directly answers the user's original query.
"""),
            ("human", """
Original Query: {original_query}

MongoDB Query: 
{mongodb_query}

Results: 
{result}

Count: {count}
Status: {status}
Error: {error}

Create a helpful response to the original query:
""")
        ])
    
    async def run(self, inputs: AgentInput) -> AgentOutput:
        """
        Format raw data into a human-friendly response.
        
        Args:
            inputs (AgentInput): The agent inputs including the original query and results
            
        Returns:
            AgentOutput: The formatted, human-friendly response
        """
        self.log(f"Formatting response for query: {inputs.query}")
        
        # Extract data from context
        original_query = inputs.query
        mongodb_query = inputs.context.get("mongodb_query", {})
        result = inputs.context.get("result", [])
        count = inputs.context.get("count", 0)
        status = inputs.context.get("status", "unknown")
        error = inputs.context.get("error", None)
        
        # Format the data for the prompt using custom encoder to handle datetime objects
        mongodb_query_str = json.dumps(mongodb_query, indent=2)
        
        # Use custom encoder to handle datetime objects in MongoDB results
        result_str = json.dumps(result, indent=2, cls=MongoJSONEncoder)
        
        chain = self.prompt | self.llm
        llm_response = await chain.ainvoke({
            "original_query": original_query,
            "mongodb_query": mongodb_query_str,
            "result": result_str,
            "count": count,
            "status": status,
            "error": error or "None"
        })
        
        formatted_response = llm_response.content
        
        self.log("Response formatted successfully")
        
        return AgentOutput(
            response=formatted_response,
            data={"original_data": result},
            status="success"
        )
    
    def get_description(self) -> str:
        """
        Get a description of what this agent does.
        
        Returns:
            str: Description of the agent
        """
        return "Formats raw data into human-friendly, natural language responses." 