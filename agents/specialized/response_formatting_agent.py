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
        
        # Create simple system prompt with no JSON examples
        system_prompt = (
            "You are an expert at converting database query results into natural, "
            "human-friendly responses. Your task is to take the original user query "
            "and the raw results from a MongoDB query, and create a helpful, "
            "well-formatted response.\n\n"
            "Guidelines:\n"
            "1. Summarize the key information from the results\n"
            "2. Format the data in a readable way (using bullet points, tables, etc.)\n"
            "3. Highlight important patterns or insights\n"
            "4. Be conversational and friendly\n"
            "5. If the results are empty, explain this clearly\n"
            "6. If there was an error, explain it in simple terms\n\n"
            "Make your response complete, well-formatted, and helpful."
        )
        
        # Simple human prompt with variables
        human_prompt = (
            "Original Query: {original_query}\n\n"
            "MongoDB Query: {mongodb_query}\n\n"
            "Is Multi-Collection: {is_multi_collection}\n\n"
            "Results: {result}\n\n"
            "Count: {count}\n"
            "Status: {status}\n"
            "Error: {error}\n\n"
            "Create a helpful response to the original query:"
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
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
        status = inputs.context.get("status", "unknown")
        error = inputs.context.get("error", None)
        
        # Check if we have multi-collection results
        is_multi_collection = False
        
        if "multi_collection_results" in inputs.context:
            is_multi_collection = True
            result = inputs.context.get("multi_collection_results", {})
            count = inputs.context.get("count", 0)
        else:
            result = inputs.context.get("result", [])
            count = inputs.context.get("count", 0)
        
        # Format the data for the prompt using custom encoder to handle datetime objects
        mongodb_query_str = json.dumps(mongodb_query, indent=2)
        result_str = json.dumps(result, indent=2, cls=MongoJSONEncoder)
        
        # Invoke the chain
        chain = self.prompt | self.llm
        llm_response = await chain.ainvoke({
            "original_query": original_query,
            "mongodb_query": mongodb_query_str,
            "result": result_str,
            "count": count,
            "status": status,
            "error": error or "None",
            "is_multi_collection": str(is_multi_collection)  # Convert to string to avoid any parsing issues
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