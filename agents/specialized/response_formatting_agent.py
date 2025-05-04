"""
Response Formatting Agent

This agent is responsible for formatting raw data into human-friendly responses.
It also handles translating responses to the user's preferred language.
"""

from typing import Dict, Any, List, Optional
import json
from datetime import datetime
from pydantic import BaseModel, Field

from agents.base.base_agent import BaseAgent, AgentInput, AgentOutput
from config.config import config
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from agents.tools.translation_tools import TranslationTool
from agents.utils.translation_utils import is_english


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
    Also handles translating responses to the user's preferred language.
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
            "1. Always acknowledge whether data was found or not\n"
            "2. Summarize the key information from the results\n"
            "3. Format the data in a readable way (using bullet points, tables, etc.)\n"
            "4. If the results are empty, clearly explain this and suggest possible reasons\n"
            "5. For errors, explain in simple, non-technical terms\n"
            "6. Always be direct, answering exactly what was asked\n"
            "7. Show a sample of the actual data, not just a summary\n\n"
            "Make your response complete, well-formatted, and helpful."
        )
        
        # Simple human prompt with variables
        human_prompt = (
            "Original Query: {original_query}\n\n"
            "Database Info: {database_info}\n\n"
            "MongoDB Query: {mongodb_query}\n\n"
            "Results: {result}\n\n"
            "Count: {count}\n"
            "Status: {status}\n"
            "Error: {error}\n\n"
            "Create a helpful response to the original query, showing samples of the actual data found:"
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
        
        # Initialize translation tool
        self.translation_tool = TranslationTool()
    
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
        
        # Get language information from context
        language_info = inputs.context.get("language_info", {})
        if not language_info and "query_result" in inputs.context:
            # Try to get language info from query_result
            language_info = inputs.context.get("query_result", {}).get("language_info", {})
            
        target_language = language_info.get("language_code", "en")
        is_translation_needed = language_info and not is_english(target_language)
        
        self.log(f"Language info: {language_info}")
        self.log(f"Target language: {target_language}, Translation needed: {is_translation_needed}")
        
        # Handle database connection errors specially
        if error and "MongoDB connection not established" in error:
            error_message = "I'm unable to process your query at the moment. The database is currently unavailable."
            
            # Translate error message if needed
            if is_translation_needed:
                error_message = await self.translation_tool.translate_from_english(
                    error_message, target_language
                )
                self.log(f"Translated error message to {language_info.get('language_name', target_language)}")
                
            return AgentOutput(
                response=error_message,
                data={"error": error},
                status="error"
            )
        
        # Check if we have multi-collection results
        is_multi_collection = False
        total_count = 0
        
        if "multi_collection_results" in inputs.context:
            is_multi_collection = True
            result = inputs.context.get("multi_collection_results", {})
            # Calculate total across all collections
            for collection_results in result.values():
                if isinstance(collection_results, list):
                    total_count += len(collection_results)
        else:
            result = inputs.context.get("result", [])
            total_count = len(result) if isinstance(result, list) else 0
            
        count = inputs.context.get("count", total_count)
        
        # Format the data for the prompt using custom encoder to handle datetime objects
        mongodb_query_str = json.dumps(mongodb_query, indent=2)
        result_str = json.dumps(result, indent=2, cls=MongoJSONEncoder)
        
        # Get database information from config
        database_info = {
            "database": config.mongodb.database,
            "collections": list(result.keys()) if is_multi_collection else None,
            "count": count,
            "is_multi_collection": is_multi_collection
        }
        database_info_str = json.dumps(database_info, indent=2)
        
        # Invoke the chain to get English response
        chain = self.prompt | self.llm
        llm_response = await chain.ainvoke({
            "original_query": original_query,
            "mongodb_query": mongodb_query_str,
            "result": result_str,
            "count": count,
            "status": status,
            "error": error or "None",
            "database_info": database_info_str
        })
        
        formatted_response = llm_response.content
        
        # Translate the response if needed
        if is_translation_needed:
            original_response = formatted_response
            formatted_response = await self.translation_tool.translate_from_english(
                formatted_response, target_language
            )
            self.log(f"Translated response from English to {language_info.get('language_name', target_language)}")
            self.log(f"Original: {original_response[:100]}...")
            self.log(f"Translated: {formatted_response[:100]}...")
        
        self.log("Response formatted successfully")
        
        return AgentOutput(
            response=formatted_response,
            data={
                "original_data": result,
                "language_info": language_info
            },
            status="success"
        )
    
    def get_description(self) -> str:
        """
        Get a description of what this agent does.
        
        Returns:
            str: Description of the agent
        """
        return "Formats raw data into human-friendly, natural language responses in the user's preferred language." 