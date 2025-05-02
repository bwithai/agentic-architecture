"""
Query Classifier Agent

This agent is responsible for classifying user queries and determining
the appropriate specialized agent to handle them.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from agents.base.base_agent import BaseAgent, AgentInput, AgentOutput
from config.config import config
from agents.tools.product_tools import extract_product_id_from_query
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


class QueryClassification(BaseModel):
    """Structured representation of a query classification."""
    query_type: str = Field(description="The type of query (general, product_availability, etc.)")
    confidence: float = Field(description="Confidence score for this classification (0-1)")
    product_id: Optional[str] = Field(default=None, description="Product ID if detected")
    requires_db_lookup: bool = Field(default=True, description="Whether the query requires database lookup")


class QueryClassifierAgent(BaseAgent):
    """
    Agent that classifies user queries to determine which specialized agent should handle them.
    """
    
    def __init__(self, model_name: str = None, verbose: bool = False):
        """
        Initialize the query classifier agent.
        
        Args:
            model_name (str): Name of the LLM model to use
            verbose (bool): Whether to output verbose logs
        """
        super().__init__(name="query_classifier", verbose=verbose)
        
        model_name = model_name or config.openai.model
        
        self.llm = ChatOpenAI(
            api_key=config.openai.api_key,
            model=model_name,
            temperature=0.1  # Lower temperature for more deterministic classifications
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert query classifier for a MongoDB assistant system.
            
Your task is to analyze a user's query and determine its category. The categories are:

1. product_availability: ONLY for queries specifically about whether a SPECIFIC product is in stock, available, or when it will be available.
   Example: "Is product ABC123 in stock?" or "When will item XYZ789 be available?"
   MUST have a specific product identifier.

2. general_db_query: Any general database question, including listing, counting, or searching for products.
   Example: "Show me all users", "List products", "Find users with gmail email", "Show me the top 5 products"

Think step by step to determine the query type:

For product_availability:
- MUST ask about availability, stock status, or when a product will be available
- MUST mention a SPECIFIC product ID or identifier (like ABC123, SKU456, etc.)
- Just mentioning the word "product" is NOT enough
- Listing or showing products is NOT about availability

For general_db_query:
- All listing, counting, filtering queries
- Queries about users, orders, etc.
- General product queries without asking about specific availability
- Any query without a specific product identifier

Respond with JSON containing:
- query_type: The determined category (product_availability or general_db_query)
- confidence: Your confidence in this classification (0-1)
- product_id: Any product ID mentioned in the query (if applicable)
- requires_db_lookup: Whether this query needs database information (usually true)

Respond ONLY with a valid JSON object containing these fields.
"""),
            ("human", "{query}")
        ])
    
    async def run(self, inputs: AgentInput) -> AgentOutput:
        """
        Classify the type of query.
        
        Args:
            inputs (AgentInput): The agent inputs including the user query
            
        Returns:
            AgentOutput: The classification results
        """
        self.log(f"Classifying query: {inputs.query}")
        
        # First try to extract product ID directly
        product_id = extract_product_id_from_query(inputs.query)
        
        # More precise rule-based classification
        if product_id and any(keyword in inputs.query.lower() for keyword in 
                          ["in stock", "available", "availability", "when will"]):
            # Only classify as product availability if both specific product ID AND availability keywords are present
            self.log(f"Directly classified as product availability query. Product ID: {product_id}")
            
            classification = QueryClassification(
                query_type="product_availability",
                confidence=0.95,
                product_id=product_id,
                requires_db_lookup=True
            )
        elif any(list_keyword in inputs.query.lower() for list_keyword in 
              ["list", "show", "get", "display", "find", "search", "query"]):
            # General listing/querying operations should be general_db_query even if they mention products
            self.log("Directly classified as general db query (listing/searching operation)")
            
            classification = QueryClassification(
                query_type="general_db_query",
                confidence=0.9,
                product_id=None,
                requires_db_lookup=True
            )
        else:
            # Use LLM for more complex classification
            chain = self.prompt | self.llm
            result = await chain.ainvoke({"query": inputs.query})
            
            # Try to extract the classification from the response
            try:
                import json
                import re
                
                # Find JSON content in the response
                json_match = re.search(r'{.*}', result.content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    classification_data = json.loads(json_str)
                    
                    # Create classification with defaults for missing fields
                    classification = QueryClassification(
                        query_type=classification_data.get("query_type", "general_db_query"),
                        confidence=classification_data.get("confidence", 0.7),
                        product_id=classification_data.get("product_id"),
                        requires_db_lookup=classification_data.get("requires_db_lookup", True)
                    )
                else:
                    # Default to general query if we can't parse the response
                    self.log("Couldn't parse LLM response, defaulting to general query")
                    classification = QueryClassification(
                        query_type="general_db_query",
                        confidence=0.5,
                        requires_db_lookup=True
                    )
            except Exception as e:
                self.log(f"Error parsing classification: {str(e)}")
                classification = QueryClassification(
                    query_type="general_db_query",
                    confidence=0.5,
                    requires_db_lookup=True
                )
        
        self.log(f"Query classified as: {classification.query_type} (confidence: {classification.confidence})")
        
        return AgentOutput(
            response=f"Query classified as {classification.query_type}",
            data={"classification": classification.model_dump()},
            status="success"
        )
    
    def get_description(self) -> str:
        """
        Get a description of what this agent does.
        
        Returns:
            str: Description of the agent
        """
        return "Classifies user queries to determine which specialized agent should handle them." 