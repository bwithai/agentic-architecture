"""
Product Availability Agent

This agent is responsible for checking product availability and providing appropriate
responses, including escalation to admin when necessary.
"""

from typing import Dict, Any, List, Optional
import json
from datetime import datetime
from pydantic import BaseModel, Field

from agents.base.base_agent import BaseAgent, AgentInput, AgentOutput
from config.config import config
from agents.tools.product_tools import (
    check_product_availability,
    escalate_to_admin,
    extract_product_id_from_query
)
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


# Custom JSON encoder to handle MongoDB types like datetime
class MongoJSONEncoder(json.JSONEncoder):
    """Custom JSON Encoder that can handle MongoDB types like datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class ProductAvailabilityAgent(BaseAgent):
    """
    Agent that checks product availability and handles responses appropriately.
    
    This agent follows a workflow:
    1. Extract product ID from the query
    2. Check availability in MongoDB
    3. If product found, respond with availability details
    4. If not found or error, escalate to admin and inform user
    """
    
    def __init__(self, model_name: str = None, verbose: bool = False):
        """
        Initialize the product availability agent.
        
        Args:
            model_name (str): Name of the LLM model to use
            verbose (bool): Whether to output verbose logs
        """
        super().__init__(name="product_availability", verbose=verbose)
        
        model_name = model_name or config.openai.model
        
        self.llm = ChatOpenAI(
            api_key=config.openai.api_key,
            model=model_name,
            temperature=0.1  # Lower temperature for more deterministic responses
        )
        
        # Prompt for generating responses based on availability data
        self.availability_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful product availability assistant.
            
Your task is to provide clear information about product availability based on the data provided.
Be factual and precise - DO NOT make up information that is not in the data.

If the product is in stock:
- Mention the product name and that it's available
- Include the current quantity if available
- Be enthusiastic and positive

If the product is out of stock but has an available date:
- Apologize for the inconvenience
- Mention when the product is expected to be available again
- Suggest that the customer can place an order for when it becomes available

If the product was not found or there was an error:
- DO NOT make up information about the product
- DO NOT guess about its availability
- Instead, provide the standard response: "I've forwarded your query to our administrator team, and we'll contact you soon via email."

Be conversational but concise.
"""),
            ("human", """
Query: {query}

Product ID: {product_id}

Availability Data: 
{availability_data}

Escalation Status:
{escalation_status}

Please provide an appropriate response:
""")
        ])
    
    async def run(self, inputs: AgentInput) -> AgentOutput:
        """
        Handle a product availability query.
        
        Args:
            inputs (AgentInput): The agent inputs including the user query
            
        Returns:
            AgentOutput: The agent's response
        """
        self.log(f"Processing product availability query: {inputs.query}")
        
        # Step 1: Extract product ID from query
        product_id = extract_product_id_from_query(inputs.query)
        
        if not product_id:
            self.log("No product ID found in query, attempting to extract from context")
            # Check if product_id is provided in context
            product_id = inputs.context.get("product_id")
        
        # Step 2: If no product ID found, we need to escalate
        if not product_id:
            self.log("No product ID could be determined, escalating to admin")
            
            # Prepare request details for escalation
            request_details = {
                "query": inputs.query,
                "reason": "Unable to extract product ID",
                "source": "product_availability_agent"
            }
            
            # Escalate to admin
            escalation_result = escalate_to_admin(request_details)
            
            return AgentOutput(
                response="I've forwarded your query to our administrator team, and we'll contact you soon via email.",
                data={
                    "escalation": escalation_result,
                    "reason": "No product ID found"
                },
                status="escalated"
            )
        
        # Step 3: Check product availability
        self.log(f"Checking availability for product ID: {product_id}")
        availability = check_product_availability(product_id)
        
        # Step 4: Handle based on availability result
        if availability["status"] == "found":
            self.log(f"Product found: {availability['product_name']}")
            
            # Generate response based on availability data
            chain = self.availability_prompt | self.llm
            result = await chain.ainvoke({
                "query": inputs.query,
                "product_id": product_id,
                "availability_data": json.dumps(availability, indent=2, cls=MongoJSONEncoder),
                "escalation_status": "No escalation needed"
            })
            
            return AgentOutput(
                response=result.content,
                data={"availability": availability},
                status="success"
            )
        
        else:
            # Product not found or error, escalate to admin
            self.log(f"Product not found or error: {availability['message']}")
            
            # Prepare request details for escalation
            request_details = {
                "query": inputs.query,
                "product_id": product_id,
                "availability_result": availability,
                "source": "product_availability_agent"
            }
            
            # Escalate to admin
            escalation_result = escalate_to_admin(request_details)
            
            # Generate appropriate response
            chain = self.availability_prompt | self.llm
            result = await chain.ainvoke({
                "query": inputs.query,
                "product_id": product_id,
                "availability_data": json.dumps(availability, indent=2, cls=MongoJSONEncoder),
                "escalation_status": "Escalated to admin"
            })
            
            return AgentOutput(
                response=result.content,
                data={
                    "availability": availability,
                    "escalation": escalation_result
                },
                status="escalated"
            )
    
    def get_description(self) -> str:
        """
        Get a description of what this agent does.
        
        Returns:
            str: Description of the agent
        """
        return "Checks product availability in the database and provides appropriate responses, escalating to admin when necessary." 