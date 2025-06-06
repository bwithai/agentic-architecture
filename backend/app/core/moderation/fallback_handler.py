"""
Fallback handler for customer support moderation.
This module handles cases where the chatbot cannot provide an accurate response.
"""

from utils import generate_fallback_email, send_email
from app.core.config import config

async def handle_fallback(user_query: str, chatbot_response: str, confidence_score: float) -> str:
    """
    Handle cases where the chatbot cannot provide an accurate response.
    This will send the query to customer support via email.
    
    Args:
        user_query: The original user question
        chatbot_response: The chatbot's attempted response
        confidence_score: The confidence score (0-1) of the response
        
    Returns:
        A message to display to the user
    """
    print(f"FALLBACK TRIGGERED: User query: {user_query}")
    print(f"FALLBACK TRIGGERED: Chatbot response: {chatbot_response}")
    print(f"FALLBACK TRIGGERED: Confidence score: {confidence_score}")
    
    # Generate and send email to support team
    email_data = generate_fallback_email(
        user_query=user_query,
        chatbot_response=chatbot_response,
        confidence_score=confidence_score
    )
    
    send_email(
        email_to=config.SUPPORT_EMAIL,
        subject=email_data.subject,
        html_content=email_data.html_content
    )
    
    return "I'm not confident I can provide an accurate answer to your question. I've forwarded your query to our customer support team who will review it and get back to you soon." 