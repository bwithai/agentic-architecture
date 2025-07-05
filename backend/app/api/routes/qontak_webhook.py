"""
Qontak Webhook and CRM API endpoints
Handles incoming webhooks and provides CRM functionality
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from datetime import datetime

from app.services.qontak_crm_service import qontak_crm, QontakCustomer

router = APIRouter(prefix="/qontak", tags=["qontak-crm"])


class WebhookPayload(BaseModel):
    """Model for incoming webhook data"""
    message_id: Optional[str] = None
    conversation_id: Optional[str] = None
    message: Optional[Dict[str, Any]] = None
    from_: Optional[Dict[str, str]] = None  # 'from' is a reserved keyword
    channel: Optional[str] = "whatsapp"
    timestamp: Optional[str] = None


class SendMessageRequest(BaseModel):
    """Model for sending message requests"""
    phone: str
    message: str
    message_type: str = "text"


class TagCustomerRequest(BaseModel):
    """Model for tagging customers"""
    phone: str
    tags: List[str]


@router.post("/webhook")
async def receive_webhook(payload: WebhookPayload, background_tasks: BackgroundTasks):
    """
    Receive and process Qontak webhooks
    Based on Message Interaction Webhooks documentation
    """
    try:
        # Convert Pydantic model to dict for processing
        webhook_data = payload.dict()
        
        # Handle the 'from' field name conflict
        if payload.from_:
            webhook_data["from"] = payload.from_
        
        # Process the webhook message in background
        background_tasks.add_task(process_webhook_background, webhook_data)
        
        return {
            "status": "success",
            "message": "Webhook received and being processed"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing webhook: {str(e)}")


async def process_webhook_background(webhook_data: Dict[str, Any]):
    """Background task to process webhook"""
    try:
        await qontak_crm.process_webhook_message(webhook_data)
    except Exception as e:
        print(f"❌ Error in background webhook processing: {e}")


@router.post("/send-message")
async def send_message(request: SendMessageRequest):
    """
    Send WhatsApp message to customer
    Based on WhatsApp Outbound Message Direct API documentation
    """
    try:
        success = await qontak_crm.send_whatsapp_message(
            phone=request.phone,
            message=request.message,
            message_type=request.message_type
        )
        
        if success:
            return {
                "status": "success",
                "message": f"Message sent to {request.phone}"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to send message")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sending message: {str(e)}")


@router.get("/customers")
async def get_all_customers():
    """Get all customers with their conversation history"""
    try:
        customers = await qontak_crm.get_all_customers()
        return {
            "status": "success",
            "customers": [customer.dict() for customer in customers],
            "total": len(customers)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching customers: {str(e)}")


@router.get("/customers/{phone}")
async def get_customer_history(phone: str):
    """Get specific customer conversation history"""
    try:
        customer = await qontak_crm.get_customer_history(phone)
        
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        return {
            "status": "success",
            "customer": customer.dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching customer: {str(e)}")


@router.get("/customers/search/{query}")
async def search_customers(query: str):
    """Search customers by name or phone number"""
    try:
        customers = await qontak_crm.search_customers(query)
        return {
            "status": "success",
            "customers": [customer.dict() for customer in customers],
            "total": len(customers),
            "query": query
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching customers: {str(e)}")


@router.post("/customers/tag")
async def tag_customer(request: TagCustomerRequest):
    """Add tags to a customer"""
    try:
        await qontak_crm.tag_customer(request.phone, request.tags)
        return {
            "status": "success",
            "message": f"Tags {request.tags} added to customer {request.phone}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error tagging customer: {str(e)}")


@router.post("/refresh-token")
async def refresh_access_token():
    """Manually refresh the Qontak access token"""
    try:
        success = await qontak_crm.refresh_access_token()
        
        if success:
            return {
                "status": "success",
                "message": "Access token refreshed successfully"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to refresh token")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error refreshing token: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "qontak-crm",
        "timestamp": datetime.now().isoformat(),
        "customers_count": len(qontak_crm.customers)
    } 