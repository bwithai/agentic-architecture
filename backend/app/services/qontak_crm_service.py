"""
Qontak CRM Service
Handles webhook messages, sends WhatsApp messages, and manages customer interactions
Based on Qontak's Message Interaction Webhooks and WhatsApp Outbound Message Direct API
"""

import httpx
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from app.core.config import config


class QontakMessage(BaseModel):
    """Model for incoming Qontak messages from webhooks"""
    message_id: str
    conversation_id: str
    customer_phone: str
    customer_name: Optional[str] = None
    message_text: str
    message_type: str
    timestamp: datetime
    channel: str = "whatsapp"


class QontakCustomer(BaseModel):
    """Model for customer data"""
    phone: str
    name: Optional[str] = None
    conversation_id: Optional[str] = None
    last_interaction: datetime
    message_history: List[Dict[str, Any]] = []
    tags: List[str] = []
    status: str = "active"  # active, archived, blocked


class QontakCRMService:
    """Main CRM service for Qontak integration"""
    
    def __init__(self):
        self.base_url = config.QONTAK_API_BASE
        self.access_token = config.QONTAK_ACCESS_TOKEN
        self.integration_id = config.QONTAK_INTEGRATION_ID
        self.whatsapp_business_id = config.QONTAK_WHATSAPP_BUSINESS_ID
        self.customers: Dict[str, QontakCustomer] = {}
        
    async def refresh_access_token(self) -> bool:
        """Refresh access token using refresh token"""
        if not config.QONTAK_REFRESH_TOKEN or not config.QONTAK_CLIENT_SECRET:
            print("❌ Missing refresh token or client secret")
            return False
        
        oauth_payload = {
            "grant_type": "refresh_token",
            "refresh_token": config.QONTAK_REFRESH_TOKEN,
            "client_id": config.QONTAK_INTEGRATION_ID,
            "client_secret": config.QONTAK_CLIENT_SECRET
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/oauth/token",
                    data=oauth_payload,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    self.access_token = data.get("access_token")
                    print("✅ Access token refreshed successfully")
                    return True
                else:
                    print(f"❌ Token refresh failed: {response.status_code} - {response.text}")
                    return False
                    
            except Exception as e:
                print(f"❌ Error refreshing token: {e}")
                return False

    async def process_webhook_message(self, webhook_data: Dict[str, Any]) -> QontakMessage:
        """
        Process incoming webhook message from Qontak
        Based on Message Interaction Webhooks documentation
        """
        try:
            # Extract message data from webhook payload
            # Adjust these field names based on actual Qontak webhook structure
            message = QontakMessage(
                message_id=webhook_data.get("message_id", ""),
                conversation_id=webhook_data.get("conversation_id", ""),
                customer_phone=webhook_data.get("from", {}).get("phone", ""),
                customer_name=webhook_data.get("from", {}).get("name"),
                message_text=webhook_data.get("message", {}).get("text", ""),
                message_type=webhook_data.get("message", {}).get("type", "text"),
                timestamp=datetime.now(),
                channel=webhook_data.get("channel", "whatsapp")
            )
            
            # Update customer record
            await self.update_customer_record(message)
            
            print(f"📨 Processed message from {message.customer_phone}: {message.message_text[:50]}...")
            return message
            
        except Exception as e:
            print(f"❌ Error processing webhook message: {e}")
            raise
    
    async def update_customer_record(self, message: QontakMessage):
        """Update or create customer record"""
        phone = message.customer_phone
        
        if phone not in self.customers:
            self.customers[phone] = QontakCustomer(
                phone=phone,
                name=message.customer_name,
                conversation_id=message.conversation_id,
                last_interaction=message.timestamp,
                message_history=[],
                tags=[],
                status="active"
            )
        
        customer = self.customers[phone]
        customer.last_interaction = message.timestamp
        customer.conversation_id = message.conversation_id
        
        # Add message to history
        customer.message_history.append({
            "message_id": message.message_id,
            "text": message.message_text,
            "type": message.message_type,
            "timestamp": message.timestamp.isoformat(),
            "direction": "inbound"
        })
        
        # Keep only last 100 messages
        if len(customer.message_history) > 100:
            customer.message_history = customer.message_history[-100:]
    
    async def send_whatsapp_message(
        self, 
        phone: str, 
        message: str, 
        message_type: str = "text"
    ) -> bool:
        """
        Send WhatsApp message using Qontak Outbound Direct API
        Based on WhatsApp Outbound Message Direct API documentation
        """
        if not self.access_token:
            print("❌ No access token available")
            return False
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        # Format phone number (ensure it starts with country code)
        if not phone.startswith("+"):
            phone = f"+{phone}"
        
        payload = {
            "to_number": phone,
            "to_name": self.customers.get(phone.replace("+", ""), {}).get("name", "Customer"),
            "message_template_id": None,  # For direct text messages
            "channel_integration_id": self.integration_id,
            "language": {
                "code": "en"
            },
            "parameters": {
                "body": [
                    {
                        "key": "1",
                        "value": message
                    }
                ]
            }
        }
        
        # For direct text messages, use simpler payload
        if message_type == "text":
            payload = {
                "phone": phone.replace("+", ""),
                "message": message,
                "channel_integration_id": self.integration_id
            }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/whatsapp/v1/outbound_message",
                    json=payload,
                    headers=headers,
                    timeout=30.0
                )
                
                if response.status_code in [200, 201]:
                    print(f"✅ Message sent to {phone}: {message[:50]}...")
                    
                    # Update customer record with sent message
                    await self.update_sent_message_record(phone, message)
                    return True
                else:
                    print(f"❌ Failed to send message: {response.status_code} - {response.text}")
                    return False
                    
            except Exception as e:
                print(f"❌ Error sending message: {e}")
                return False
    
    async def update_sent_message_record(self, phone: str, message: str):
        """Update customer record with sent message"""
        phone_key = phone.replace("+", "")
        if phone_key in self.customers:
            self.customers[phone_key].message_history.append({
                "text": message,
                "type": "text",
                "timestamp": datetime.now().isoformat(),
                "direction": "outbound"
            })
    
    async def get_customer_history(self, phone: str) -> Optional[QontakCustomer]:
        """Get customer conversation history"""
        phone_key = phone.replace("+", "")
        return self.customers.get(phone_key)
    
    async def get_all_customers(self) -> List[QontakCustomer]:
        """Get all customers"""
        return list(self.customers.values())
    
    async def tag_customer(self, phone: str, tags: List[str]):
        """Add tags to customer"""
        phone_key = phone.replace("+", "")
        if phone_key in self.customers:
            for tag in tags:
                if tag not in self.customers[phone_key].tags:
                    self.customers[phone_key].tags.append(tag)
    
    async def search_customers(self, query: str) -> List[QontakCustomer]:
        """Search customers by name or phone"""
        results = []
        query_lower = query.lower()
        
        for customer in self.customers.values():
            if (query_lower in customer.phone.lower() or 
                (customer.name and query_lower in customer.name.lower())):
                results.append(customer)
        
        return results

    async def send_template_message(
        self,
        phone: str,
        template_id: str,
        parameters: Dict[str, Any]
    ) -> bool:
        """Send template-based WhatsApp message"""
        if not self.access_token:
            await self.refresh_access_token()
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "to_number": phone,
            "to_name": self.customers.get(phone.replace("+", ""), {}).get("name", "Customer"),
            "message_template_id": template_id,
            "channel_integration_id": self.integration_id,
            "language": {
                "code": "en"
            },
            "parameters": parameters
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/whatsapp/v1/send_message_template",
                    json=payload,
                    headers=headers,
                    timeout=30.0
                )
                
                if response.status_code in [200, 201]:
                    print(f"✅ Template message sent to {phone}")
                    return True
                else:
                    print(f"❌ Failed to send template: {response.status_code} - {response.text}")
                    return False
                    
            except Exception as e:
                print(f"❌ Error sending template: {e}")
                return False

# Global CRM service instance
qontak_crm = QontakCRMService() 