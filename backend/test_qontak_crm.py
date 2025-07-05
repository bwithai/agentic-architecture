"""
Test script for Qontak CRM integration
Demonstrates sending messages, processing webhooks, and managing customers
"""

import asyncio
import json
from app.services.qontak_crm_service import qontak_crm


async def test_crm_functionality():
    """Test all CRM functionality"""
    print("🚀 Testing Qontak CRM Integration")
    print("=" * 50)
    
    # Test 1: Check configuration
    print("\n1️⃣ Checking Configuration...")
    print(f"   API Base: {qontak_crm.base_url}")
    print(f"   Integration ID: {qontak_crm.integration_id}")
    print(f"   WhatsApp Business ID: {qontak_crm.whatsapp_business_id}")
    print(f"   Access Token Available: {'✅' if qontak_crm.access_token else '❌'}")
    
    # Test 2: Refresh token (if client secret is available)
    print("\n2️⃣ Testing Token Refresh...")
    try:
        refresh_success = await qontak_crm.refresh_access_token()
        if refresh_success:
            print("   ✅ Token refresh successful")
        else:
            print("   ⚠️ Token refresh failed (may need QONTAK_CLIENT_SECRET)")
    except Exception as e:
        print(f"   ❌ Token refresh error: {e}")
    
    # Test 3: Simulate webhook message processing
    print("\n3️⃣ Testing Webhook Message Processing...")
    sample_webhook = {
        "message_id": "msg_12345",
        "conversation_id": "conv_67890",
        "from": {
            "phone": "1234567890",
            "name": "Test Customer"
        },
        "message": {
            "text": "Hello, I need help with my order",
            "type": "text"
        },
        "channel": "whatsapp",
        "timestamp": "2025-01-06T10:00:00Z"
    }
    
    try:
        message = await qontak_crm.process_webhook_message(sample_webhook)
        print(f"   ✅ Webhook processed: {message.customer_phone}")
        print(f"   📱 Customer: {message.customer_name}")
        print(f"   💬 Message: {message.message_text}")
    except Exception as e:
        print(f"   ❌ Webhook processing error: {e}")
    
    # Test 4: Send a test message (if access token is available)
    print("\n4️⃣ Testing Message Sending...")
    test_phone = "1234567890"  # Use the same phone from webhook test
    test_message = "Thank you for contacting us! How can we help you today?"
    
    try:
        send_success = await qontak_crm.send_whatsapp_message(
            phone=test_phone,
            message=test_message
        )
        if send_success:
            print("   ✅ Message sent successfully")
        else:
            print("   ⚠️ Message sending failed")
    except Exception as e:
        print(f"   ❌ Message sending error: {e}")
    
    # Test 5: Customer management
    print("\n5️⃣ Testing Customer Management...")
    try:
        # Get all customers
        customers = await qontak_crm.get_all_customers()
        print(f"   📊 Total customers: {len(customers)}")
        
        # Get specific customer
        customer = await qontak_crm.get_customer_history(test_phone)
        if customer:
            print(f"   👤 Customer found: {customer.name} ({customer.phone})")
            print(f"   📧 Message count: {len(customer.message_history)}")
            print(f"   🏷️ Tags: {customer.tags}")
        
        # Add tags to customer
        await qontak_crm.tag_customer(test_phone, ["test", "demo", "support"])
        print("   🏷️ Tags added to customer")
        
        # Search customers
        search_results = await qontak_crm.search_customers("Test")
        print(f"   🔍 Search results for 'Test': {len(search_results)} customers")
        
    except Exception as e:
        print(f"   ❌ Customer management error: {e}")
    
    # Test 6: Display conversation history
    print("\n6️⃣ Conversation History...")
    try:
        customer = await qontak_crm.get_customer_history(test_phone)
        if customer and customer.message_history:
            print(f"   📜 Conversation with {customer.name or customer.phone}:")
            for msg in customer.message_history[-5:]:  # Show last 5 messages
                direction = "📨" if msg["direction"] == "inbound" else "📤"
                print(f"   {direction} {msg['timestamp'][:19]}: {msg['text'][:50]}...")
        else:
            print("   📜 No conversation history found")
    except Exception as e:
        print(f"   ❌ History display error: {e}")
    
    print("\n✅ CRM Test Completed!")
    print("=" * 50)


async def test_webhook_endpoint():
    """Test webhook endpoint with sample data"""
    print("\n🔗 Testing Webhook Endpoint...")
    
    # This would normally be called by Qontak's webhook
    sample_webhook_data = {
        "message_id": "msg_webhook_test",
        "conversation_id": "conv_webhook_test",
        "from": {
            "phone": "9876543210",
            "name": "Webhook Test Customer"
        },
        "message": {
            "text": "This is a test message from webhook",
            "type": "text"
        },
        "channel": "whatsapp"
    }
    
    try:
        await qontak_crm.process_webhook_message(sample_webhook_data)
        print("   ✅ Webhook test successful")
    except Exception as e:
        print(f"   ❌ Webhook test error: {e}")


def print_setup_instructions():
    """Print setup instructions for the CRM"""
    print("\n📋 Qontak CRM Setup Instructions:")
    print("=" * 50)
    print("1. Get your QONTAK_CLIENT_SECRET from your client")
    print("2. Set environment variable: $env:QONTAK_CLIENT_SECRET = 'your-secret'")
    print("3. Configure webhook URL in Qontak dashboard:")
    print("   - Webhook URL: http://your-domain.com/api/v1/qontak/webhook")
    print("4. Start your FastAPI server: uvicorn server:app --reload")
    print("5. Test endpoints:")
    print("   - Health: GET /api/v1/qontak/health")
    print("   - Send Message: POST /api/v1/qontak/send-message")
    print("   - Get Customers: GET /api/v1/qontak/customers")
    print("   - Refresh Token: POST /api/v1/qontak/refresh-token")
    print("\n🔧 Available CRM Features:")
    print("- ✅ Receive webhook messages from Qontak")
    print("- ✅ Send WhatsApp messages")
    print("- ✅ Customer conversation history")
    print("- ✅ Customer search and tagging")
    print("- ✅ Automatic token refresh")
    print("- ✅ Message tracking (inbound/outbound)")


async def main():
    """Main test function"""
    print_setup_instructions()
    await test_crm_functionality()
    await test_webhook_endpoint()
    
    print("\n🎉 Your Qontak CRM is ready to use!")
    print("Remember to:")
    print("1. Set QONTAK_CLIENT_SECRET environment variable")
    print("2. Configure webhook URL in Qontak dashboard")
    print("3. Start your FastAPI server")


if __name__ == "__main__":
    asyncio.run(main()) 