import os
import asyncio
import sys
from dotenv import load_dotenv

from agents.tools.registry import ToolRegistry
from mongodb.mongodb_setup import setup_mongodb
from mongodb.client import MongoDBClient
from agents.specialized.medical_expert_agent import MedicalExpertAgent

async def run_main_bot():
    """Initialize and run the chatbot."""
    # Load environment variables
    load_dotenv()

    # Get required configuration
    mongodb_url = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/test")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        return
    
    # Setup MongoDB client for database saving
    mongo_client = None
    try:
        from pymongo import MongoClient
        from pymongo.errors import ConnectionFailure
        
        print("🔌 Connecting to MongoDB...")
        mongo_client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
        mongo_client.admin.command('ping')
        print("✅ MongoDB connection successful! Patient data will be saved automatically.")
    except ConnectionFailure:
        print("⚠️  MongoDB connection failed. Patient data will not be saved to database.")
        print("   To enable database saving, ensure MongoDB is running on localhost:27017")
        mongo_client = None
    except ImportError:
        print("⚠️  pymongo not installed. Patient data will not be saved to database.")
        print("   Install with: pip install pymongo")
        mongo_client = None
    except Exception as e:
        print(f"⚠️  MongoDB setup error: {e}")
        print("   Patient data will not be saved to database.")
        mongo_client = None
        
     # Run the chatbot interaction loop
    print("\nMain Bot Ready!")
    try:
        # Initialize the agent
        agent = MedicalExpertAgent(mongo_client=mongo_client)
        MEDICAL_EXPERT_AGENT_TASK_COMPLETE = False
        
        # Start the conversation
        print("\n🤖 Agent:", agent.start_conversation())
        
        # Simulate a conversation
        print("\n" + "=" * 50)
        print("💬 Interactive Conversation Demo")
        print("Type 'quit' to exit, 'summary' to see patient info")
        print("Type 'extract' to manually trigger LLM extraction")
        print("Type 'missing' to see missing basic information")
        print("=" * 50)

        while True:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() == 'quit':
                break

            if user_input:
                if not MEDICAL_EXPERT_AGENT_TASK_COMPLETE:
                    result = agent.process_user_input(user_input)
                    print(f"\n🤖 Agent: {result['response']}")
                    # Show flow information
                    print(f"\n🔄 Flow: {result['flow_action']} - {result['flow_reason']}")

                    # Check if conversation ended
                    if result['conversation_ended']:
                        print(f"\n🎯 CONVERSATION ENDED DETECTED!")
                        
                        save_result = result.get('database_save_result')
                        if save_result:
                            print(f"💾 Database Save Attempted: {save_result}")
                            if save_result['success']:
                                print(f"✅ Patient saved with ID: {save_result['patient_id']}")
                            else:
                                print(f"❌ Save failed: {save_result['message']}")
                        else:
                            print("❌ No database save result returned!")
                        MEDICAL_EXPERT_AGENT_TASK_COMPLETE = True

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure you have:")
        print("1. Installed required packages: pip install langchain openai")
        print("2. Set your OpenAI API key as environment variable: OPENAI_API_KEY")
    
    finally:
        # Clean up MongoDB connection
        if mongo_client:
            try:
                mongo_client.close()
                print("🔌 MongoDB connection closed.")
            except Exception as e:
                print(f"⚠️  Error closing MongoDB connection: {e}")


if __name__ == "__main__":
    asyncio.run(run_main_bot())