"""
Example usage of the Medical Expert Agent
This script demonstrates how to use the MedicalExpertAgent to gather patient information
through a conversational interface.
"""

import os
import sys

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.specialized.medical_expert_agent import MedicalExpertAgent


def main():
    """
    Example usage of the Medical Expert Agent
    """
    print("🏥 Medical Expert Agent Demo")
    print("=" * 50)
    
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
    
    # Initialize the agent with MongoDB client
    # Note: You'll need to set your OpenAI API key
    # You can do this by:
    # 1. Setting environment variable: OPENAI_API_KEY
    # 2. Or passing it directly: MedicalExpertAgent(openai_api_key="your-key-here")
    
    try:
        agent = MedicalExpertAgent(mongo_client=mongo_client)
        
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
            elif user_input.lower() == 'summary':
                print("\n📊", agent.get_conversation_summary())
                continue
            elif user_input.lower() == 'export':
                print("\n📄 Patient Data Export:")
                print(agent.export_patient_data())
                continue
            elif user_input.lower() == 'extract':
                print("\n🔍 Manually triggering LLM extraction...")
                result = agent.trigger_manual_extraction()
                if result.get("success"):
                    print("✅", result["message"])
                    print(agent.get_conversation_summary())
                else:
                    print("❌", result.get("message", "Extraction failed"))
                continue
            elif user_input.lower() == 'missing':
                missing_basic = agent._get_missing_basic_info()
                missing_all = agent._get_missing_information()
                print(f"\n📝 Missing Basic Info: {missing_basic}")
                print(f"📝 Missing All Info: {missing_all}")
                continue
            
            if user_input:
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
                    break
                
                # Check if information is complete
                if agent.is_information_complete():
                    print("\n✅ Basic patient information is complete!")
        
        print("\n👋 Thank you for using the Medical Expert Agent!")
        
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


def demo_conversation():
    """
    Demonstrate a sample conversation without user interaction
    """
    print("\n🎭 Demo Conversation (Simulated)")
    print("=" * 50)
    
    # Extended sample conversation flow to trigger LLM extraction
    sample_inputs = [
        "Hi, I'm not feeling well",
        "My name is Sarah Johnson",
        "I'm 28 years old",
        "I'm a woman",
        "I have a severe headache and I'm feeling nauseous",
        "The headache started yesterday morning and it's getting progressively worse",
        "I also feel dizzy when I stand up quickly",
        "I take birth control pills daily and sometimes ibuprofen for pain",
        "I had migraines as a teenager but they stopped for years",
        "The pain is throbbing and mainly on the right side of my head"
    ]
    
    try:
        agent = MedicalExpertAgent()
        
        # Start conversation
        print("🤖 Agent:", agent.start_conversation())
        
        # Process sample inputs
        for i, user_input in enumerate(sample_inputs, 1):
            print(f"\n👤 User {i}: {user_input}")
            result = agent.process_user_input(user_input)
            print(f"🤖 Agent: {result['response']}")
            print(f"🔄 Flow: {result['flow_action']}")
            
            # Show when LLM extraction is triggered
            if result['extraction_performed'] and i <= 8:
                print("\n" + "🔍" * 50)
                print(f"LLM EXTRACTION TRIGGERED DYNAMICALLY AFTER {i} QA PAIRS!")
                print("🔍" * 50)
            
            # Check if conversation ended
            if result['conversation_ended']:
                print(f"\n🏁 Conversation ended after {i} inputs.")
                break
        
        # Show final summary
        print("\n" + "=" * 50)
        print("📊 Final Patient Information:")
        print(agent.get_conversation_summary())
        
        print("\n📄 Complete Patient Data:")
        patient_data = agent.get_patient_info()
        print(f"Name: {patient_data['name']}")
        print(f"Age: {patient_data['age']}")
        print(f"Gender: {patient_data['gender']}")
        print(f"Symptoms: {', '.join(patient_data['symptoms'])}")
        print(f"Medical History: {', '.join(patient_data['medical_history'])}")
        print(f"Medications: {', '.join(patient_data['medications'])}")
        print(f"QA Pairs Count: {patient_data['qa_pairs_count']}")
        print(f"LLM Extraction Performed: {patient_data['extraction_performed']}")
        print(f"Timestamp: {patient_data['timestamp']}")
        
    except Exception as e:
        print(f"❌ Demo Error: {e}")
        print("This is just a demo - the actual agent requires OpenAI API key")


def demo_llm_extraction():
    """
    Demonstrate the LLM-based extraction feature specifically
    """
    print("\n🧠 LLM Extraction Demo")
    print("=" * 50)
    print("This demo shows how the agent uses dynamic LLM extraction based on")
    print("medical content density rather than fixed QA pair counts.")
    print("=" * 50)
    
    # Complex conversation that would be hard to extract with simple patterns
    complex_inputs = [
        "Hello doctor, I've been having some health issues lately",
        "Oh, I'm Jennifer Martinez, nice to meet you",
        "I just turned 35 last month, so I'm 35 years old",
        "I'm a female, and I work as a software engineer",
        "Well, I've been experiencing these terrible migraines that come and go",
        "The pain usually starts behind my left eye and spreads across my forehead",
        "I also get really sensitive to light and sometimes feel like throwing up",
        "I'm currently on Lexapro for anxiety, 10mg daily, and I take vitamin D supplements",
        "My mother had chronic migraines too, and I had a concussion playing soccer in college",
        "The headaches seem worse when I'm stressed at work or don't get enough sleep"
    ]
    
    try:
        agent = MedicalExpertAgent()
        
        print("🤖 Starting conversation...")
        print("🤖 Agent:", agent.start_conversation())
        
        print(f"\n📝 Processing {len(complex_inputs)} user inputs...")
        print("🔍 LLM extraction will be triggered dynamically based on medical content\n")
        
        for i, user_input in enumerate(complex_inputs, 1):
            print(f"👤 Input {i}: {user_input}")
            result = agent.process_user_input(user_input)
            response = result['response']
            
            # Show abbreviated response for demo
            if len(response) > 150:
                print(f"🤖 Agent: {response[:150]}...")
            else:
                print(f"🤖 Agent: {response}")
            
            print(f"🔄 Flow: {result['flow_action']}")
            
            # Highlight when LLM extraction happens
            if result['extraction_performed'] and i <= 8:
                print("\n" + "🚀" * 60)
                print(f"🧠 DYNAMIC LLM EXTRACTION TRIGGERED AT INPUT {i}!")
                print("🚀" * 60)
            
            # Check if conversation ended
            if result['conversation_ended']:
                print(f"\n🏁 Conversation ended after {i} inputs.")
                break
            
            print()
        
        # Show extracted information
        print("=" * 60)
        print("🎯 EXTRACTION RESULTS:")
        print("=" * 60)
        print(agent.get_conversation_summary())
        
        # Show what the LLM was able to extract vs simple patterns
        patient_data = agent.get_patient_info()
        print(f"\n🔍 Advanced Information Extracted by LLM:")
        print(f"   • Full Name: {patient_data['name']}")
        print(f"   • Detailed Symptoms: {patient_data['symptoms']}")
        print(f"   • Medical History: {patient_data['medical_history']}")
        print(f"   • Medications: {patient_data['medications']}")
        print(f"   • Additional Info: {patient_data['additional_info']}")
        
        print(f"\n📊 Extraction Statistics:")
        print(f"   • Total QA Pairs: {patient_data['qa_pairs_count']}")
        print(f"   • LLM Extraction Used: {'✅ Yes' if patient_data['extraction_performed'] else '❌ No'}")
        
    except Exception as e:
        print(f"❌ Demo Error: {e}")
        print("This demo requires OpenAI API key to show LLM extraction")


if __name__ == "__main__":
    print("🏥 Medical Expert Agent - Choose an option:")
    print("1. Interactive conversation (requires OpenAI API key)")
    print("2. Demo conversation (simulated)")
    print("3. LLM Extraction Demo (shows advanced extraction features)")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        demo_conversation()
    elif choice == "3":
        demo_llm_extraction()
    else:
        print("Invalid choice. Running LLM extraction demo...")
        demo_llm_extraction() 