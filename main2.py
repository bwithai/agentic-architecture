import os
import asyncio
import sys
import json
from dotenv import load_dotenv
from datetime import datetime
from agents.tools.registry import ToolRegistry
from mongodb.mongodb_setup import setup_mongodb
from mongodb.client import MongoDBClient
from agents.specialized.medical_expert_agent import MedicalExpertAgent
from agents.specialized.pharmacist_agent import PharmacistAgent
from utils import generate_fallback_email, send_email, generate_medical_consultation_email
from config.config import config

class MedicalSystemFlow:
    """Enhanced medical system flow with complete patient journey"""
    
    def __init__(self):
        self.patient_id = None
        self.patient_data = None
        self.recommendations = None
        self.conversation_summary = []
        self.mongo_client = None
        self.mongodb_client = None
        self.medical_expert_agent = None
        self.pharmacist_agent = None
        
    async def initialize_system(self):
        """Initialize all system components"""
        # Load environment variables
        load_dotenv()

        # Get required configuration
        mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        database_name = os.getenv("MONGODB_DATABASE", "default_database")
        openai_api_key = os.environ.get("OPENAI_API_KEY")

        if not openai_api_key:
            print("ERROR: OPENAI_API_KEY environment variable not set")
            return False
        
        # Setup MongoDB client for database saving
        try:
            from pymongo import MongoClient
            from pymongo.errors import ConnectionFailure
            
            print("🔌 Connecting to MongoDB...")
            self.mongo_client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
            self.mongo_client.admin.command('ping')
            print("✅ MongoDB connection successful! Patient data will be saved automatically.")
        except ConnectionFailure:
            print("⚠️  MongoDB connection failed. Patient data will not be saved to database.")
            print("   To enable database saving, ensure MongoDB is running on localhost:27017")
            self.mongo_client = None
        except ImportError:
            print("⚠️  pymongo not installed. Patient data will not be saved to database.")
            print("   Install with: pip install pymongo")
            self.mongo_client = None
        except Exception as e:
            print(f"⚠️  MongoDB setup error: {e}")
            print("   Patient data will not be saved to database.")
            self.mongo_client = None
            
        # Initialize medical expert agent
        self.medical_expert_agent = MedicalExpertAgent(mongo_client=self.mongo_client)
        
        # Setup MongoDB client for pharmacist
        self.mongodb_client = MongoDBClient(mongodb_uri, database_name)
        
        # Initialize pharmacist agent
        self.pharmacist_agent = PharmacistAgent(
            openai_api_key=openai_api_key,
            mongodb_client=self.mongodb_client
        )
        
        return True
        
    async def step1_patient_consultation(self):
        """Step 1: Medical Expert Agent collects patient information"""
        print("\n" + "=" * 60)
        print("🏥 STEP 1: MEDICAL EXPERT CONSULTATION")
        print("=" * 60)
        print("\n🤖 Agent:", self.medical_expert_agent.start_conversation())
        
        print("\n" + "=" * 50)
        print("💬 Interactive Conversation Demo")
        print("Type 'quit' to exit, 'summary' to see patient info")
        print("Type 'extract' to manually trigger LLM extraction")
        print("Type 'missing' to see missing basic information")
        print("=" * 50)

        conversation_ended = False
        while not conversation_ended:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Thank you for using our medical system!")
                return False

            if user_input:
                result = self.medical_expert_agent.process_user_input(user_input)
                print(f"\n🤖 Agent: {result['response']}")
                
                # Add to conversation summary
                self.conversation_summary.append({
                    "timestamp": datetime.now().isoformat(),
                    "user_input": user_input,
                    "agent_response": result['response'],
                    "flow_action": result['flow_action'],
                    "flow_reason": result['flow_reason']
                })
                
                # Show flow information
                print(f"\n🔄 Flow: {result['flow_action']} - {result['flow_reason']}")

                # Check if conversation ended
                if result['conversation_ended']:
                    print(f"\n🎯 MEDICAL CONSULTATION COMPLETED!")
                    
                    save_result = result.get('database_save_result')
                    if save_result and save_result['success']:
                        self.patient_id = save_result['patient_id']
                        print(f"✅ Patient saved with ID: {save_result['patient_id']}")
                        conversation_ended = True
                    else:
                        print(f"❌ Save failed: {save_result.get('message', 'Unknown error')}")
                        return False
                        
        return True
        
    async def step2_pharmacist_analysis(self):
        """Step 2: Pharmacist analyzes patient and recommends products"""
        print("\n" + "=" * 60)
        print("💊 STEP 2: PHARMACIST PRODUCT ANALYSIS")
        print("=" * 60)
        
        # Create database indexes for optimal performance
        print("\n🔧 Optimizing database for product searches...")
        index_results = await self.pharmacist_agent.create_product_search_indexes()
        print(f"✅ Created {index_results.get('successful_indexes', 0)}/{index_results.get('total_indexes', 0)} database indexes")

        # Fetch patient data
        print(f"\n📋 Fetching patient data for ID: {self.patient_id}")
        self.patient_data = await self.pharmacist_agent.get_patient_by_id(self.patient_id)

        if "error" in self.patient_data:
            print(f"❌ Error fetching patient: {self.patient_data['error']}")
            return False

        # Display patient summary
        print("\n" + "=" * 40)
        print("👤 PATIENT INFORMATION SUMMARY")
        print("=" * 40)
        patient_summary = self.pharmacist_agent.get_patient_summary(self.patient_data)
        print(patient_summary)

        # Generate product recommendations
        print(f"\n🧠 Generating AI-powered product recommendations...")
        print("This may take a few moments as we analyze products...")

        self.recommendations = await self.pharmacist_agent.generate_product_recommendations(
            patient_data=self.patient_data,
            max_recommendations=5
        )

        if "error" in self.recommendations:
            print(f"❌ Error generating recommendations: {self.recommendations['error']}")
            return False
        
        # Display results
        print("\n" + "=" * 50)
        print("🎯 PRODUCT RECOMMENDATION RESULTS")
        print("=" * 50)
        print(f"📊 Total products analyzed: {self.recommendations['total_products_analyzed']}")
        print(f"⏰ Analysis timestamp: {self.recommendations['analysis_timestamp']}")
        
        print("\n🏆 TOP RECOMMENDATIONS:")
        for i, rec in enumerate(self.recommendations['recommendations'], 1):
            print(f"\n{i}. 💊 {rec['product_name']}")
            print(f"   📈 Recommendation Score: {rec['recommendation_score']:.2f}/1.00")
            print(f"   🎯 Confidence: {rec['symptom_match']['confidence']}")
            print(f"   📂 Category: {rec['additional_factors'].get('product_category', 'Unknown')}")
            print(f"   💰 Price: ${rec['additional_factors'].get('selling_price', 0):,}")
            print(f"   🔍 Reasoning: {rec['symptom_match']['reasoning']}")
            
            if rec['symptom_match']['matched_symptoms']:
                print(f"   ✅ Matched Symptoms: {', '.join(rec['symptom_match']['matched_symptoms'])}")
        
        print("\n" + "=" * 50)
        print("📋 PHARMACIST CONSULTATION REPORT")
        print("=" * 50)
        print(self.recommendations['consultation_report'])
        
        return True
        
    async def step3_user_choice(self):
        """Step 3: Ask user what they want to do with recommendations"""
        print("\n" + "=" * 60)
        print("🤔 STEP 3: WHAT WOULD YOU LIKE TO DO?")
        print("=" * 60)
        
        choice_made = False
        user_choice = None
        
        while not choice_made:
            print("\nBased on our analysis and recommendations, what would you like to do?")
            print("\n1. 🛒 Purchase/Order the recommended products")
            print("2. 🩺 Connect with a nurse for guidance with the products")
            print("3. 📞 Speak with customer support for more options")
            print("4. 📧 Send all information to my email")
            print("5. ❌ Exit without taking action")
            
            try:
                choice = input("\n👤 Please enter your choice (1-5): ").strip()
                
                if choice in ['1', '2', '3', '4', '5']:
                    user_choice = choice
                    choice_made = True
                else:
                    print("❌ Invalid choice. Please enter a number between 1-5.")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                return "exit"
                
        # Process the choice
        if user_choice == '1':
            print("\n🛒 PRODUCT ORDERING SELECTED")
            print("Redirecting you to our ordering system...")
            print("📋 Your recommendations will be prepared for ordering.")
            return "order_products"
            
        elif user_choice == '2':
            print("\n🩺 NURSE CONSULTATION SELECTED")
            print("Connecting you with our nursing team...")
            print("💊 A nurse will guide you on how to use the recommended products safely.")
            return "nurse_consultation"
            
        elif user_choice == '3':
            print("\n📞 CUSTOMER SUPPORT SELECTED")
            print("Transferring you to our customer support team...")
            return "customer_support"
            
        elif user_choice == '4':
            print("\n📧 EMAIL SUMMARY SELECTED")
            print("Preparing comprehensive summary for email...")
            return "email_summary"
            
        elif user_choice == '5':
            print("\n❌ EXIT SELECTED")
            print("Thank you for using our medical system.")
            return "exit"
            
    async def step4_send_summary_email(self, user_choice):
        """Step 4: Send comprehensive summary to customer support"""
        print("\n" + "=" * 60)
        print("📧 STEP 4: SENDING COMPREHENSIVE SUMMARY")
        print("=" * 60)
        
        try:
            # Prepare comprehensive summary
            patient_info = self.patient_data.get("patient", {})
            
            # Format recommendations for email
            formatted_recommendations = []
            for i, rec in enumerate(self.recommendations['recommendations'], 1):
                formatted_recommendations.append(f"""
{i}. {rec['product_name']}
   - Score: {rec['recommendation_score']:.2f}/1.00
   - Confidence: {rec['symptom_match']['confidence']}
   - Category: {rec['additional_factors'].get('product_category', 'Unknown')}
   - Price: ${rec['additional_factors'].get('selling_price', 0):,}
   - Reasoning: {rec['symptom_match']['reasoning']}
   - Matched Symptoms: {', '.join(rec['symptom_match']['matched_symptoms']) if rec['symptom_match']['matched_symptoms'] else 'None'}
                """)
            
            # Create comprehensive summary
            summary_content = f"""
MEDICAL SYSTEM CONSULTATION SUMMARY
=====================================

PATIENT INFORMATION:
- Name: {patient_info.get('name', 'Unknown')}
- Age: {patient_info.get('age', 'Unknown')}
- Gender: {patient_info.get('gender', 'Unknown')}
- Patient ID: {self.patient_id}

SYMPTOMS REPORTED:
{chr(10).join(f"- {symptom}" for symptom in patient_info.get('symptoms', [])) or "- No symptoms recorded"}

MEDICAL ANALYSIS:
- Total products analyzed: {self.recommendations['total_products_analyzed']}
- Analysis timestamp: {self.recommendations['analysis_timestamp']}

TOP PRODUCT RECOMMENDATIONS:
{''.join(formatted_recommendations)}

PHARMACIST CONSULTATION REPORT:
{self.recommendations['consultation_report'][:1000]}...

USER CHOICE:
{self._get_choice_description(user_choice)}

CONVERSATION FLOW:
{chr(10).join(f"- {entry['timestamp']}: {entry['flow_action']} - {entry['flow_reason']}" for entry in self.conversation_summary[-5:])}

NEXT ACTIONS REQUIRED:
- Follow up with patient based on their choice
- Review recommendations with medical team
- Ensure proper product availability
- Document consultation in patient records
            """
            
            # Generate email using existing email system
            email_data = generate_medical_consultation_email(
                patient_name=patient_info.get('name', 'Unknown Patient'),
                patient_id=self.patient_id,
                user_choice_description=self._get_choice_description(user_choice),
                summary_content=summary_content,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            
            # Send email to customer support
            send_email(
                email_to=config.email.SUPPORT_EMAIL,
                subject=email_data.subject,
                html_content=email_data.html_content
            )
            
            print("✅ Comprehensive summary sent to customer support team!")
            print(f"📧 Email sent to: {config.email.SUPPORT_EMAIL}")
            print("🔄 Our team will review and follow up within 24 hours.")
            
            return True
            
        except Exception as e:
            print(f"❌ Error sending email: {e}")
            print("📞 Please contact customer support directly for assistance.")
            return False
            
    def _get_choice_description(self, choice):
        """Get human-readable description of user choice"""
        choices = {
            "order_products": "Patient wants to purchase/order recommended products",
            "nurse_consultation": "Patient wants nurse guidance with product usage",
            "customer_support": "Patient wants to speak with customer support",
            "email_summary": "Patient requested email summary",
            "exit": "Patient exited without taking action"
        }
        return choices.get(choice, "Unknown choice")
        
    async def cleanup(self):
        """Clean up system resources"""
        if self.mongo_client:
            try:
                self.mongo_client.close()
                print("🔌 MongoDB connection closed.")
            except Exception as e:
                print(f"⚠️  Error closing MongoDB connection: {e}")

async def run_main_bot():
    """Main function to run the enhanced medical system flow"""
    medical_system = MedicalSystemFlow()
    
    try:
        print("🚀 INITIALIZING ADVANCED MEDICAL CONSULTATION SYSTEM")
        print("=" * 70)
        
        # Initialize system
        if not await medical_system.initialize_system():
            return
            
        # Step 1: Medical Expert Consultation
        if not await medical_system.step1_patient_consultation():
            return
            
        # Step 2: Pharmacist Analysis and Recommendations  
        if not await medical_system.step2_pharmacist_analysis():
            return
            
        # Step 3: User Choice
        user_choice = await medical_system.step3_user_choice()
        if user_choice == "exit":
            return
            
        # Step 4: Send Summary Email (always send, regardless of choice)
        await medical_system.step4_send_summary_email(user_choice)
        
        print("\n" + "=" * 70)
        print("✅ MEDICAL CONSULTATION SYSTEM COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"🎯 Patient journey completed with choice: {medical_system._get_choice_description(user_choice)}")
        print(f"📧 Summary sent to customer support for follow-up")
        print(f"⏰ Session ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ System Error: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Ensure MongoDB is running on localhost:27017")
        print("2. Verify OPENAI_API_KEY is set in environment variables")
        print("3. Check email configuration in config/config.py")
        import traceback
        traceback.print_exc()
    
    finally:
        await medical_system.cleanup()

if __name__ == "__main__":
    asyncio.run(run_main_bot())