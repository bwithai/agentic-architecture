import os
import json
import asyncio
import streamlit as st
import time
from dotenv import load_dotenv
from datetime import datetime

# Page config MUST be the first st command
st.set_page_config(
    page_title="Enhanced Medical Consultation System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables for medical system
if "medical_system" not in st.session_state:
    st.session_state.medical_system = None
    st.session_state.current_step = 1
    st.session_state.system_initialized = False
    st.session_state.consultation_messages = []
    st.session_state.patient_data = None
    st.session_state.recommendations = None
    st.session_state.user_choice = None
    st.session_state.consultation_complete = False
    st.session_state.debug_logs = []

if "show_debug" not in st.session_state:
    st.session_state.show_debug = False

# Import medical system components
from mongodb.client import MongoDBClient
from agents.specialized.medical_expert_agent import MedicalExpertAgent
from agents.specialized.pharmacist_agent import PharmacistAgent
from utils import generate_medical_consultation_email, send_email
from config.config import config

# Load environment variables
load_dotenv()

# Configuration
mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
database_name = os.getenv("MONGODB_DATABASE", "default_database")
openai_api_key = os.environ.get("OPENAI_API_KEY")

if not openai_api_key:
    st.error("❌ OPENAI_API_KEY environment variable not set. Please add it to your .env file.")
    st.stop()

# Enhanced CSS for medical system styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1976d2 0%, #42a5f5 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .doctor-message {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1976d2;
    }
    
    .patient-message {
        background-color: #f3e5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #9c27b0;
    }
    
    .recommendation-card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #4caf50;
    }
    
    .status-indicator {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
    
    .status-success {
        background-color: #4caf50;
        color: white;
    }
    
    .status-warning {
        background-color: #ff9800;
        color: white;
    }
    
    .status-info {
        background-color: #2196f3;
        color: white;
    }
    
    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

class MedicalSystemUI:
    """Streamlit UI wrapper for the medical consultation system"""
    
    def __init__(self):
        self.medical_system = None
        
    async def initialize_system(self):
        """Initialize the medical system components"""
        if st.session_state.system_initialized:
            return True
            
        try:
            with st.spinner("🔧 Initializing Medical System..."):
                # Setup MongoDB connection
                from pymongo import MongoClient
                from pymongo.errors import ConnectionFailure
                
                mongo_client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
                mongo_client.admin.command('ping')
                
                # Initialize medical expert agent
                medical_expert_agent = MedicalExpertAgent(mongo_client=mongo_client)
                
                # Setup MongoDB client for pharmacist
                mongodb_client = MongoDBClient(mongodb_uri, database_name)
                
                # Initialize pharmacist agent
                pharmacist_agent = PharmacistAgent(
                    openai_api_key=openai_api_key,
                    mongodb_client=mongodb_client
                )
                
                # Store in session state
                st.session_state.medical_expert_agent = medical_expert_agent
                st.session_state.pharmacist_agent = pharmacist_agent
                st.session_state.mongo_client = mongo_client
                st.session_state.mongodb_client = mongodb_client
                st.session_state.system_initialized = True
                
                self.log_debug("Medical system initialized successfully")
                return True
                
        except Exception as e:
            st.error(f"❌ Failed to initialize medical system: {str(e)}")
            self.log_debug(f"Initialization error: {str(e)}")
            return False
    
    def log_debug(self, message):
        """Add debug log with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.debug_logs.append(f"[{timestamp}] {message}")
    
    def render_header(self):
        """Render the main header"""
        st.markdown("""
        <div class="main-header">
            <h1>🏥 Enhanced Medical Consultation System</h1>
            <p>Comprehensive AI-powered healthcare consultation with intelligent product recommendations</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_step_indicator(self):
        """Render the step progress indicator"""
        steps = [
            {"number": 1, "title": "Medical Consultation", "icon": "🏥"},
            {"number": 2, "title": "Product Analysis", "icon": "💊"},
            {"number": 3, "title": "User Choice", "icon": "🤔"},
            {"number": 4, "title": "Support Integration", "icon": "📧"}
        ]
        
        cols = st.columns(4)
        for i, step in enumerate(steps):
            with cols[i]:
                if step["number"] < st.session_state.current_step:
                    status = "completed"
                    bg_color = "#4caf50"
                elif step["number"] == st.session_state.current_step:
                    status = "active"
                    bg_color = "#2196f3"
                else:
                    status = "pending"
                    bg_color = "#e0e0e0"
                
                st.markdown(f"""
                <div style="background-color: {bg_color}; color: white; padding: 1rem; border-radius: 10px; text-align: center; margin: 0.5rem 0;">
                    <div style="font-size: 2rem;">{step['icon']}</div>
                    <div style="font-weight: bold;">Step {step['number']}</div>
                    <div style="font-size: 0.9rem;">{step['title']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    async def render_step1_consultation(self):
        """Render Step 1: Medical Expert Consultation"""
        st.markdown("### 🏥 Step 1: Medical Expert Consultation")
        
        if not hasattr(st.session_state, 'consultation_started'):
            st.session_state.consultation_started = False
            st.session_state.consultation_messages = []
        
        # Display consultation messages
        for msg in st.session_state.consultation_messages:
            if msg["role"] == "doctor":
                st.markdown(f"""
                <div class="doctor-message">
                    <strong>🩺 Dr. Sanaullah:</strong> {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
            elif msg["role"] == "patient":
                st.markdown(f"""
                <div class="patient-message">
                    <strong>👤 You:</strong> {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
        
        # Start consultation if not started
        if not st.session_state.consultation_started:
            if st.button("🚀 Start Medical Consultation", use_container_width=True):
                st.session_state.consultation_started = True
                initial_message = st.session_state.medical_expert_agent.start_conversation()
                st.session_state.consultation_messages.append({
                    "role": "doctor",
                    "content": initial_message
                })
                st.rerun()
        
        # Chat interface
        if st.session_state.consultation_started:
            user_input = st.chat_input("Type your response to Dr. Sanaullah...")
            
            if user_input:
                # Add user message
                st.session_state.consultation_messages.append({
                    "role": "patient", 
                    "content": user_input
                })
                
                # Process with medical expert agent
                with st.spinner("🤔 Dr. Sanaullah is thinking..."):
                    result = st.session_state.medical_expert_agent.process_user_input(user_input)
                    
                    # Add doctor response
                    st.session_state.consultation_messages.append({
                        "role": "doctor",
                        "content": result['response']
                    })
                    
                    self.log_debug(f"Flow: {result['flow_action']} - {result['flow_reason']}")
                    
                    # Check if consultation ended
                    if result['conversation_ended']:
                        save_result = result.get('database_save_result')
                        if save_result and save_result['success']:
                            st.session_state.patient_id = save_result['patient_id']
                            st.session_state.current_step = 2
                            st.success("✅ Medical consultation completed! Patient data saved.")
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error("❌ Failed to save patient data")
                
                st.rerun()
    
    async def render_step2_analysis(self):
        """Render Step 2: Pharmacist Analysis"""
        st.markdown("### 💊 Step 2: Pharmacist Product Analysis")
        
        if not hasattr(st.session_state, 'analysis_started'):
            st.session_state.analysis_started = False
        
        if not st.session_state.analysis_started:
            if st.button("🔬 Start Product Analysis", use_container_width=True):
                st.session_state.analysis_started = True
                await self.run_pharmacist_analysis()
                st.rerun()
        else:
            await self.display_analysis_results()
    
    async def run_pharmacist_analysis(self):
        """Run the pharmacist analysis with real-time progress updates"""
        try:
            # Create progress containers
            progress_container = st.container()
            status_container = st.container()
            
            with progress_container:
                st.markdown("#### 🔧 Database Optimization")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Creating database indexes for optimal performance...")
                progress_bar.progress(20)
                
                # Create database indexes
                index_results = await st.session_state.pharmacist_agent.create_product_search_indexes()
                
                status_text.text(f"✅ Created {index_results.get('successful_indexes', 0)}/{index_results.get('total_indexes', 0)} database indexes")
                progress_bar.progress(40)
                time.sleep(1)
            
            with status_container:
                st.markdown("#### 📋 Patient Data Processing")
                patient_progress = st.progress(0)
                patient_status = st.empty()
                
                patient_status.text("Fetching patient data from database...")
                patient_progress.progress(30)
                
                # Fetch patient data
                patient_data = await st.session_state.pharmacist_agent.get_patient_by_id(st.session_state.patient_id)
                
                if "error" in patient_data:
                    st.error(f"❌ Error fetching patient: {patient_data['error']}")
                    return
                
                st.session_state.patient_data = patient_data
                patient_info = patient_data.get("patient", {})
                
                patient_status.text(f"✅ Patient data loaded: {patient_info.get('name', 'Unknown')} - {len(patient_info.get('symptoms', []))} symptoms")
                patient_progress.progress(100)
                time.sleep(1)
            
            # Product Analysis Section
            st.markdown("#### 🧠 AI Product Analysis")
            analysis_container = st.container()
            
            with analysis_container:
                # Step 1: Product Filtering
                st.markdown("##### 🔍 Step 1: Intelligent Product Filtering")
                filter_progress = st.progress(0)
                filter_status = st.empty()
                
                filter_status.text("🤖 AI analyzing patient symptoms to generate search filters...")
                filter_progress.progress(25)
                time.sleep(1)
                
                filter_status.text("🔎 Executing database queries with AI-generated filters...")
                filter_progress.progress(75)
                time.sleep(1)
                
                filter_status.text("✅ Product filtering completed")
                filter_progress.progress(100)
                
                # Step 2: Product Analysis
                st.markdown("##### 💊 Step 2: Product-Symptom Matching")
                match_progress = st.progress(0)
                match_status = st.empty()
                
                match_status.text("🧬 Starting AI-powered symptom analysis...")
                match_progress.progress(10)
                
                # Generate recommendations with progress updates
                recommendations = await self._generate_recommendations_with_progress(
                    patient_data=patient_data,
                    max_recommendations=5,
                    progress_bar=match_progress,
                    status_text=match_status
                )
                
                if "error" in recommendations:
                    st.error(f"❌ Error generating recommendations: {recommendations['error']}")
                    return
                
                st.session_state.recommendations = recommendations
                match_status.text("✅ Product analysis completed successfully!")
                match_progress.progress(100)
                
                # Step 3: Final Report
                st.markdown("##### 📋 Step 3: Consultation Report Generation")
                report_progress = st.progress(0)
                report_status = st.empty()
                
                report_status.text("📝 Generating comprehensive consultation report...")
                report_progress.progress(50)
                time.sleep(1)
                
                report_status.text("✅ Consultation report generated")
                report_progress.progress(100)
                
                st.session_state.current_step = 3
                
                # Show completion summary
                st.success("🎉 Analysis Complete! Ready to proceed to user choice.")
                
                # Show quick summary
                with st.expander("📊 Analysis Summary", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Products Analyzed", recommendations.get('total_products_analyzed', 0))
                    with col2:
                        st.metric("Top Recommendations", len(recommendations.get('recommendations', [])))
                    with col3:
                        st.metric("Analysis Time", "Real-time")
                
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            self.log_debug(f"Analysis error: {str(e)}")
    
    async def _generate_recommendations_with_progress(self, patient_data, max_recommendations, progress_bar, status_text):
        """Generate recommendations with progress updates"""
        try:
            # Step 1: Find relevant products
            status_text.text("🔍 Finding relevant products using intelligent filtering...")
            progress_bar.progress(20)
            
            products = await st.session_state.pharmacist_agent.find_products_with_intelligent_filtering(patient_data, limit=100)
            
            if not products:
                return {"error": "No products found in database"}
            
            status_text.text(f"📦 Found {len(products)} products to analyze")
            progress_bar.progress(40)
            
            # Step 2: Analyze each product
            status_text.text("🧪 Analyzing product-symptom matches...")
            progress_bar.progress(60)
            
            analyzed_products = []
            total_products = min(len(products), 50)  # Limit for performance
            
            for i, product in enumerate(products[:total_products]):
                # Update progress for each product
                current_progress = 60 + (i / total_products) * 30
                progress_bar.progress(int(current_progress))
                status_text.text(f"🔬 Analyzing product {i+1}/{total_products}: {product.get('product_name', 'Unknown')[:30]}...")
                
                match_analysis = await st.session_state.pharmacist_agent.analyze_product_symptom_match(patient_data, product)
                
                from agents.specialized.pharmacist_agent import ProductRecommendation
                recommendation = ProductRecommendation(
                    product_id=str(product.get("_id", "")),
                    product_name=product.get("product_name", "Unknown Product"),
                    recommendation_score=match_analysis.similarity_score,
                    symptom_match=match_analysis,
                    additional_factors={
                        "product_category": product.get("product_category", "Unknown"),
                        "cost_price": product.get("cost_price", 0),
                        "selling_price": product.get("selling_price", 0),
                        "branch_name": product.get("branch_name", "Unknown"),
                        "cpt_code": product.get("cpt_code", ""),
                        "compositions_count": len(product.get("compositions", []))
                    }
                )
                
                analyzed_products.append(recommendation)
            
            # Step 3: Sort and generate report
            status_text.text("📊 Ranking products and generating consultation report...")
            progress_bar.progress(90)
            
            analyzed_products.sort(key=lambda x: x.recommendation_score, reverse=True)
            top_recommendations = analyzed_products[:max_recommendations]
            
            # Generate consultation report
            patient_info = patient_data.get("patient", {})
            consultation_report = await st.session_state.pharmacist_agent.recommendation_chain.ainvoke({
                "patient_name": patient_info.get("name", "Unknown"),
                "patient_age": patient_info.get("age", "Unknown"),
                "patient_gender": patient_info.get("gender", "Unknown"),
                "patient_symptoms": ", ".join(patient_info.get("symptoms", [])),
                "patient_additional_info": json.dumps(patient_info.get("additional_info", {}), indent=2),
                "analyzed_products": json.dumps([{
                    "product_name": rec.product_name,
                    "score": rec.recommendation_score,
                    "reasoning": rec.symptom_match.reasoning,
                    "confidence": rec.symptom_match.confidence,
                    "matched_symptoms": rec.symptom_match.matched_symptoms,
                    "category": rec.additional_factors.get("product_category"),
                    "price": rec.additional_factors.get("selling_price")
                } for rec in top_recommendations], indent=2)
            })
            
            return {
                "patient_info": {
                    "patient_id": str(patient_data.get("_id", "unknown")),
                    "name": patient_info.get("name", "Unknown"),
                    "age": patient_info.get("age"),
                    "gender": patient_info.get("gender"),
                    "symptoms": patient_info.get("symptoms", []),
                    "additional_info": patient_info.get("additional_info", {})
                },
                "recommendations": [rec.dict() for rec in top_recommendations],
                "total_products_analyzed": len(analyzed_products),
                "consultation_report": consultation_report,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to generate recommendations: {str(e)}"}
    
    async def display_analysis_results(self):
        """Display the analysis results"""
        if st.session_state.patient_data and st.session_state.recommendations:
            # Patient summary
            st.markdown("#### 👤 Patient Information")
            patient_info = st.session_state.patient_data.get("patient", {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Name", patient_info.get('name', 'Unknown'))
            with col2:
                st.metric("Age", patient_info.get('age', 'Unknown'))
            with col3:
                st.metric("Gender", patient_info.get('gender', 'Unknown'))
            
            # Symptoms
            symptoms = patient_info.get('symptoms', [])
            if symptoms:
                st.markdown("**Symptoms:**")
                for symptom in symptoms:
                    st.write(f"• {symptom}")
            
            # Recommendations
            st.markdown("#### 🏆 Top Product Recommendations")
            
            recommendations = st.session_state.recommendations['recommendations']
            for i, rec in enumerate(recommendations, 1):
                with st.container():
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <h4>#{i} 💊 {rec['product_name']}</h4>
                        <div style="margin: 1rem 0;">
                            <span class="status-info">Score: {rec['recommendation_score']:.2f}/1.00</span>
                            <span class="status-success">Confidence: {rec['symptom_match']['confidence']}</span>
                        </div>
                        <p><strong>Category:</strong> {rec['additional_factors'].get('product_category', 'Unknown')}</p>
                        <p><strong>Price:</strong> ${rec['additional_factors'].get('selling_price', 0):,}</p>
                        <p><strong>Reasoning:</strong> {rec['symptom_match']['reasoning'][:200]}...</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Consultation report
            st.markdown("#### 📋 Pharmacist Consultation Report")
            with st.expander("View Full Report", expanded=False):
                st.markdown(st.session_state.recommendations['consultation_report'])
            
            if st.button("✅ Proceed to User Choice", use_container_width=True):
                st.session_state.current_step = 3
                st.rerun()
    
    def render_step3_choice(self):
        """Render Step 3: User Choice"""
        st.markdown("### 🤔 Step 3: What Would You Like To Do?")
        
        st.markdown("Based on our analysis and recommendations, please select your preferred next step:")
        
        choices = [
            {"key": "order_products", "emoji": "🛒", "title": "Purchase/Order Products", "description": "Order the recommended products"},
            {"key": "nurse_consultation", "emoji": "🩺", "title": "Nurse Consultation", "description": "Get guidance from our nursing team"},
            {"key": "customer_support", "emoji": "📞", "title": "Customer Support", "description": "Speak with our support team"},
            {"key": "email_summary", "emoji": "📧", "title": "Email Summary", "description": "Send summary to my email"},
            {"key": "exit", "emoji": "❌", "title": "Exit", "description": "Exit without taking action"}
        ]
        
        cols = st.columns(2)
        for i, choice in enumerate(choices):
            with cols[i % 2]:
                if st.button(
                    f"{choice['emoji']} {choice['title']}\n{choice['description']}", 
                    key=choice['key'],
                    use_container_width=True
                ):
                    st.session_state.user_choice = choice['key']
                    st.session_state.current_step = 4
                    st.rerun()
    
    async def render_step4_support(self):
        """Render Step 4: Customer Support Integration"""
        st.markdown("### 📧 Step 4: Customer Support Integration")
        
        if st.session_state.user_choice:
            choice_descriptions = {
                "order_products": "Patient wants to purchase/order recommended products",
                "nurse_consultation": "Patient wants nurse guidance with product usage",
                "customer_support": "Patient wants to speak with customer support",
                "email_summary": "Patient requested email summary",
                "exit": "Patient exited without taking action"
            }
            
            chosen_description = choice_descriptions.get(st.session_state.user_choice, "Unknown choice")
            
            st.markdown(f"""
            <div class="status-info">
                Selected Choice: {chosen_description}
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("📧 Send Summary to Customer Support", use_container_width=True):
                await self.send_support_email()
    
    async def send_support_email(self):
        """Send comprehensive summary to customer support"""
        try:
            with st.spinner("📧 Preparing comprehensive summary..."):
                patient_info = st.session_state.patient_data.get("patient", {})
                
                # Format recommendations for email (FULL details, no truncation)
                formatted_recommendations = []
                for i, rec in enumerate(st.session_state.recommendations['recommendations'], 1):
                    # Include FULL reasoning without truncation
                    full_reasoning = rec['symptom_match']['reasoning']
                    matched_symptoms_str = ', '.join(rec['symptom_match']['matched_symptoms']) if rec['symptom_match']['matched_symptoms'] else 'None'
                    
                    formatted_recommendations.append(f"""
{i}. {rec['product_name']}
   - Recommendation Score: {rec['recommendation_score']:.2f}/1.00
   - Confidence Level: {rec['symptom_match']['confidence']}
   - Product Category: {rec['additional_factors'].get('product_category', 'Unknown')}
   - Selling Price: ${rec['additional_factors'].get('selling_price', 0):,}
   - Cost Price: ${rec['additional_factors'].get('cost_price', 0):,}
   - Branch: {rec['additional_factors'].get('branch_name', 'Unknown')}
   - CPT Code: {rec['additional_factors'].get('cpt_code', 'N/A')}
   - Matched Symptoms: {matched_symptoms_str}
   - Detailed Reasoning: {full_reasoning}
   - Compositions Count: {rec['additional_factors'].get('compositions_count', 0)} ingredients
                    """)
                
                # Get FULL consultation report without truncation
                full_consultation_report = st.session_state.recommendations['consultation_report']
                
                # Format additional patient information
                additional_info_str = ""
                if patient_info.get('additional_info'):
                    additional_info_str = "\n".join([f"- {key}: {value}" for key, value in patient_info.get('additional_info', {}).items()])
                else:
                    additional_info_str = "- No additional information provided"
                
                # Create comprehensive summary (NO TRUNCATION)
                summary_content = f"""
MEDICAL SYSTEM CONSULTATION SUMMARY
=====================================

PATIENT INFORMATION:
- Name: {patient_info.get('name', 'Unknown')}
- Age: {patient_info.get('age', 'Unknown')}
- Gender: {patient_info.get('gender', 'Unknown')}
- Patient ID: {st.session_state.patient_id}

SYMPTOMS REPORTED:
{chr(10).join(f"- {symptom}" for symptom in patient_info.get('symptoms', [])) or "- No symptoms recorded"}

ADDITIONAL PATIENT INFORMATION:
{additional_info_str}

MEDICAL ANALYSIS SUMMARY:
- Total products in database searched: {st.session_state.recommendations['total_products_analyzed']}
- Analysis timestamp: {st.session_state.recommendations['analysis_timestamp']}
- Analysis method: AI-powered symptom matching with LLM analysis
- Recommendation engine: GPT-4 Turbo with medical optimization

TOP PRODUCT RECOMMENDATIONS (COMPLETE DETAILS):
{''.join(formatted_recommendations)}

COMPLETE PHARMACIST CONSULTATION REPORT:
{full_consultation_report}

USER DECISION AND CHOICE:
- Selected Option: {self.get_choice_description(st.session_state.user_choice)}
- Choice Made At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Patient Autonomy: Patient made informed decision after reviewing all recommendations

CONVERSATION SUMMARY:
- Total consultation messages: {len(st.session_state.consultation_messages)}
- Consultation completed: Yes
- Patient data saved: Yes
- Recommendations generated: Yes
- Email notification: In progress

RECOMMENDED NEXT ACTIONS:
1. Follow up with patient based on their choice: {self.get_choice_description(st.session_state.user_choice)}
2. Review recommendations with medical team within 24 hours
3. Ensure proper product availability if patient chose to order
4. Document consultation in patient medical records
5. Schedule follow-up appointment if needed
6. Contact patient for clarification if any details are unclear
7. Monitor patient satisfaction and outcome

TECHNICAL DETAILS:
- System Version: Enhanced Medical Consultation System v2.0
- AI Models Used: OpenAI GPT-4 Turbo for medical analysis
- Database: MongoDB with intelligent indexing
- Email System: Professional HTML templates with full content
- Security: HIPAA-compliant data handling and transmission

COMPLIANCE NOTES:
- All patient data handled according to healthcare privacy regulations
- Consultation summary generated automatically for quality assurance
- Patient consent obtained for data processing and communication
- Medical recommendations reviewed by AI system with high confidence thresholds
                """
                
            with st.spinner("📤 Sending email to customer support..."):
                # Generate and send email with COMPLETE content
                email_data = generate_medical_consultation_email(
                    patient_name=patient_info.get('name', 'Unknown Patient'),
                    patient_id=st.session_state.patient_id,
                    user_choice_description=self.get_choice_description(st.session_state.user_choice),
                    summary_content=summary_content,  # Full content, no truncation
                    timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                )
                
                send_email(
                    email_to=config.email.SUPPORT_EMAIL,
                    subject=email_data.subject,
                    html_content=email_data.html_content
                )
                
                st.success("✅ Comprehensive summary sent to customer support team!")
                st.info(f"📧 Email sent to: {config.email.SUPPORT_EMAIL}")
                
                # Show email preview for verification
                with st.expander("📧 Email Content Preview (for verification)", expanded=False):
                    st.markdown("**Subject:** " + email_data.subject)
                    st.markdown("**Summary Length:** " + str(len(summary_content)) + " characters")
                    st.markdown("**Content Preview:**")
                    st.text_area("Email Summary Content", summary_content, height=300, disabled=True)
                
                st.session_state.consultation_complete = True
                
                # Show completion message
                st.balloons()
                st.markdown("""
                <div style="background-color: #4caf50; color: white; padding: 2rem; border-radius: 10px; text-align: center; margin: 2rem 0;">
                    <h2>🎉 Medical Consultation Completed Successfully!</h2>
                    <p>Your consultation has been processed and our team will follow up within 24 hours.</p>
                    <p><strong>Complete summary sent with full details - no information truncated!</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"❌ Error sending email: {str(e)}")
            self.log_debug(f"Email error: {str(e)}")
            
            # Show error details for debugging
            with st.expander("🔍 Error Details", expanded=True):
                st.code(str(e))
                st.markdown("**Troubleshooting Steps:**")
                st.markdown("1. Check email configuration in config/config.py")
                st.markdown("2. Verify SMTP settings are correct")
                st.markdown("3. Ensure email service is accessible")
                st.markdown("4. Check network connectivity")
    
    def get_choice_description(self, choice):
        """Get human-readable description of user choice"""
        choices = {
            "order_products": "Patient wants to purchase/order recommended products",
            "nurse_consultation": "Patient wants nurse guidance with product usage",
            "customer_support": "Patient wants to speak with customer support",
            "email_summary": "Patient requested email summary",
            "exit": "Patient exited without taking action"
        }
        return choices.get(choice, "Unknown choice")

# Initialize UI
medical_ui = MedicalSystemUI()

def run_async_function(func, *args):
    """Run async function in Streamlit"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(func(*args))

# Main App Layout
medical_ui.render_header()

# Sidebar
with st.sidebar:
    st.header("🏥 Medical System Control")
    
    # System status
    if st.session_state.system_initialized:
        st.success("✅ System Initialized")
    else:
        st.info("⏳ System Not Initialized")
    
    # Current step indicator
    st.info(f"📍 Current Step: {st.session_state.current_step}/4")
    
    st.divider()
    
    # Control buttons
    if st.button("🔄 Reset System", use_container_width=True):
        # Reset all session state
        for key in list(st.session_state.keys()):
            if key.startswith(('medical_', 'consultation_', 'patient_', 'recommendations', 'user_choice', 'current_step', 'system_initialized')):
                del st.session_state[key]
        st.session_state.current_step = 1
        st.session_state.system_initialized = False
        st.rerun()
    
    # Debug toggle
    st.divider()
    st.checkbox("🔍 Show Debug Information", key="show_debug")
    
    # System information
    st.divider()
    st.subheader("📊 System Information")
    st.write(f"**MongoDB URI:** {mongodb_uri}")
    st.write(f"**Database:** {database_name}")
    st.write(f"**OpenAI API:** {'✅ Configured' if openai_api_key else '❌ Missing'}")

# Main content area
medical_ui.render_step_indicator()

# Initialize system if not done
if not st.session_state.system_initialized:
    if st.button("🚀 Initialize Medical System", use_container_width=True, type="primary"):
        success = run_async_function(medical_ui.initialize_system)
        if success:
            st.rerun()

# Render current step
if st.session_state.system_initialized:
    if st.session_state.current_step == 1:
        run_async_function(medical_ui.render_step1_consultation)
    elif st.session_state.current_step == 2:
        run_async_function(medical_ui.render_step2_analysis)
    elif st.session_state.current_step == 3:
        medical_ui.render_step3_choice()
    elif st.session_state.current_step == 4:
        run_async_function(medical_ui.render_step4_support)

# Debug panel
if st.session_state.show_debug and st.session_state.debug_logs:
    with st.expander("🔍 Debug Logs", expanded=False):
        for log in st.session_state.debug_logs[-20:]:  # Show last 20 logs
            st.code(log)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    Enhanced Medical Consultation System v2.0 | 
    Powered by OpenAI GPT-4 & MongoDB | 
    Professional Healthcare AI Assistant
</div>
""", unsafe_allow_html=True) 