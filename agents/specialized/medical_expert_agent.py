from datetime import datetime
from typing import Dict, List, Any, Optional
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from enum import Enum
import json
import re
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError


class ConversationAction(str, Enum):
    """Possible conversation actions"""
    CONTINUE_GATHERING = "continue_gathering"
    OFFER_ANALYSIS = "offer_analysis"
    END_CONVERSATION = "end_conversation"


class ConversationFlow(BaseModel):
    """LLM-driven conversation flow decision"""
    action: ConversationAction = Field(..., description="Next action to take in the conversation")
    reason: str = Field(..., description="Reasoning behind the decision")
    suggested_response: Optional[str] = Field(None, description="Suggested response to the patient")
    missing_info: List[str] = Field(default_factory=list, description="List of missing information that should be gathered")


class PatientInformation(BaseModel):
    """Structured patient information model for LLM extraction"""
    name: Optional[str] = Field(None, description="Patient's full name or first name")
    age: Optional[int] = Field(None, description="Patient's age in years")
    gender: Optional[str] = Field(None, description="Patient's gender (Male/Female/Other)")
    symptoms: List[str] = Field(default_factory=list, description="List of symptoms mentioned by the patient")
    medical_history: List[str] = Field(default_factory=list, description="Any medical history or conditions mentioned")
    medications: List[str] = Field(default_factory=list, description="Any medications mentioned")
    additional_info: Dict[str, str] = Field(default_factory=dict, description="Any other relevant medical information")


class MedicalExpertAgent:
    def __init__(self, openai_api_key: str = None, mongo_client: MongoClient = None):
        """
        Initialize the Medical Expert Agent with LangChain components
        
        Args:
            openai_api_key: OpenAI API key for the language model
            mongo_client: MongoDB client for saving patient data
        """
        self.llm = ChatOpenAI(
            temperature=0.75,  # Slightly higher temperature for more natural conversation
            model_name="gpt-3.5-turbo",
            openai_api_key=openai_api_key
        )
        
        # Separate LLM for information extraction with lower temperature for accuracy
        self.extraction_llm = ChatOpenAI(
            temperature=0.1,  # Lower temperature for more consistent extraction
            model_name="gpt-3.5-turbo",
            openai_api_key=openai_api_key
        )
        
        # Patient information template
        self.patient_template = {
            "name": "",
            "age": None,
            "gender": "",
            "symptoms": [],
            "medical_history": [],
            "medications": [],
            "additional_info": {},
            "chat_history": [],
            "timestamp": datetime.now(),
            "completion_notified": False,
            "qa_pairs_count": 0,
            "extraction_performed": False
        }
        
        # Current patient data
        self.current_patient = self.patient_template.copy()
        
        # Setup conversation, extraction, and flow management chains
        self._setup_conversation_chain()
        self._setup_extraction_chain()
        self._setup_flow_management_chain()
        
        # Real-time LLM extraction configuration
        self.real_time_extraction_enabled = True
        
        # MongoDB client
        self.mongo_client = mongo_client

        # Initialize conversation state
        self.conversation_ended = False
    
    def _setup_conversation_chain(self):
        """Setup the conversational QA chain with memory"""
        
        # System prompt for the medical expert
        self.medical_expert_prompt = """You are Dr. Sarah Mitchell, a caring medical specialist. You have a reputation for being exceptionally attentive to your patients and making them feel heard. Your communication style is natural, conversational, and puts patients at ease while maintaining professionalism.

        REAL-TIME PATIENT INFORMATION (automatically extracted from conversation):
        {patient_info}
        
        Conversation Style Guidelines:
        1. Use natural, flowing conversation like a real doctor would - avoid robotic or scripted responses
        2. Show genuine concern and empathy through your word choice and tone
        3. ALWAYS acknowledge what the patient just shared before asking for more information
        4. Use gentle, caring transitions: "I understand that must be uncomfortable. Before we explore this further..."
        5. Make information gathering feel natural: "So I can help you better, could you tell me..."
        6. Validate feelings and show you're listening: "That sounds really frustrating..."
        7. Maintain a warm, caring tone while staying professional
        8. Never rush to treatment - take time to understand the person first
        9. **IMPORTANT**: If you notice the patient provided information (like their name) but you're still missing it in the patient info, acknowledge what they said and naturally ask them to repeat it: "I want to make sure I heard you correctly - could you tell me your name again?"
        
        SMART INFORMATION GATHERING:
        - Check the current patient information status before each response
        - If basic information (name, age, gender) is missing, prioritize gathering it naturally
        - If you have the patient's name, USE IT in your responses to show you're listening
        - Acknowledge symptoms and concerns, but guide conversation to gather missing essentials
        - Be responsive to what the patient just said - don't ignore their input
        
        CONVERSATION FLOW PRIORITIES:
        1. **FIRST PRIORITY - Basic Information (name, age, gender):**
           - If name is missing: Acknowledge their concern, then ask: "Before we continue, I'd love to know your name so I can address you properly."
           - If age is missing: "Could you tell me your age? This helps me provide better care."
           - If gender is missing: "What is your gender? This helps me understand your health needs."
        
        2. **SECOND PRIORITY - Symptom Exploration:**
           - Acknowledge symptoms with empathy
           - Ask about duration, triggers, severity, impact on daily life
        
        3. **THIRD PRIORITY - Medical History and Treatment:**
           - Previous conditions, medications, family history
           - Treatment options with our medical team
        
        MEDICAL FACILITY CONTEXT:
        - We have our own medical team including doctors, nurses, and specialists
        - Always refer to "our medical team" or "our specialists" rather than external providers
        - Mention "our customer support team" for scheduling and coordination
        
        Remember: 
        - Be responsive to what the patient just said - acknowledge their input
        - Use their name if you have it to show you're listening
        - Make information gathering feel like a natural conversation, not an interrogation
        - Focus on building rapport while gathering essential information
        """
        
        # Create the conversation prompt template
        self.conversation_prompt = ChatPromptTemplate.from_messages([
            ("system", self.medical_expert_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        # Create the conversation chain
        self.conversation_chain = (
            RunnablePassthrough.assign(
                patient_info=lambda x: self._format_patient_info()
            )
            | self.conversation_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _setup_extraction_chain(self):
        """Setup the LLM-based information extraction chain"""
        
        # Create parser for structured output
        self.parser = PydanticOutputParser(pydantic_object=PatientInformation)
        
        # System prompt for information extraction
        extraction_prompt = """You are an expert medical information extraction system. Your task is to analyze a conversation between a doctor and patient and extract all relevant patient information.

        Analyze the entire conversation history and extract:
        1. Patient's name (first name, last name, or any name mentioned)
        2. Patient's age (exact number if mentioned)
        3. Patient's gender (Male, Female, or Other - normalize variations)
        4. All symptoms mentioned (be comprehensive, include all health complaints)
        5. Medical history (past conditions, surgeries, family history)
        6. Current medications (prescription drugs, over-the-counter, supplements)
        7. Additional relevant information (allergies, lifestyle factors, etc.)

        Instructions:
        - Extract information from BOTH patient and doctor messages
        - Be thorough but accurate - only extract information that is clearly stated
        - Normalize similar terms (e.g., "headache", "head pain", "head hurts" → "headache")
        - For symptoms, include descriptive details when provided
        - If age is mentioned as a range or approximation, use the most specific number given
        - For gender, standardize to: Male, Female, or Other
        - If no information is found for a field, leave it empty/null

        {format_instructions}

        Conversation History:
        {conversation_history}
        """
        
        self.extraction_prompt = ChatPromptTemplate.from_template(extraction_prompt)
        
        # Create the extraction chain
        self.extraction_chain = (
            self.extraction_prompt
            | self.extraction_llm
            | self.parser
        )
    
    def _setup_flow_management_chain(self):
        """Setup the LLM-based conversation flow management chain"""
        
        # Create parser for conversation flow decisions
        self.flow_parser = PydanticOutputParser(pydantic_object=ConversationFlow)
        
        # System prompt for conversation flow management
        flow_prompt = """You are an expert medical conversation flow manager. Your task is to analyze the current state of a doctor-patient conversation and decide what action should be taken next.

        Analyze the conversation and current patient information to determine:
        1. Whether to continue gathering information
        2. Whether to offer analysis/treatment suggestions
        3. Whether to end the conversation naturally

        Decision Guidelines:
        - CONTINUE_GATHERING: If ANY basic information is missing (name, age, gender) OR if symptoms need more exploration
        - OFFER_ANALYSIS: ONLY if ALL basic information is complete (name, age, gender) AND sufficient symptom details are gathered
        - END_CONVERSATION: If patient clearly indicates they want to end (goodbye, thanks for help, I'm done) AND has provided substantial information
        
        CRITICAL: Never offer analysis or treatment if basic information (name, age, gender) is incomplete!

        Current Patient Information:
        {patient_info}

        Recent Conversation Context (last 4 messages):
        {recent_conversation}

        Patient's Latest Message: "{latest_message}"

        Consider:
        - Patient's tone and engagement level
        - Whether they're asking for more help or seem satisfied
        - If they're providing new information or just acknowledging
        - Whether they've indicated they want to end the conversation

        {format_instructions}
        """
        
        self.flow_prompt = ChatPromptTemplate.from_template(flow_prompt)
        
        # Create the flow management chain
        self.flow_chain = (
            self.flow_prompt
            | self.extraction_llm  # Use extraction LLM for consistent decisions
            | self.flow_parser
        )
    
    def _format_conversation_for_extraction(self) -> str:
        """Format the conversation history for the extraction chain"""
        formatted_messages = []
        
        for i, message in enumerate(self.current_patient["chat_history"]):
            if isinstance(message, HumanMessage):
                formatted_messages.append(f"Patient: {message.content}")
            elif isinstance(message, AIMessage):
                formatted_messages.append(f"Doctor: {message.content}")
        
        return "\n\n".join(formatted_messages)
    
    def _get_recent_conversation(self, num_messages: int = 4) -> str:
        """Get recent conversation context for flow management"""
        recent_messages = self.current_patient["chat_history"][-num_messages:]
        formatted_messages = []
        
        for message in recent_messages:
            if isinstance(message, HumanMessage):
                formatted_messages.append(f"Patient: {message.content}")
            elif isinstance(message, AIMessage):
                formatted_messages.append(f"Doctor: {message.content}")
        
        return "\n\n".join(formatted_messages)
    
    def _extract_information_with_llm(self):
        """Use LLM to extract patient information from conversation history in real-time"""
        if len(self.current_patient["chat_history"]) < 2:
            return
        
        try:
            conversation_text = self._format_conversation_for_extraction()
            
            extracted_info = self.extraction_chain.invoke({
                "conversation_history": conversation_text,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # Track what information was newly extracted
            newly_extracted = []
            
            # Update patient information with extracted data (always update if LLM found better info)
            if extracted_info.name and not self.current_patient["name"]:
                self.current_patient["name"] = extracted_info.name
                newly_extracted.append(f"Name: {extracted_info.name}")
            elif extracted_info.name and extracted_info.name != self.current_patient["name"]:
                # Update if we found a better/more complete name
                self.current_patient["name"] = extracted_info.name
                newly_extracted.append(f"Name (updated): {extracted_info.name}")
            
            if extracted_info.age and not self.current_patient["age"]:
                self.current_patient["age"] = extracted_info.age
                newly_extracted.append(f"Age: {extracted_info.age}")
            
            if extracted_info.gender and not self.current_patient["gender"]:
                self.current_patient["gender"] = extracted_info.gender
                newly_extracted.append(f"Gender: {extracted_info.gender}")
            
            # Merge symptoms (avoid duplicates)
            new_symptoms = []
            for symptom in extracted_info.symptoms:
                if symptom.lower() not in [s.lower() for s in self.current_patient["symptoms"]]:
                    self.current_patient["symptoms"].append(symptom)
                    new_symptoms.append(symptom)
            if new_symptoms:
                newly_extracted.append(f"New symptoms: {', '.join(new_symptoms)}")
            
            # Update medical history
            new_history = []
            for history_item in extracted_info.medical_history:
                if history_item not in self.current_patient["medical_history"]:
                    self.current_patient["medical_history"].append(history_item)
                    new_history.append(history_item)
            if new_history:
                newly_extracted.append(f"Medical history: {', '.join(new_history)}")
            
            # Update medications
            new_medications = []
            for medication in extracted_info.medications:
                if medication not in self.current_patient["medications"]:
                    self.current_patient["medications"].append(medication)
                    new_medications.append(medication)
            if new_medications:
                newly_extracted.append(f"Medications: {', '.join(new_medications)}")
            
            # Update additional info
            for key, value in extracted_info.additional_info.items():
                if key not in self.current_patient["additional_info"]:
                    self.current_patient["additional_info"][key] = value
                    newly_extracted.append(f"{key}: {value}")
            
            # Log what was newly extracted (for debugging)
            if newly_extracted:
                print(f"✅ Extracted: {'; '.join(newly_extracted)}")
            
            self.current_patient["extraction_performed"] = True
            
        except Exception as e:
            print(f"Warning: LLM extraction failed: {e}")
            # Fall back to basic extraction if LLM fails
            self._basic_extraction_fallback()
    
    def _basic_extraction_fallback(self):
        """Fallback extraction method using simple patterns"""
        # Simple pattern-based extraction as backup
        for message in self.current_patient["chat_history"]:
            if isinstance(message, HumanMessage):
                content = message.content.lower()
                
                # Basic name extraction
                if not self.current_patient["name"]:
                    name_patterns = [r"my name is (\w+)", r"i'm (\w+)", r"i am (\w+)", r"call me (\w+)"]
                    for pattern in name_patterns:
                        match = re.search(pattern, content)
                        if match:
                            self.current_patient["name"] = match.group(1).title()
                            break
                
                # Basic age extraction
                if not self.current_patient["age"]:
                    age_patterns = [r"i'm (\d+)", r"i am (\d+)", r"(\d+) years old"]
                    for pattern in age_patterns:
                        match = re.search(pattern, content)
                        if match:
                            try:
                                self.current_patient["age"] = int(match.group(1))
                                break
                            except ValueError:
                                pass
    
    def _format_patient_info(self) -> str:
        """Format current patient information for the prompt"""
        info_parts = []
        
        if self.current_patient["name"]:
            info_parts.append(f"Name: {self.current_patient['name']}")
        
        if self.current_patient["age"]:
            info_parts.append(f"Age: {self.current_patient['age']}")
        
        if self.current_patient["gender"]:
            info_parts.append(f"Gender: {self.current_patient['gender']}")
        
        if self.current_patient["symptoms"]:
            symptoms_str = ", ".join(self.current_patient["symptoms"])
            info_parts.append(f"Symptoms: {symptoms_str}")
        
        if self.current_patient["medical_history"]:
            history_str = ", ".join(self.current_patient["medical_history"])
            info_parts.append(f"Medical History: {history_str}")
        
        if self.current_patient["medications"]:
            meds_str = ", ".join(self.current_patient["medications"])
            info_parts.append(f"Medications: {meds_str}")
        
        if not info_parts:
            return "No patient information gathered yet."
        
        return "\n".join(info_parts)
    

    
    def _determine_conversation_flow(self, user_input: str) -> ConversationFlow:
        """Use LLM to determine the next conversation action"""
        try:
            recent_conversation = self._get_recent_conversation()
            patient_info = self._format_patient_info()
            
            flow_decision = self.flow_chain.invoke({
                "patient_info": patient_info,
                "recent_conversation": recent_conversation,
                "latest_message": user_input,
                "format_instructions": self.flow_parser.get_format_instructions()
            })
            
            return flow_decision
            
        except Exception as e:
            print(f"Warning: Flow management failed: {e}")
            # Fallback to continue gathering if flow management fails
            return ConversationFlow(
                action=ConversationAction.CONTINUE_GATHERING,
                reason="Flow management failed, defaulting to continue gathering",
                suggested_response=None,
                missing_info=[]
            )
    
    def start_conversation(self) -> str:
        """Start the conversation with a warm, doctor-like greeting"""
        greeting = """Hello! I'm Dr. Sanaullah, and I'm so glad you're here today. *warm smile* 

I want you to feel completely comfortable sharing what's brought you in. I believe in taking the time to really get to know my patients as people first - your name, a bit about you, and then we'll explore what's concerning you.

So let's start with the basics - what's your name? I'd love to know what to call you. 😊"""
        
        # Add to chat history
        self.current_patient["chat_history"].append(AIMessage(content=greeting))
        
        return greeting
    
    def _get_missing_information(self) -> List[str]:
        """Get a list of missing required information fields"""
        missing = []
        if not self.current_patient["name"]:
            missing.append("name")
        if self.current_patient["age"] is None:
            missing.append("age")
        if not self.current_patient["gender"]:
            missing.append("gender")
        if not self.current_patient["symptoms"]:
            missing.append("symptoms")
        return missing
    
    def _get_missing_basic_info(self) -> List[str]:
        """Get missing basic information (name, age, gender) with priority"""
        missing = []
        if not self.current_patient["name"]:
            missing.append("name")
        if self.current_patient["age"] is None:
            missing.append("age")
        if not self.current_patient["gender"]:
            missing.append("gender")
        return missing

    def _get_next_question(self, missing_fields: List[str]) -> str:
        """Generate the next question to ask based on missing information"""
        if "name" in missing_fields:
            return "Could you please tell me your name?"
        elif "age" in missing_fields:
            return "And how old are you?"
        elif "gender" in missing_fields:
            return "What is your gender?"
        elif "symptoms" in missing_fields:
            return "What symptoms are you experiencing today?"
        return ""

    def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input and generate appropriate response with flow management"""
        # Add user message to chat history
        self.current_patient["chat_history"].append(HumanMessage(content=user_input))
        
        # Increment QA pairs count
        self.current_patient["qa_pairs_count"] += 1
        
        # Run LLM extraction on EVERY user input for real-time information gathering
        print("🔍 Running real-time LLM extraction...")
        self._extract_information_with_llm()
        
        # Determine conversation flow using LLM
        flow_decision = self._determine_conversation_flow(user_input)
        
        # Handle different conversation actions
        if flow_decision.action == ConversationAction.END_CONVERSATION:
            response = self._generate_farewell_response(flow_decision)
            self.conversation_ended = True
            
        elif flow_decision.action == ConversationAction.OFFER_ANALYSIS:
            response = self._generate_analysis_offer_response(flow_decision)
            self.conversation_ended = False
            
        else:  # CONTINUE_GATHERING
            response = self._generate_gathering_response(user_input, flow_decision)
            self.conversation_ended = False
        
        # Add AI response to chat history
        self.current_patient["chat_history"].append(AIMessage(content=response))
        
        # Save patient information to database when conversation ends
        database_save_result = None
        if self.conversation_ended:
            print("💾 Conversation ended - saving patient information to database...")
            database_save_result = self._save_patient_to_database()
        
        # Return comprehensive response information
        return {
            "response": response,
            "conversation_ended": self.conversation_ended,
            "flow_action": flow_decision.action.value,
            "flow_reason": flow_decision.reason,
            "patient_info": self.get_patient_info(),
            "extraction_performed": self.current_patient["extraction_performed"],
            "database_save_result": database_save_result
        }
    
    def _generate_gathering_response(self, user_input: str, flow_decision: ConversationFlow) -> str:
        """Generate response for continuing information gathering"""
        # Use the normal conversation flow with real-time extracted information
        response = self.conversation_chain.invoke({
            "question": user_input,
            "chat_history": self.current_patient["chat_history"][:-1],  # Exclude the current message
        })
        
        return response
    
    def _generate_basic_info_request(self, user_input: str, missing_field: str) -> str:
        """Generate a natural request for missing basic information"""
        # Acknowledge what the patient said first
        acknowledgments = {
            "pain": "I can hear that you're experiencing pain, and I want to help you with that.",
            "hurt": "I understand you're hurting, and that must be really difficult.",
            "sensitivity": "Tooth sensitivity can be quite uncomfortable, I know.",
            "problem": "I can see this is causing you problems.",
            "issue": "I understand this is concerning you."
        }
        
        # Find appropriate acknowledgment
        user_lower = user_input.lower()
        acknowledgment = "I hear what you're sharing with me."
        for key, ack in acknowledgments.items():
            if key in user_lower:
                acknowledgment = ack
                break
        
        # Generate request based on missing field
        if missing_field == "name":
            return f"""{acknowledgment} 

Before we dive deeper into what's going on, I'd love to know your name so I can address you properly. What should I call you?"""
        
        elif missing_field == "age":
            name = self.current_patient.get("name", "")
            name_part = f"{name}, " if name else ""
            return f"""{acknowledgment} 

{name_part}could you tell me your age? This helps me provide better care that's right for you."""
        
        elif missing_field == "gender":
            name = self.current_patient.get("name", "")
            name_part = f"{name}, " if name else ""
            return f"""{acknowledgment} 

{name_part}what is your gender? This is important for me to understand your health needs properly."""
        
        return acknowledgment
    
    def _generate_analysis_offer_response(self, flow_decision: ConversationFlow) -> str:
        """Generate response offering analysis or treatment suggestions"""
        # Double-check that basic info is complete before offering analysis
        missing_basic = self._get_missing_basic_info()
        if missing_basic:
            # Force back to gathering if basic info is still missing
            return self._generate_basic_info_request("", missing_basic[0])
        
        if flow_decision.suggested_response:
            base_response = flow_decision.suggested_response
        else:
            # Generate a response offering analysis
            patient_name = self.current_patient.get("name", "")
            name_part = f"{patient_name}, " if patient_name else ""
            
            base_response = f"""Thank you for sharing all that information with me, {name_part}😊. 
            
I have a good understanding of your situation now. Based on what you've shared, I can connect you with our medical team for proper evaluation and treatment. We have experienced doctors, nurses, and specialists who can provide comprehensive care for your condition.

Would you like me to:
1. Provide some initial guidance and recommendations
2. Connect you with our customer support team to schedule an appointment with our specialists
3. Discuss treatment options available through our medical facility

What would be most helpful for you right now?"""
        
        # # Add summary if not already shown
        # if not self.current_patient.get("completion_notified"):
        #     self.current_patient["completion_notified"] = True
        #     base_response += "\n\n📋 **Summary of what you've shared:**"
        #     base_response += "\n" + self.get_conversation_summary()
        
        return base_response
    
    def _generate_farewell_response(self, flow_decision: ConversationFlow) -> str:
        """Generate farewell response when ending conversation"""
        if flow_decision.suggested_response:
            return flow_decision.suggested_response
        
        patient_name = self.current_patient.get("name", "")
        name_part = f"{patient_name}" if patient_name else "there"
        
        farewell = f"""Thank you so much for sharing with me today, {name_part}. *warm smile* 

It's been a pleasure talking with you. I hope you feel heard and that our conversation has been helpful. Please don't hesitate to seek professional medical care if your symptoms persist or worsen.

Take care of yourself, and I wish you all the best! 😊

*End of consultation*"""
        
        # Add final summary if we have information
        # if any([self.current_patient["name"], self.current_patient["symptoms"], 
        #        self.current_patient["age"], self.current_patient["gender"]]):
        #     farewell += "\n\n📋 **Final Summary:**"
        #     farewell += "\n" + self.get_conversation_summary()
        
        return farewell
    
    def process_user_input_simple(self, user_input: str) -> str:
        """Backward compatibility method that returns just the response string"""
        result = self.process_user_input(user_input)
        return result["response"]
    
    def get_patient_info(self) -> Dict[str, Any]:
        """Get current patient information"""
        return self.current_patient.copy()
    
    def is_information_complete(self) -> bool:
        """Check if basic patient information is complete"""
        return (
            bool(self.current_patient["name"]) and
            self.current_patient["age"] is not None and
            bool(self.current_patient["gender"]) and
            len(self.current_patient["symptoms"]) > 0
        )
    
    def reset_patient(self):
        """Reset patient information for a new consultation"""
        self.current_patient = self.patient_template.copy()
        self.current_patient["timestamp"] = datetime.now()
    
    def trigger_manual_extraction(self) -> Dict[str, Any]:
        """Manually trigger LLM extraction for testing purposes"""
        if len(self.current_patient["chat_history"]) < 2:
            return {"error": "Not enough conversation history for extraction"}
        
        try:
            self._extract_information_with_llm()
            return {
                "success": True,
                "extracted_info": self.get_patient_info(),
                "message": "LLM extraction completed successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "LLM extraction failed"
            }
    
    def export_patient_data(self) -> str:
        """Export patient data as JSON string"""
        # Convert datetime to string for JSON serialization
        export_data = self.current_patient.copy()
        export_data["timestamp"] = export_data["timestamp"].isoformat()
        
        # Convert chat history to serializable format
        chat_history_serializable = []
        for message in export_data["chat_history"]:
            if isinstance(message, HumanMessage):
                chat_history_serializable.append({"type": "human", "content": message.content})
            elif isinstance(message, AIMessage):
                chat_history_serializable.append({"type": "ai", "content": message.content})
        
        export_data["chat_history"] = chat_history_serializable
        
        return json.dumps(export_data, indent=2)
    
    def get_conversation_summary(self) -> str:
        """Generate a summary of the conversation and gathered information"""
        if not any([self.current_patient["name"], self.current_patient["age"], 
                   self.current_patient["gender"], self.current_patient["symptoms"],
                   self.current_patient["medical_history"], self.current_patient["medications"]]):
            return "No patient information has been gathered yet."
        
        summary_parts = ["📋 **Patient Information Summary:**\n"]
        
        if self.current_patient["name"]:
            summary_parts.append(f"👤 **Name:** {self.current_patient['name']}")
        
        if self.current_patient["age"]:
            summary_parts.append(f"🎂 **Age:** {self.current_patient['age']} years old")
        
        if self.current_patient["gender"]:
            summary_parts.append(f"⚧ **Gender:** {self.current_patient['gender']}")
        
        if self.current_patient["symptoms"]:
            symptoms_str = ", ".join(self.current_patient["symptoms"])
            summary_parts.append(f"🩺 **Reported Symptoms:** {symptoms_str}")
        
        if self.current_patient["medical_history"]:
            history_str = ", ".join(self.current_patient["medical_history"])
            summary_parts.append(f"📜 **Medical History:** {history_str}")
        
        if self.current_patient["medications"]:
            meds_str = ", ".join(self.current_patient["medications"])
            summary_parts.append(f"💊 **Current Medications:** {meds_str}")
        
        if self.current_patient["additional_info"]:
            additional_items = [f"{k}: {v}" for k, v in self.current_patient["additional_info"].items()]
            if additional_items:
                summary_parts.append(f"ℹ️ **Additional Information:** {', '.join(additional_items)}")
        
        # Add extraction method info
        summary_parts.append(f"\n🔍 **Extraction Method:** 🤖 Real-time LLM extraction")
        summary_parts.append(f"💬 **QA Pairs Processed:** {self.current_patient['qa_pairs_count']}")
        summary_parts.append(f"⏰ **Consultation Time:** {self.current_patient['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(summary_parts)
    
    def _save_patient_to_database(self) -> Dict[str, Any]:
        """Save patient information to MongoDB when conversation ends"""
        if not self.mongo_client:
            return {
                "success": False,
                "message": "No MongoDB client configured",
                "patient_saved": False
            }
        
        # Check if we have minimum required information
        if not (self.current_patient.get("name") or 
                self.current_patient.get("symptoms") or 
                self.current_patient.get("age")):
            return {
                "success": False,
                "message": "Insufficient patient information to save",
                "patient_saved": False
            }
        
        try:
            # Prepare patient document for MongoDB
            patient_document = {
                "name": self.current_patient.get("name", ""),
                "age": self.current_patient.get("age"),
                "gender": self.current_patient.get("gender", ""),
                "symptoms": self.current_patient.get("symptoms", []),
                "medical_history": self.current_patient.get("medical_history", []),
                "medications": self.current_patient.get("medications", []),
                "additional_info": self.current_patient.get("additional_info", {}),
                "chat_history": self._serialize_chat_history(),
                "timestamp": self.current_patient.get("timestamp", datetime.now()).isoformat(),
                "completion_notified": self.current_patient.get("completion_notified", False),
                "qa_pairs_count": self.current_patient.get("qa_pairs_count", 0),
                "extraction_performed": self.current_patient.get("extraction_performed", False)
            }
            
            # Get the database (assuming 'kami' database as shown in the test file)
            db = self.mongo_client['kami']
            collection = db['patients']
            
            # Insert the patient document
            result = collection.insert_one(patient_document)
            
            print(f"✅ Patient information saved to database with ID: {result.inserted_id}")
            
            return {
                "success": True,
                "message": f"Patient information saved successfully",
                "patient_id": str(result.inserted_id),
                "patient_saved": True
            }
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"❌ MongoDB connection error: {e}")
            return {
                "success": False,
                "message": f"Database connection error: {str(e)}",
                "patient_saved": False
            }
        except Exception as e:
            print(f"❌ Error saving patient to database: {e}")
            return {
                "success": False,
                "message": f"Error saving patient: {str(e)}",
                "patient_saved": False
            }
    
    def _serialize_chat_history(self) -> List[Dict[str, str]]:
        """Convert chat history to serializable format for database storage"""
        serialized_history = []
        for message in self.current_patient.get("chat_history", []):
            if isinstance(message, HumanMessage):
                serialized_history.append({"type": "human", "content": message.content})
            elif isinstance(message, AIMessage):
                serialized_history.append({"type": "ai", "content": message.content})
        return serialized_history


# Example usage and testing
if __name__ == "__main__":
    # Initialize the agent (you'll need to provide your OpenAI API key)
    agent = MedicalExpertAgent()
    
    print("Medical Expert Agent initialized!")
    print("To use this agent, call:")
    print("1. agent.start_conversation() - to begin")
    print("2. agent.process_user_input(user_message) - to process user input")
    print("3. agent.get_patient_info() - to get gathered information")
    print("4. agent.get_conversation_summary() - to get a summary")


