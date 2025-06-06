import re

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from app.agents.utils.serialization_utils import serialize_mongodb_doc

# -------------------------------[Medical Expert Bot Agents]----------------------------------

def _setup_conversation_chain_prompt():
        """Setup the conversational QA chain with memory"""
        
        # System prompt for the medical expert
        medical_expert_prompt = """You are Dr. Sanaullah, a caring medical specialist. You have a reputation for being exceptionally attentive to your patients and making them feel heard. Your communication style is natural, conversational, and puts patients at ease while maintaining professionalism.

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
        conversation_prompt = ChatPromptTemplate.from_messages([
            ("system", medical_expert_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        return conversation_prompt


def _setup_extraction_chain_prompt():
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
        
    extraction_prompt = ChatPromptTemplate.from_template(extraction_prompt)

    return extraction_prompt

def _setup_flow_management_chain_prompt():
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
        
    flow_prompt = ChatPromptTemplate.from_template(flow_prompt)

    return flow_prompt

# -------------------------------[END OF Medical Expert Bot BASE AGENT]----------------------------

def _create_patient_profile_chain(llm):
    """Create a chain for handling patient profile creation"""
    patient_profile_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a medical expert, patient come to you for treatment assest the patient and collect the following information:

**CONVERSATION APPROACH:**
- Start with a warm, natural greeting
- Ask about symptoms in a caring, conversational way
- Listen actively and respond empathetically
- Ask follow-up questions naturally
- Collect information gradually through conversation
         
**INFORMATION TO GATHER (naturally through conversation):**
1. **Symptoms**: What they're experiencing, where, how long, severity
2. **Basic Info**: Name, age, gender, contact details
3. **Additional**: Any other symptoms or concerns
        """),
        ("human", "{query}")
    ])
    patient_profile_chain = patient_profile_prompt | llm
    return patient_profile_chain

def _create_intent_classifier(llm):
    """Create an intent classifier using LangChain"""
    intent_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intent classifier for a medical assistant chatbot.
Analyze the user's message and classify it into one of the following categories:
1. casual_conversation: General greetings, small talk, personal questions, etc.
2. database_query: Requests for data, information about medical records, etc.
3. mixed: Contains elements of both casual conversation and requests for data

Output ONLY the category name as a string, nothing else.
"""),
        ("human", "{query}")
    ])
    
    intent_chain = intent_prompt | llm
    return intent_chain

def _create_casual_conversation_chain(llm):
    """Create a chain for handling casual conversation"""
    casual_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a friendly, conversational medical assistant. 
Respond to the user's message in a warm, friendly manner. 
You can discuss general topics, provide general medical advice, and engage in casual conversation.
Keep responses concise and natural.
"""),
        ("human", "{query}")
    ])
    
    casual_chain = casual_prompt | llm
    return casual_chain

async def _evaluate_response_quality(llm, query: str, response: str) -> float:
    """
    Evaluate the quality of the response relative to the user query.
    Returns a confidence score (0-1) indicating how well the response addresses the query.
    """
    # Create a prompt to ask the LLM to evaluate the response
    evaluation_prompt = f"""
    You are strictly evaluating the quality of an AI assistant's response to a user query.
    
    User Query: {query}
    
    Assistant Response: {response}
    
    On a scale of 0 to 1, how well did the assistant's response answer the user's query?
    Consider the following factors:
    - Need: Strictly evaluating the AI assistant's response to a user need
    - Relevance: Did the response address what the user was asking about?
    - Completeness: Did the response provide all the information the user requested?
    - Accuracy: Is the information provided likely to be correct based on available data?
    - Clarity: Is the response clear and understandable?
    
    Return only a single decimal number between 0 and 1 representing your confidence score.
    If the response indicates the assistant couldn't find information or returned empty/null results, give a lower score.
    """
    
    # Get the evaluation
    evaluation_response = await llm.ainvoke([HumanMessage(content=evaluation_prompt)])
    
    # Try to extract a confidence score (float between 0-1)
    try:
        # Strip any non-numeric characters and convert to flgoat
        confidence_text = evaluation_response.content.strip()
        # Extract the first decimal number from the response
        confidence_match = re.search(r'0\.\d+|\d+\.\d+|0|1', confidence_text)
        if confidence_match:
            confidence = float(confidence_match.group())
            # Ensure it's between 0 and 1
            confidence = max(0.0, min(1.0, confidence))
            return confidence
        else:
            # Default moderate confidence if no number found
            print(f"Could not extract confidence score from: {confidence_text}")
            return 0.5
    except Exception as e:
        print(f"Error parsing confidence score: {e}, defaulting to 0.5")
        return 0.5
        
def _serialize_conversation(messages):
    """Serialize MongoDB objects in the conversation for proper JSON conversion."""
    serialized_conversation = []
    
    for msg in messages:
        # Handle messages with tool calls
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            # Create new message with serialized tool calls
            serialized_msg = AIMessage(
                content=msg.content,
                tool_calls=[
                    {
                        "name": tc["name"],
                        "id": tc["id"], 
                        "args": serialize_mongodb_doc(tc["args"])
                    }
                    for tc in msg.tool_calls
                ]
            )
            serialized_conversation.append(serialized_msg)
        # Handle tool messages with tool_call_id
        elif hasattr(msg, 'tool_call_id') and msg.tool_call_id:
            # Make sure any MongoDB objects in content are serialized
            if isinstance(msg.content, str):
                # Keep as is
                serialized_conversation.append(msg)
            else:
                # Serialize non-string content
                serialized_content = serialize_mongodb_doc(msg.content)
                from langchain_core.messages import ToolMessage
                serialized_msg = ToolMessage(
                    content=str(serialized_content),
                    tool_call_id=msg.tool_call_id
                )
                serialized_conversation.append(serialized_msg)
        else:
            # No tool calls, just add the original message
            serialized_conversation.append(msg)
            
    return serialized_conversation 