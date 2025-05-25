import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from agents.utils.serialization_utils import serialize_mongodb_doc

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