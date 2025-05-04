"""
Streamlit app for AI agents with MongoDB integration.
This app provides a user-friendly interface to query MongoDB using natural language.
"""

import os
import asyncio
import streamlit as st
import json
from datetime import datetime
from dotenv import load_dotenv
from core.graph.agent_graph import create_agent_graph
from config.config import config
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="AI MongoDB Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B67A8;
        text-align: center;
        margin-bottom: 30px;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4B67A8;
        margin-top: 20px;
    }
    .stTextInput > div > div > input {
        font-size: 1.1rem;
        padding: 15px;
    }
    .response-area {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 5px;
        border-left: 5px solid #4B67A8;
    }
    .debug-area {
        background-color: #f0f0f0;
        padding: 15px;
        border-radius: 5px;
        font-family: monospace;
        margin-top: 20px;
    }
    .tool-message {
        background-color: #e8f4f8;
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
        font-size: 0.9rem;
        margin-top: 5px;
        margin-bottom: 5px;
        border-left: 3px solid #4B9CD3;
    }
    .query-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        border: 1px solid #e0e0e0;
    }
    .stButton > button {
        background-color: #4B67A8;
        color: white;
        font-weight: bold;
        padding: 0.5rem 2rem;
        font-size: 1.1rem;
    }
    .hint-text {
        color: #666;
        font-size: 0.9rem;
        font-style: italic;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'need_rerun' not in st.session_state:
    st.session_state.need_rerun = False
if 'last_query' not in st.session_state:
    st.session_state.last_query = None
if 'show_tool_messages' not in st.session_state:
    st.session_state.show_tool_messages = False

# Custom JSON encoder for MongoDB types
class MongoJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# Function to run the agent asynchronously
async def run_agent_async(query):
    """Run the agent with the given query asynchronously."""
    # Create the agent graph
    agent_graph = create_agent_graph()
    
    # Initialize message state with the user query using LangChain message objects
    initial_state = {
        "messages": [
            HumanMessage(content=query)
        ]
    }
    
    # Run the agent with the query
    return await agent_graph.ainvoke(initial_state)

# Function to bridge async and sync for Streamlit
def run_agent(query):
    """Run the agent in a way compatible with Streamlit's synchronous model."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(run_agent_async(query))
    finally:
        loop.close()

# Function to process query and trigger rerun
def process_query(query):
    # Store query for processing
    st.session_state.last_query = query
    # Set flag to indicate we need to rerun
    st.session_state.need_rerun = True
    # Rerun to process the query
    st.rerun()

# Extract the final response from the MessagesState result
def get_final_response(result):
    """Extract the final assistant response from the agent result."""
    assistant_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
    if assistant_messages:
        return assistant_messages[-1].content
    return "No response generated"

# Main app layout
st.markdown('<h1 class="main-header">AI MongoDB Assistant</h1>', unsafe_allow_html=True)

# Sidebar for configuration and information
with st.sidebar:
    st.image("https://raw.githubusercontent.com/mongodb/mongo/master/docs/leaf.svg", width=100)
    st.markdown("## Configuration")
    
    # Display current MongoDB settings
    st.markdown("### MongoDB Settings")
    st.text(f"Database: {config.mongodb.database}")
    
    # Option to change database
    new_db = st.text_input("Change database:", config.mongodb.database)
    if new_db != config.mongodb.database:
        config.mongodb.database = new_db
        st.success(f"Database changed to {new_db}")
    
    # Add debug mode toggle
    debug_mode = st.checkbox("Debug Mode", value=False)
    
    # Toggle to show tool messages
    st.session_state.show_tool_messages = st.checkbox("Show Agent Thought Process", value=st.session_state.show_tool_messages)
    
    st.markdown("### Examples")
    example_queries = [
        "Hello, how are you today?",  # General conversation
        "Show me all documents from the users collection",  # Business inquiry
        "Find users with email containing gmail.com",  # Business inquiry
        "Count how many documents are in the products collection",  # Business inquiry
        "Show me the most recent 5 users"  # Business inquiry
    ]
    
    for query in example_queries:
        if st.button(query):
            process_query(query)

# Process the stored query if needed
if st.session_state.need_rerun and st.session_state.last_query:
    user_query = st.session_state.last_query
    
    # Show spinner while processing
    with st.spinner("Processing your query..."):
        # Add user query to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Run the agent
        result = run_agent(user_query)
        
        # Extract final response and store all messages for debug
        final_response = get_final_response(result)
        
        # Add agent response to chat history
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": final_response,
            "full_messages": result["messages"] if debug_mode else None
        })
    
    # Reset for next query
    st.session_state.need_rerun = False
    st.session_state.last_query = None

# Display chat history
if st.session_state.chat_history:
    st.markdown('<h2 class="sub-header">Conversation</h2>', unsafe_allow_html=True)

    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown('<div class="response-area">', unsafe_allow_html=True)
            st.markdown(f"**AI:** {message['content']}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display tool messages if enabled and available
            if st.session_state.show_tool_messages and message.get("full_messages"):
                for msg in message["full_messages"]:
                    if isinstance(msg, ToolMessage):
                        st.markdown(
                            f'<div class="tool-message">[{msg.tool_call_id}] {msg.content}</div>', 
                            unsafe_allow_html=True
                        )
            
        # Add a small space between messages
        st.markdown("")

# Debug area
if debug_mode and st.session_state.chat_history and 'result' in locals():
    st.markdown('<h2 class="sub-header">Debug Information</h2>', unsafe_allow_html=True)
    with st.expander("View Debug Information", expanded=False):
        st.markdown('<div class="debug-area">', unsafe_allow_html=True)
        
        # Show message sequence
        st.markdown("### Message Sequence")
        for i, msg in enumerate(result["messages"]):
            if isinstance(msg, HumanMessage):
                st.markdown(f"{i}: **Human**: {msg.content}")
            elif isinstance(msg, AIMessage):
                st.markdown(f"{i}: **AI**: {msg.content}")
            elif isinstance(msg, ToolMessage):
                st.markdown(f"{i}: **Tool [{msg.tool_call_id}]**: {msg.content}")
            else:
                st.markdown(f"{i}: **{type(msg).__name__}**: {msg.content}")
            
        st.markdown('</div>', unsafe_allow_html=True)

# Input area - Moved to the bottom and styled better
st.markdown('<h2 class="sub-header">Ask a Question</h2>', unsafe_allow_html=True)

st.markdown('<div class="query-container">', unsafe_allow_html=True)
st.markdown('<p class="hint-text">Ask anything! You can chat casually or ask specific questions about your MongoDB data. Try questions like "How many users are in the database?" or "Show me products with price greater than 100"</p>', unsafe_allow_html=True)

# Use columns for better layout
col1, col2 = st.columns([4, 1])

with col1:
    # Use text area for more space to write longer queries
    user_query = st.text_area(
        "Your question:",
        height=100,
        placeholder="Enter your question here...",
        label_visibility="hidden",
        key="query_input"
    )

with col2:
    # Add some space to align the button vertically
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 Send", key="search_button", use_container_width=True):
        if user_query:
            process_query(user_query)
        else:
            st.warning("Please enter a question first.")

st.markdown('</div>', unsafe_allow_html=True) 