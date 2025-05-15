import os
import json
import asyncio
import streamlit as st
import time
from dotenv import load_dotenv

# Page config MUST be the first st command
st.set_page_config(
    page_title="MongoDB AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "db_client" not in st.session_state:
    st.session_state.db_client = None
    st.session_state.tool_registry = None
    st.session_state.chat_bot = None
    st.session_state.is_initialized = False
    st.session_state.debug_logs = []
    
if "show_debug" not in st.session_state:
    st.session_state.show_debug = False

from mongodb.client import MongoDBClient
from agents.tools.registry import ToolRegistry
from main import MongoDBChatBot

# Load environment variables
load_dotenv()

# MongoDB connection string and OpenAI API key
mongodb_url = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/test")
openai_api_key = os.environ.get("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OPENAI_API_KEY environment variable not set. Please add it to your .env file.")
    st.stop()

# Simplified CSS for professional styling
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 8px;
        margin-bottom: 0.75rem;
    }
    .thinking {
        color: #6c757d;
        font-style: italic;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 3px solid #4a86e8;
    }
    .tool-call {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-family: monospace;
        border-left: 3px solid #4a86e8;
    }
    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

def log_debug_info(message):
    """Add a debug log message with timestamp"""
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.debug_logs.append(f"[{timestamp}] {message}")


# Helper function to format JSON data for better display
def format_json_output(content):
    try:
        # Check if the content is a JSON string
        if isinstance(content, str) and (content.startswith('[') or content.startswith('{')):
            data = json.loads(content)
            return st.json(data)
        return st.markdown(content)
    except:
        # If not valid JSON, just return as markdown
        return st.markdown(content)


# Initialize MongoDB and ChatBot
async def initialize_chatbot():
    log_debug_info("Initializing MongoDB connection...")
    db_client = MongoDBClient()
    await db_client.connect(mongodb_url)
    
    # Set the global db reference
    import sys
    from mongodb import client as mongodb_client_module
    mongodb_client_module.db = db_client.db
    
    # Initialize tool registry and chatbot
    log_debug_info("Setting up tool registry and chatbot...")
    tool_registry = ToolRegistry()
    chat_bot = MongoDBChatBot(
        mongodb_client=db_client,
        tool_registry=tool_registry,
        openai_api_key=openai_api_key
    )
    
    log_debug_info("Initialization complete!")
    return db_client, tool_registry, chat_bot


# Custom version of process_query to capture tool usage
async def custom_process_query(user_input):
    """Process the user input and track tool usage for debugging"""
    log_debug_info(f"Processing query: {user_input}")
    
    # Original function 
    if not st.session_state.is_initialized:
        with st.spinner("Initializing MongoDB connection..."):
            db_client, tool_registry, chat_bot = await initialize_chatbot()
            st.session_state.db_client = db_client
            st.session_state.tool_registry = tool_registry
            st.session_state.chat_bot = chat_bot
            st.session_state.is_initialized = True
    
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Create a placeholder for thinking indicator and thinking content
    thinking_placeholder = st.empty()
    thinking_placeholder.markdown('<div class="thinking">Thinking...</div>', unsafe_allow_html=True)
    
    # Placeholder for tool execution indicators
    tools_placeholder = st.empty()
    
    # Process the user message with the chatbot
    log_debug_info("Sending to AI model...")
    
    # Apply monkey patches to track tool calls
    original_execute = {}
    
    # Helper function to create logging wrapper for tools
    async def create_execute_with_logging(tool_name, original_func):
        async def execute_with_logging(*args, **kwargs):
            # Log the tool call
            log_message = f"Tool called: {tool_name} with args: {json.dumps(args[1] if len(args) > 1 else 'None')}"
            log_debug_info(log_message)
            
            # Update the tools indicator in UI
            tools_placeholder.markdown(f'<div class="tool-call">Executing: {tool_name}...</div>', unsafe_allow_html=True)
            
            # Execute the original function
            result = await original_func(*args, **kwargs)
            return result
        return execute_with_logging
    
    # Apply monkey patches to all tools
    for tool_name, tool in st.session_state.chat_bot.tool_registry._tools.items():
        original_execute[tool_name] = tool.execute
        tool.execute = await create_execute_with_logging(tool_name, tool.execute)
    
    # Capture thinking output
    thinking_output = None
    
    # Override print function to capture thinking process
    original_print = print
    def custom_print(message, *args, **kwargs):
        nonlocal thinking_output
        if isinstance(message, str) and message.startswith("Thinking process:"):
            thinking_output = message.replace("Thinking process: ", "")
            thinking_placeholder.markdown(f'<div class="thinking">{thinking_output}</div>', unsafe_allow_html=True)
        original_print(message, *args, **kwargs)
    
    # Apply print monkey patch
    import builtins
    builtins.print = custom_print
    
    try:
        # Process the query
        response = await st.session_state.chat_bot.process_query(user_input)
        
        # If we captured thinking, add it as a special message
        if thinking_output and st.session_state.show_debug:
            # Add the thinking process as a special message type
            st.session_state.messages.append({
                "role": "thinking", 
                "content": thinking_output
            })
    finally:
        # Restore original print function
        builtins.print = original_print
        
        # Remove all monkey patches
        for tool_name, original_func in original_execute.items():
            if tool_name in st.session_state.chat_bot.tool_registry._tools:
                st.session_state.chat_bot.tool_registry._tools[tool_name].execute = original_func
    
    log_debug_info("Response received from AI model")
    
    # Clear the placeholders
    thinking_placeholder.empty()
    tools_placeholder.empty()
    
    # Add assistant response to UI
    st.session_state.messages.append({"role": "assistant", "content": response})


# Run async function in Streamlit
def run_async_function(func, *args):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(func(*args))


# App Layout
st.title("MongoDB AI Assistant")

# Sidebar with controls and information
with st.sidebar:
    st.header("Controls")
    
    # Connection status indicator
    if st.session_state.is_initialized:
        st.success("✓ Connected to MongoDB")
    else:
        st.info("Not connected to MongoDB")
    
    st.divider()
    
    # Action buttons
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Debug toggle (more discreet)
    st.divider()
    st.checkbox("Show Debug Information", key="show_debug")
    
    # Example queries
    st.divider()
    st.subheader("Example Queries")
    examples = [
        "List all collections",
        "Show me all documents in users collection",
        "Count documents in products collection", 
        "Find documents with specific criteria",
        "Create a new document"
    ]
    
    for example in examples:
        if st.button(example, key=example, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": example})
            run_async_function(custom_process_query, example)
            st.rerun()

# Main content area
col1, col2 = st.columns([4, 1])

with col1:
    # Brief description
    st.caption("Ask questions about your MongoDB database using natural language")
    
    # Chat container
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "thinking" and st.session_state.show_debug:
                st.info(f"🤔 {message['content']}")
            elif message["role"] != "thinking":
                with st.chat_message(message["role"]):
                    format_json_output(message["content"])
    
    # Chat input
    user_input = st.chat_input("Ask a question about your database...")
    if user_input:
        run_async_function(custom_process_query, user_input)
        st.rerun()

# Debug panel (only shown when enabled)
if st.session_state.show_debug:
    with st.expander("Debug Logs", expanded=False):
        logs_text = "\n".join(st.session_state.debug_logs[-20:])  # Show last 20 logs
        st.code(logs_text)

# Clean up resources when the app is closed
def cleanup():
    if st.session_state.db_client:
        run_async_function(st.session_state.db_client.close)

# Register cleanup function
import atexit
atexit.register(cleanup)
