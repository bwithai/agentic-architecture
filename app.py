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
    st.session_state.has_test_data = False
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

# Custom CSS for better ChatGPT-like styling
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .stChatMessage[data-testid="user-message"] {
        background-color: #f7f7f8;
    }
    .stChatMessage[data-testid="assistant-message"] {
        background-color: #f0f7fb;
    }
    .mongo-json {
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 5px;
        font-family: monospace;
        overflow-x: auto;
    }
    .thinking {
        color: #6c757d;
        font-style: italic;
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 3px solid #17a2b8;
    }
    .thinking-message {
        background-color: #f8f9fa;
        border-left: 3px solid #17a2b8;
        font-style: italic;
        color: #6c757d;
    }
    .tool-call {
        background-color: #e6f3e6;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-family: monospace;
        border-left: 3px solid #28a745;
    }
    .stButton button {
        width: 100%;
    }
    .debug-panel {
        background-color: #f6f6f6;
        border-radius: 5px;
        padding: 10px;
        max-height: 200px;
        overflow-y: auto;
    }
    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Sample data for initializing the database
SAMPLE_DATA = {
    "users": [
        {"name": "Alice", "age": 28, "email": "alice@example.com", "roles": ["admin", "user"]},
        {"name": "Bob", "age": 35, "email": "bob@example.com", "roles": ["user"]},
        {"name": "Charlie", "age": 42, "email": "charlie@example.com", "roles": ["developer", "user"]},
        {"name": "David", "age": 24, "email": "david@example.com", "roles": ["user"]}
    ],
    "products": [
        {"name": "Laptop", "price": 999.99, "category": "Electronics", "in_stock": True},
        {"name": "Smartphone", "price": 699.99, "category": "Electronics", "in_stock": True},
        {"name": "Headphones", "price": 149.99, "category": "Accessories", "in_stock": False},
        {"name": "Monitor", "price": 249.99, "category": "Electronics", "in_stock": True},
        {"name": "Keyboard", "price": 79.99, "category": "Accessories", "in_stock": True}
    ],
    "orders": [
        {"user_id": "alice@example.com", "products": ["Laptop", "Headphones"], "total": 1149.98, "date": "2023-11-15"},
        {"user_id": "bob@example.com", "products": ["Smartphone"], "total": 699.99, "date": "2023-11-16"},
        {"user_id": "charlie@example.com", "products": ["Monitor", "Keyboard"], "total": 329.98, "date": "2023-11-17"}
    ]
}


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
            # Format as a pretty table if it's a list of dictionaries
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                return st.json(data)
            # Otherwise format as pretty JSON
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


# Function to initialize database with sample data
async def initialize_database():
    log_debug_info("Starting database initialization...")
    if not st.session_state.is_initialized:
        db_client, tool_registry, chat_bot = await initialize_chatbot()
        st.session_state.db_client = db_client
        st.session_state.tool_registry = tool_registry
        st.session_state.chat_bot = chat_bot
        st.session_state.is_initialized = True
    
    # Insert sample data into collections
    for collection_name, documents in SAMPLE_DATA.items():
        # Drop existing collection if it exists
        log_debug_info(f"Dropping collection: {collection_name}")
        await st.session_state.db_client.db.drop_collection(collection_name)
        
        # Insert documents
        if documents:
            log_debug_info(f"Inserting {len(documents)} documents into {collection_name}")
            await st.session_state.db_client.db[collection_name].insert_many(documents)
    
    st.session_state.has_test_data = True
    log_debug_info("Database initialization complete!")
    return "Database initialized with sample data for users, products, and orders collections."


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
        if thinking_output:
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


# Layout: Two columns - main chat and sidebar
main_col, sidebar_col = st.columns([3, 1])

with main_col:
    # App title and description
    st.title("MongoDB AI Assistant")
    st.markdown("Chat with your MongoDB database using natural language!")
    
    # Main chat interface
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "thinking":
                # Display thinking messages with a different style
                st.markdown(f'<div class="thinking-message">🤔 {message["content"]}</div>', unsafe_allow_html=True)
            else:
                # Display regular user/assistant messages
                with st.chat_message(message["role"]):
                    format_json_output(message["content"])
    
    # Debug panel (collapsible)
    if st.session_state.show_debug:
        with st.expander("Debug Logs", expanded=True):
            logs_text = "\n".join(st.session_state.debug_logs[-20:])  # Show last 20 logs
            st.code(logs_text)
    
    # Chat input
    user_input = st.chat_input("Ask me about your MongoDB database...")
    if user_input:
        run_async_function(custom_process_query, user_input)
        # Rerun to update UI immediately
        st.rerun()


# Sidebar with MongoDB info and buttons
with sidebar_col:
    st.header("MongoDB Connection")
    st.write(f"Connection: {mongodb_url}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("Load Test Data"):
            result = run_async_function(initialize_database)
            st.success(result)
            # Add system message about initialization
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "I've initialized the database with sample data. You can now query the users, products, and orders collections!"
            })
            time.sleep(1)
            st.rerun()
    
    # Connection status
    if st.session_state.is_initialized:
        st.success("Connected to MongoDB")
        if st.session_state.has_test_data:
            st.info("Test data loaded")
    else:
        st.info("Not connected to MongoDB yet")
    
    # Debug toggle
    st.checkbox("Show Debug Panel", key="show_debug")
    
    st.write("---")
    st.markdown("## Example Queries")
    st.markdown("- List all collections")
    st.markdown("- Show me users older than 30")
    st.markdown("- Count products in Electronics category")
    st.markdown("- What's the total value of all orders?")
    st.markdown("- Find users with admin role")
    
    st.write("---")
    st.markdown("## Sample Operations")
    if st.button("Create a new user"):
        st.session_state.messages.append({
            "role": "user", 
            "content": "Insert a new user with name Eva, age 31, email eva@example.com, and roles [user, manager]"
        })
        st.rerun()
    
    if st.button("Find out-of-stock products"):
        st.session_state.messages.append({
            "role": "user", 
            "content": "Find all products that are out of stock"
        })
        st.rerun()
        

# Clean up resources when the app is closed
def cleanup():
    if st.session_state.db_client:
        run_async_function(st.session_state.db_client.close)


# Register cleanup function
import atexit
atexit.register(cleanup)
