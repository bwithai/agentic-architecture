# AI Agents with LangGraph

A modular architecture for building AI agents using LangGraph to query and interact with MongoDB.

## Project Structure

```
ai-agents/
\── backend
|   app.py
|   get_patient_example.py
|   initial_data.py
|   main.py
|   main2.py
|   poetry.lock
|   pyproject.toml
|   requirements.txt
|   run_streamlit.py
|   utils.py
|   
+---app
|   |   __init__.py
|   |   
|   +---agents
|   |   |   mongodb_agent.py
|   |   |   README.md
|   |   |   requirements.txt
|   |   |   __init__.py
|   |   |   
|   |   +---base
|   |   |       base_agent.py
|   |   |       __init__.py
|   |   |
|   |   +---specialized
|   |   |       example_usage.py
|   |   |       medical_expert_agent.py
|   |   |       mongodb_agent.py
|   |   |       pharmacist_agent.py
|   |   |       pharmacist_usage_example.py
|   |   |       __init__.py
|   |   |
|   |   +---tools
|   |   |   |   registry.py
|   |   |   |   __init__.py
|   |   |   |
|   |   |   +---base
|   |   |   |       tool.py
|   |   |   |       __init__.py
|   |   |   |
|   |   |   +---collection
|   |   |   |       list_collections.py
|   |   |   |
|   |   |   +---documents
|   |   |   |       count.py
|   |   |   |       delete_one.py
|   |   |   |       find.py
|   |   |   |       insert_one.py
|   |   |   |       update_one.py
|   |   |   |
|   |   |   +---indexes
|   |   |   |       create_index.py
|   |   |   |       drop_index.py
|   |   |   |       list_indexes.py
|   |   |   |
|   |   |   \---patient
|   |   |           create_patient_profile.py
|   |   |           get_patient.py
|   |   |           __init__.py
|   |   |
|   |   \---utils
|   |           serialization_utils.py
|   |           translation_utils.py
|   |           __init__.py
|   |
|   +---api
|   +---core
|   |   |   config.py
|   |   |   db.py
|   |   |   __init__.py
|   |   |
|   |   +---graph
|   |   |       agent_graph.py
|   |   |       __init__.py
|   |   |
|   |   +---moderation
|   |   |       fallback_handler.py
|   |   |       __init__.py
|   |   |
|   |   +---state
|   |   |       agent_state.py
|   |   |       __init__.py
|   |   |
|   |   \---utils
|   \---mongodb
|           client.py
|           mongodb_setup.py
|           schema.py
|           __init__.py
|
+---email-templates
|   \---build
|           fallback_support.html
|           medical_consultation_summary.html
|           test_email.html
|
\---test_logics
        check_mongodb.py
        insert_patient_info.py
        remove_pycache.py
        test_imports.py
```

## Features

- Connect to any MongoDB database using a standard connection URI
- Natural language queries translated into MongoDB operations
- Support for all common MongoDB operations:
  - Listing collections
  - Querying documents with filters
  - Inserting documents
  - Updating documents
  - Deleting documents
  - Creating and listing indexes
  - Schema inference
  
## Getting Started

1. **Setup Environment**:
   ```bash
   # Create a virtual environment
   python -m venv .venv
   
   # Activate the virtual environment
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   
   # Install dependencies
   poetry install
   ```

2. **Configure Settings**:
   Create a `.env` file based on `.env.example` and add your MongoDB and OpenAI credentials.

3. **Run the Application**:
   ```bash
   python main.py
   ```

## Configuration

The application uses the OpenAI API for natural language processing. You'll need:

1. An OpenAI API key (get one at https://platform.openai.com/api-keys)
2. A MongoDB connection string (from your MongoDB provider or local instance)

## Example Commands

Once the chatbot is running, you can interact with it using natural language:

- "Show me all collections in this database"
- "Find users in San Francisco"
- "Insert a new user named John with email john@example.com"
- "What is the schema of the products collection?"
- "Update the email for user with id 12345 to newemail@example.com"
- "Delete all orders with status 'cancelled'"
- "Create an index on the email field in the users collection"

Type 'exit' or 'quit' to end the session.

## Agent Flow

1. **Query Understanding Agent**: Parses natural language queries into structured MongoDB operations
2. **MongoDB Agent**: Executes the structured queries against MongoDB
3. **Response Formatting Agent**: Converts raw data into human-friendly responses

## Extending the System

To add new agent types:

1. Create a new class in `agents/specialized/` that inherits from `BaseAgent`
2. Implement the required methods (`run()` and `get_description()`)
3. Update the agent graph in `core/graph/agent_graph.py` to include your new agent

## Requirements

- Python 3.12+
- OpenAI API key
- MongoDB instance 