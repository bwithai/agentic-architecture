# AI Agents with LangGraph

A modular architecture for building AI agents using LangGraph to query and interact with MongoDB.

## Project Structure

```
ai-agents/
├── agents/                    # Main agents directory
│   ├── base/                 # Base classes and interfaces
│   ├── specialized/          # Specialized agents
│   ├── tools/                # Shared tools
│   └── utils/                # Utility functions
├── config/                   # Configuration files
├── core/                     # Core functionality
│   ├── graph/               # LangGraph implementations
│   └── state/               # State management
├── tests/                    # Test files
└── main.py                   # Entry point
```

## Features

- **Modular Agent Architecture**: Easily extendable with new agent types
- **LangGraph Integration**: Orchestrates agent workflows
- **MongoDB Connectivity**: Retrieves data from MongoDB based on natural language queries
- **Human-Friendly Responses**: Formats data into readable, conversational outputs

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