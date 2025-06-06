# MongoDB Query Agent with OpenAI

A conversational agent that allows you to query your MongoDB database using natural language, powered by OpenAI.

## Setup

1. Make sure your MongoDB container is running:
   ```
   docker ps
   ```

2. Create a `.env` file in the agents directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. Install required packages:
   ```
   pip install pymongo openai python-dotenv
   ```

## Usage

1. Run the agent:
   ```
   python mongodb_agent.py
   ```

2. Enter natural language queries like:
   - "How many users are there?"
   - "List all products"
   - "Find users where name is John"
   - "Show me 5 products with price less than 100"
   - "Get users in New York"
   - "Search for products called iPhone"

3. Type 'exit' to quit the agent.

## Features

- Natural language processing using OpenAI
- Structured query parsing with GPT models
- Query counts of documents in collections
- List documents from collections (with optional limits)
- Search for documents based on specific criteria
- Complex query handling

## Configuration

The agent connects to:
- Host: localhost
- Port: 27017
- Database: kami
- Collections: users, products

To modify these settings, edit the `__init__` method in the `MongoDBAgent` class or set them in your `.env` file. 