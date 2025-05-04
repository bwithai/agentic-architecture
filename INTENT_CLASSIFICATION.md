# Intent Classification System

This document explains the intent classification system added to the AI Agents application.

## Overview

The intent classification system analyzes user messages to determine whether they are:

1. **General Conversation**: Small talk, greetings, or casual exchanges
2. **Business Inquiry**: Questions about products, services, or other business-related information

## How It Works

The system uses the following flow:

1. User inputs a message
2. The intent classifier agent analyzes the message using NLP
3. Based on the classification:
   - **General Conversation**: The system responds directly with an appropriate friendly, professional reply
   - **Business Inquiry**: The system processes the query through the specialized agents (query understanding, MongoDB agent, response formatting)

## Architecture Changes

The implementation includes the following changes:

1. Uses `MessagesState` from LangGraph with LangChain message objects
2. Adds a new `IntentClassifierAgent` class
3. Modifies the agent graph to include intent classification as the first step
4. Implements conditional routing based on intent type

## LangChain Message Types

The system uses the following LangChain message types:

- `HumanMessage`: Represents user input
- `AIMessage`: Represents assistant responses
- `ToolMessage`: Represents messages from tools/agents in the system
- `SystemMessage`: Represents system instructions (not currently used)

## Examples

### General Conversation Example

**User**: "Hello there! How are you today?"

**System Process**:
1. Intent classified as GENERAL_CONVERSATION
2. System generates appropriate friendly response
3. Response returned directly to user

### Business Inquiry Example

**User**: "What services do you offer for data analytics?"

**System Process**:
1. Intent classified as BUSINESS_INQUIRY
2. Query understanding agent processes the query
3. MongoDB agent retrieves relevant information
4. Response formatting agent creates a human-friendly response
5. Final response returned to user

## Benefits

- **Improved User Experience**: More natural conversations with appropriate responses
- **Efficiency**: Simple queries are handled directly without unnecessary processing
- **Flexibility**: Easy to expand with more intent types in the future

## Implementation Note

When working with LangGraph's MessagesState, it's important to use LangChain message objects rather than simple dictionaries. Our implementation uses:

```python
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

# Creating a message
message = HumanMessage(content="Hello there!")

# Checking message type
if isinstance(message, HumanMessage):
    # Process human message
    pass
``` 