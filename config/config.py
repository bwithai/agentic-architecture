"""
Configuration module for the AI agent application.
Manages settings for MongoDB, OpenAI, and other components.
"""

import os
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional, Dict

class MongoDBConfig(BaseModel):
    load_dotenv()
    """MongoDB connection configuration."""
    uri: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    database: str = os.getenv("MONGODB_DATABASE", "default_database")
    

class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    max_tokens: int = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))


class AgentConfig(BaseModel):
    """Agent configuration."""
    verbose: bool = os.getenv("AGENT_VERBOSE", "True").lower() == "true"
    max_iterations: int = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))
    

class AppConfig(BaseModel):
    """Main application configuration."""
    mongodb: MongoDBConfig = MongoDBConfig()
    openai: OpenAIConfig = OpenAIConfig()
    agent: AgentConfig = AgentConfig()
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"


# Create a global config instance
config = AppConfig() 