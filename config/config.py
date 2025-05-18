"""
Configuration module for the AI agent application.
Manages settings for MongoDB, OpenAI, and other components.
"""

import os
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional, Dict

load_dotenv()

class MongoDBConfig(BaseModel):
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
    

class EmailConfig(BaseModel):
    """Email configuration."""
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "AI Support Assistant")
    EMAILS_FROM_EMAIL: str = os.getenv("EMAILS_FROM_EMAIL", "noreply@example.com")
    EMAILS_FROM_NAME: str = os.getenv("EMAILS_FROM_NAME", "AI Support Team")
    SMTP_HOST: str = os.getenv("SMTP_HOST", "")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER: str = os.getenv("SMTP_USER", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    SMTP_TLS: bool = os.getenv("SMTP_TLS", "True").lower() == "true"
    SMTP_SSL: bool = os.getenv("SMTP_SSL", "False").lower() == "true"
    SUPPORT_EMAIL: str = os.getenv("SUPPORT_EMAIL", "support@example.com")


class AppConfig(BaseModel):
    """Main application configuration."""
    mongodb: MongoDBConfig = MongoDBConfig()
    openai: OpenAIConfig = OpenAIConfig()
    agent: AgentConfig = AgentConfig()
    email: EmailConfig = EmailConfig()
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"


# Create a global config instance
config = AppConfig() 