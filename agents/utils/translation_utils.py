"""
Translation Utilities

This module provides utilities for language detection and translation using LLM.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from config.config import config
import json

# Create a logger
logger = logging.getLogger(__name__)

# Common language codes for reference
LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "bn": "Bengali",
    "ur": "Urdu"
}

async def detect_language(text: str) -> Tuple[str, float, str]:
    """
    Detect the language of the input text using LLM.
    
    Args:
        text (str): The text to detect language
        
    Returns:
        Tuple[str, float]: A tuple containing language code and confidence score
    """
    try:
        # Initialize LLM
        llm = ChatOpenAI(
            model=config.openai.model,
            temperature=0.0,
            api_key=config.openai.api_key
        )
        
        # Create prompt for language detection with no template variables
        system_message = """
            You are a language detection specialist. Your task is to identify the language of the given text.
            Respond with a JSON containing:
            - language_code: ISO 639-1 language code (2 letters like 'en', 'es', 'fr')
            - language_name: Full name of the language in English
            - confidence: Your confidence level from 0.0 to 1.0
            
            Only respond with valid JSON. For example:
            {"language_code": "en", "language_name": "English", "confidence": 0.98}
        """
        
        # Direct approach without template variables
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=f"Detect the language of this text: {text}")
        ]
        
        # Get response directly
        response = await llm.ainvoke(messages)

        # Parse JSON from content
        try:
            result = json.loads(response.content)
            lang_code = result.get("language_code")
            confidence = result.get("confidence")
            language_name = result.get("language_name")
        except Exception as parse_error:
            logger.error(f"Error parsing language detection response: {parse_error}")
            # Default to English on failure
            lang_code = "en" 
            confidence = 0.5
            language_name = "English"
        
        # Log the detected language
        logger.info(f"LLM detected language: {lang_code} with confidence {confidence} for text: {text[:50]}...")
        
        return lang_code, confidence, language_name
    except Exception as e:
        logger.error(f"Error detecting language with LLM: {e}")
        # Default to English on failure
        return "en", 0.0, "English"

async def translate_text(text: str, source_lang_name: str = "auto", target_lang_name: str = "en") -> str:
    """
    Translate text from source language to target language using LLM.
    
    Args:
        text (str): Text to translate
        source_lang_name (str): Source language name (default: auto-detect)
        target_lang_name (str): Target language name (default: English)
        
    Returns:
        str: Translated text
    """
    try:
        # Skip if same language or empty text
        if source_lang_name == target_lang_name or not text:
            return text
        
        # Initialize LLM
        llm = ChatOpenAI(
            model=config.openai.model,
            temperature=0.1,  # Slightly higher for translation creativity
            api_key=config.openai.api_key
        )
        
        # Direct approach without template variables
        from langchain_core.messages import SystemMessage, HumanMessage
        
        system_content = f"""
            You are a professional translator specializing in translation between different languages.
            Translate the provided text to {target_lang_name}.
            Maintain the original meaning, tone, and format.
            Respond only with the translated text, nothing else.
        """
        
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=f"Translate this text: {text}")
        ]
        
        # Get response directly
        response = await llm.ainvoke(messages)
        translated_text = response.content
        
        # Log the translation
        logger.info(f"LLM translated from {source_lang_name} to {target_lang_name}: {text[:30]}... -> {translated_text[:30]}...")
        
        return translated_text
    except Exception as e:
        logger.error(f"LLM translation error: {e}")
        # Return original text on failure
        return text


def is_english(lang_code: str) -> bool:
    """
    Check if the detected language is English.
    
    Args:
        lang_code (str): Language code
        
    Returns:
        bool: True if language is English, False otherwise
    """
    return lang_code.lower() in ["en", "eng", "english"] 