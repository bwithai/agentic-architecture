"""
Translation Tools

This module provides translation tools for agents to use.
"""

from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field
from agents.utils.translation_utils import (
    detect_language, 
    translate_text,
    is_english
)
from core.state.agent_state import AgentState


class TranslationTool:
    """
    Tool for language detection and translation using LLM capabilities.
    """
    
    @staticmethod
    async def detect_query_language(query: str, state: AgentState) -> Tuple[str, str, float]:
        """
        Detect the language of a user query and update state.
        
        Args:
            query (str): The user query to analyze
            state (AgentState): The current agent state
            
        Returns:
            Tuple[str, str, float]: Language code, language name, and confidence
        """
        try:
            lang_code, confidence, lang_name = await detect_language(query)
            
            # Update user language preference in state
            state.update_language_preference(lang_code, lang_name, confidence)
            
            return lang_code, lang_name, confidence
        except Exception as e:
            # If language detection fails, default to English
            state.update_language_preference("en", "English", 1.0)
            return "en", "English", 1.0
    
    @staticmethod
    async def translate_to_english(text: str, source_lang_code: str, source_lang_name: str) -> str:
        """
        Translate text to English using LLM.
        
        Args:
            text (str): The text to translate
            source_lang_code (str): Source language code
            source_lang_name (str): Source language name
            
        Returns:
            str: Translated text in English
        """
        try:
            if is_english(source_lang_code):
                return text
                
            return await translate_text(text, source_lang_name=source_lang_name, target_lang_name="English")
        except Exception as e:
            # If translation fails, return original text
            return text
    
    @staticmethod
    async def translate_from_english(
        text: str, 
        target_lang: str
    ) -> str:
        """
        Translate text from English to target language using LLM.
        
        Args:
            text (str): The English text to translate
            target_lang (str): Target language code
            
        Returns:
            str: Translated text
        """
        try:
            if is_english(target_lang):
                return text
                
            return await translate_text(text, source_lang="en", target_lang=target_lang)
        except Exception as e:
            # If translation fails, return original text
            return text
    
    @staticmethod
    async def get_translation_info(state: AgentState) -> Dict[str, Any]:
        """
        Get information about the current translation settings.
        
        Args:
            state (AgentState): The current agent state
            
        Returns:
            Dict[str, Any]: Information about translation settings
        """
        lang_pref = state.language_preference
        return {
            "language_code": lang_pref.language_code,
            "language_name": lang_pref.language_name,
            "confidence": lang_pref.confidence,
            "last_updated": lang_pref.last_updated.isoformat(),
            "is_english": is_english(lang_pref.language_code)
        } 