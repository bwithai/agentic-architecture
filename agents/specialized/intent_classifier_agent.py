"""
Intent Classifier Agent

This agent determines whether a user query is:
1. GENERAL_CONVERSATION: casual chat, greetings, small talk.
2. BUSINESS_INQUIRY: questions requiring database access or business logic.

It also detects the language of the query and handles translation if needed.
"""

from typing import Dict, Any, Tuple

from pydantic import Field
from agents.base.base_agent import BaseAgent, AgentInput, AgentOutput
from config.config import config
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agents.tools.translation_tools import TranslationTool
from agents.utils.translation_utils import is_english


class IntentClassifierAgent(BaseAgent):
    """
    Agent that classifies user input as general conversation or business-related,
    using a hybrid rule + few-shot LLM approach. Also detects language and translates
    non-English queries to English for processing.
    """
    
    def __init__(self, model_name: str = None, verbose: bool = False):
        super().__init__(name="intent_classifier", verbose=verbose)
        self.model_name = model_name or config.openai.model
        self._initialize_components()
        self.translation_tool = TranslationTool()
        self.log("Initialized Intent Classifier Agent")
    
    def _initialize_components(self):
        """Initialize LLM, prompts, parser, and rule patterns."""

        # === 1. Few-shot examples for classification ===
        few_shot_examples = [
            {"query": "Hello, how are you?",                 "label": "GENERAL_CONVERSATION"},
            {"query": "What's the weather like?",            "label": "GENERAL_CONVERSATION"},
            {"query": "Show me our Q1 revenue report",       "label": "BUSINESS_INQUIRY"},
            {"query": "I need pricing info on product X",    "label": "BUSINESS_INQUIRY"},
            {"query": "List the first 5 users",              "label": "BUSINESS_INQUIRY"},
        ]
        examples_str = "\n".join(f"{e['query']} -> {e['label']}" for e in few_shot_examples)
        classification_system = f"""
            You are an AI assistant that classifies messages into one of two categories:
            
            1. GENERAL_CONVERSATION: casual chat, greetings, small talk.
            2. BUSINESS_INQUIRY: questions requiring access to business data (e.g., products, users, orders).
            
            Here are examples:
            {examples_str}
            
            Now classify only the label (GENERAL_CONVERSATION or BUSINESS_INQUIRY) for the new user message.
        """
        # === 2. LLM and prompt chain ===
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0.0,            # zero for deterministic output
            api_key=config.openai.api_key
        )
        self.classification_prompt = ChatPromptTemplate.from_messages([
            ("system", classification_system.strip()),
            ("human", "{query}")
        ])
        self.classification_chain = (
            self.classification_prompt 
            | self.llm 
            | StrOutputParser()           # enforces exact label
        )

        # === 3. Response chain for general conversation ===
        response_system = """
You are a friendly, professional assistant. Respond concisely and helpfully to the user's chat message.
"""
        self.response_prompt = ChatPromptTemplate.from_messages([
            ("system", response_system.strip()),
            ("human", "{query}")
        ])
        self.response_chain = self.response_prompt | self.llm | StrOutputParser()
    
    async def detect_language_and_translate(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Detect the language of the query and translate to English if needed.
        
        Args:
            query (str): The user query
            
        Returns:
            Tuple[str, Dict[str, Any]]: Translated query (if needed) and language info
        """
        # Create dummy state for language detection
        from core.state.agent_state import AgentState
        dummy_state = AgentState(session_id="temp")
        
        # Detect language
        lang_code, lang_name, confidence = await self.translation_tool.detect_query_language(
            query, dummy_state
        )
        
        # Translate if needed
        if not is_english(lang_code):
            translated_query = await self.translation_tool.translate_to_english(query, lang_code, lang_name)
            self.log(f"Translated query from {lang_name} to English: {translated_query}")
        else:
            translated_query = query
            
        return translated_query, {
            "language_code": lang_code,
            "language_name": lang_name,
            "confidence": confidence,
            "original_query": query,
            "is_english": is_english(lang_code),
            "translated_query": translated_query
        }

    async def run(self, inputs: AgentInput) -> AgentOutput:
        """
        Classify the intent and, if it's general conversation, generate a reply.
        Also detect language and translate if needed.
        """
        query = inputs.query.strip()
        try:
            # 1. Detect language and translate if needed
            translated_query, language_info = await self.detect_language_and_translate(query)

            # 2. LLM classification on translated query
            raw_label = await self.classification_chain.ainvoke({"query": translated_query})
            classification = raw_label.strip().upper()
            self.log(f"LLM classified intent as: {classification}")

            # Fallback safety
            if classification not in {"GENERAL_CONVERSATION", "BUSINESS_INQUIRY"}:
                self.log(f"Unexpected label '{classification}', defaulting to GENERAL_CONVERSATION")
                classification = "GENERAL_CONVERSATION"

            # 4. Generate a chat response if appropriate (in English)
            response = ""
            if classification == "GENERAL_CONVERSATION":
                raw_resp = await self.response_chain.ainvoke({"query": language_info.get("original_query")})
                response = raw_resp.strip()

            # 5. Return structured output with language info
            return AgentOutput(
                response=response,
                data={
                    "classification": {"intent_type": classification},
                    "language_info": language_info
                },
                status="success"
            )

        except Exception as e:
            error_msg = f"Error in IntentClassifierAgent: {e}"
            self.log(error_msg)
            return AgentOutput(
                response="",
                data={},
                status="error",
                error=error_msg
            )
    
    def get_description(self) -> str:
        return "Classifies user input as general conversation or business inquiry, with language detection and translation"
