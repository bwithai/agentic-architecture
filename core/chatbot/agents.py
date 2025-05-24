from langchain_core.prompts import ChatPromptTemplate

def _create_intent_classifier(llm):
        """Create an intent classifier using LangChain"""
        intent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intent classifier for a medical assistant chatbot.
                Analyze the user's message and classify it into one of the following categories:
                1. casual_conversation: General greetings, small talk, personal questions, etc.
                2. database_query: Requests for data, information about medical records, etc.
                3. mixed: Contains elements of both casual conversation and requests for data

                Output ONLY the category name as a string, nothing else.
            """),
            ("human", "{query}")
        ])
        
        intent_chain = intent_prompt | llm
        return intent_chain


def _create_casual_conversation_chain(llm):
        """Create a chain for handling casual conversation"""
        casual_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a friendly, conversational medical assistant. 
Respond to the user's message in a warm, friendly manner. 
You can discuss general topics, provide general medical advice, and engage in casual conversation.
Keep responses concise and natural.
"""),
            ("human", "{query}")
        ])
        
        casual_chain = casual_prompt | llm
        return casual_chain