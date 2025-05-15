
    graph TD
        User([User]) --> |Query| Start([Start])
        Start --> ClassifyIntent[Classify Intent]
        
        %% Language detection and translation
        ClassifyIntent --> |Detect Language| LangDetect[Language Detection]
        LangDetect --> |If not English| Translate1[Translate to English]
        Translate1 --> ClassifyIntent
        
        %% Intent classification
        ClassifyIntent --> Decision{Intent Type?}
        Decision --> |GENERAL_CONVERSATION| ConvResponse[Generate Conversation Response]
        Decision --> |BUSINESS_INQUIRY| QueryUnderstanding[Query Understanding]
        
        %% For general conversation, translate back if needed
        ConvResponse --> |If not English| Translate2[Translate to Original Language]
        Translate2 --> End1([End])
        ConvResponse --> |If English| End1
        
        %% For business inquiries
        QueryUnderstanding --> |Create MongoDB Query| MongoDB[(MongoDB)]
        MongoDB --> |Query Results| FormatResponse[Format Response]
        
        %% Format and translate response
        FormatResponse --> |Format in English| TranslateResult[Translate to Original Language]
        TranslateResult --> End2([End])
        
        %% State flow
        subgraph State[State Management]
            LanguageState[Language Preferences]
            QueryState[Query Information]
            ResultsState[Query Results]
        end
        
        %% Tools
        subgraph Tools[Tools]
            TranslationTool[Translation Tool]
            MongoDBTool[MongoDB Tool]
        end
        
        ClassifyIntent -.-> LanguageState
        QueryUnderstanding -.-> QueryState
        MongoDB -.-> ResultsState
        ClassifyIntent -.-> TranslationTool
        FormatResponse -.-> TranslationTool
        QueryUnderstanding -.-> MongoDBTool
        
        %% Styling
        classDef agents fill:#a2d5f2,stroke:#0275d8,stroke-width:2px;
        classDef state fill:#f9d56e,stroke:#f3c623,stroke-width:2px;
        classDef tools fill:#d3f6d1,stroke:#7fb77e,stroke-width:2px;
        classDef decision fill:#ffa5a5,stroke:#ff6b6b,stroke-width:2px;
        
        class ClassifyIntent,QueryUnderstanding,ConvResponse,FormatResponse agents;
        class LanguageState,QueryState,ResultsState state;
        class TranslationTool,MongoDBTool tools;
        class Decision decision;
    