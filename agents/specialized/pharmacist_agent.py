from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from enum import Enum
import json
import re
from bson import ObjectId
from bson.errors import InvalidId
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from agents.tools.registry import ToolRegistry
from mongodb.client import MongoDBClient
from agents.utils.serialization_utils import serialize_mongodb_doc


class SymptomMatch(BaseModel):
    """Model for LLM-based symptom analysis and scoring"""
    similarity_score: float = Field(..., description="Similarity score between 0.0 and 1.0", ge=0.0, le=1.0)
    reasoning: str = Field(..., description="Explanation of why this product matches the patient's symptoms")
    matched_symptoms: List[str] = Field(default_factory=list, description="List of patient symptoms that match this product")
    confidence: str = Field(..., description="Confidence level: high, medium, or low")


class ProductRecommendation(BaseModel):
    """Model for product recommendation with scoring"""
    product_id: str = Field(..., description="MongoDB ObjectId of the product")
    product_name: str = Field(..., description="Name of the recommended product")
    recommendation_score: float = Field(..., description="Overall recommendation score between 0.0 and 1.0", ge=0.0, le=1.0)
    symptom_match: SymptomMatch = Field(..., description="Symptom matching analysis")
    additional_factors: Dict[str, Any] = Field(default_factory=dict, description="Additional factors considered in recommendation")


class PharmacistAgent:
    def __init__(self, openai_api_key: str = None, mongodb_client: MongoDBClient = None):
        """
        Initialize the Pharmacist Agent with intelligent product recommendation capabilities
        
        Args:
            openai_api_key: OpenAI API key for the language model
            mongodb_client: MongoDB client for accessing patient and product data
        """
        self.llm = ChatOpenAI(
            temperature=0.2,  # Lower temperature for more consistent medical recommendations
            model_name="gpt-4-turbo-preview",
            openai_api_key=openai_api_key
        )
        
        # Analysis LLM with very low temperature for medical accuracy
        self.analysis_llm = ChatOpenAI(
            temperature=0.05,
            model_name="gpt-4-turbo-preview", 
            openai_api_key=openai_api_key
        )
        
        self.mongodb_client = mongodb_client
        self.tool_registry = ToolRegistry(mongodb_client) if mongodb_client else None
        
        # Setup analysis chains
        self._setup_symptom_analysis_chain()
        self._setup_product_filtering_chain()
        self._setup_recommendation_chain()
    
    def _setup_symptom_analysis_chain(self):
        """Setup LLM chain for analyzing symptom similarity"""
        
        symptom_analysis_template = """
        You are a medical AI assistant specializing in symptom analysis for pharmaceutical recommendations.
        
        **TASK**: Analyze the similarity between a patient's symptoms and a product's intended symptoms.
        
        **PATIENT SYMPTOMS AND CONTEXT**:
        {patient_symptoms}
        
        **PATIENT ADDITIONAL INFO**:
        {patient_additional_info}
        
        **PRODUCT INFORMATION**:
        Product Name: {product_name}
        Product Description: {product_description}
        Product Category: {product_category}
        
        **PRODUCT SYMPTOMS**:
        {product_symptoms}
        
        **PRODUCT COMPOSITIONS**:
        {product_compositions}
        
        **INSTRUCTIONS**:
        1. Analyze how well the patient's symptoms match the product's intended treatment symptoms
        2. Consider the patient's additional information (sleep patterns, triggers, etc.)
        3. Evaluate the product's compositions and their relevance to the patient's condition
        4. Provide a similarity score between 0.0 (no match) and 1.0 (perfect match)
        5. Explain your reasoning clearly
        6. List which specific patient symptoms match this product
        7. Assign a confidence level (high/medium/low) based on the strength of the match
        
        **SCORING GUIDELINES**:
        - 0.9-1.0: Excellent match - product directly targets patient's primary symptoms
        - 0.7-0.89: Good match - product addresses most symptoms with some relevance
        - 0.5-0.69: Moderate match - some symptoms align but not primary indication
        - 0.3-0.49: Weak match - minimal symptom overlap
        - 0.0-0.29: Poor match - little to no relevance
        
        {format_instructions}
        """
        
        self.symptom_parser = PydanticOutputParser(pydantic_object=SymptomMatch)
        
        self.symptom_analysis_prompt = ChatPromptTemplate.from_template(symptom_analysis_template)
        
        self.symptom_analysis_chain = (
            self.symptom_analysis_prompt
            | self.analysis_llm
            | self.symptom_parser
        )
    
    def _setup_product_filtering_chain(self):
        """Setup LLM chain for intelligent product filtering"""
        
        filtering_template = """
        You are a pharmaceutical AI assistant. Given a patient's condition, identify the most relevant product categories and keywords for searching.
        
        **PATIENT INFORMATION**:
        Symptoms: {patient_symptoms}
        Additional Info: {patient_additional_info}
        Age: {patient_age}
        Gender: {patient_gender}
        
        **TASK**: Generate MongoDB search filters to find relevant products.
        
        **AVAILABLE PRODUCT FIELDS**:
        - product_name: Name of the product
        - product_description: Description of the product  
        - product_category: Category (e.g., "IV", "Oral", "Injection")
        - symptoms: Array of symptom objects with icd10_code, symptom_name, symptom_description
        - compositions: Array of ingredients
        
        **INSTRUCTIONS**:
        1. Based on patient symptoms, suggest text search terms for product names/descriptions
        2. Identify relevant symptom names that should be searched
        3. Consider product categories that might be suitable
        4. Return as JSON with MongoDB query filters
        
        **RETURN FORMAT** (JSON only):
        {{
            "text_search_terms": ["term1", "term2"],
            "symptom_keywords": ["symptom1", "symptom2"],
            "category_filters": ["category1"],
            "priority_order": ["high_priority_field", "medium_priority_field"]
        }}
        """
        
        self.filtering_prompt = ChatPromptTemplate.from_template(filtering_template)
        
        self.filtering_chain = (
            self.filtering_prompt
            | self.analysis_llm
            | StrOutputParser()
        )
    
    def _setup_recommendation_chain(self):
        """Setup LLM chain for final product recommendations"""
        
        recommendation_template = """
        You are a senior pharmacist AI. Provide final product recommendations for a patient.
        
        **PATIENT PROFILE**:
        Name: {patient_name}
        Age: {patient_age}
        Gender: {patient_gender}
        Symptoms: {patient_symptoms}
        Additional Info: {patient_additional_info}
        
        **ANALYZED PRODUCTS WITH SCORES**:
        {analyzed_products}
        
        **TASK**: 
        1. Rank the top 3-5 most suitable products
        2. Provide clear medical reasoning for each recommendation
        3. Consider contraindications based on patient info
        4. Explain why certain products rank higher than others
        
        **RETURN**: A clear, professional pharmaceutical consultation report.
        """
        
        self.recommendation_prompt = ChatPromptTemplate.from_template(recommendation_template)
        
        self.recommendation_chain = (
            self.recommendation_prompt
            | self.llm
            | StrOutputParser()
        )

    async def get_patient_by_id(self, patient_id: str) -> Dict[str, Any]:
        """
        Fetch patient data by MongoDB ObjectID using the patient tool
        
        Args:
            patient_id: MongoDB ObjectID as string
            
        Returns:
            Dictionary containing patient data or error info
        """
        try:
            if not self.tool_registry:
                return {"error": "Tool registry not initialized"}
            
            # Get the patient tool and execute it
            patient_tool = self.tool_registry.get_tool("get_patient")
            result = await patient_tool.execute({"patient_id": patient_id})
            
            if result.is_error:
                error_text = "Unknown error"
                if result.content and len(result.content) > 0:
                    if isinstance(result.content[0], dict):
                        error_text = result.content[0].get("text", "Unknown error")
                    else:
                        error_text = str(result.content[0])
                return {"error": error_text}
            
            # Parse the JSON response
            if not result.content or len(result.content) == 0:
                return {"error": "No content returned from patient tool"}
            
            content_item = result.content[0]
            if not isinstance(content_item, dict):
                return {"error": f"Invalid content format: {type(content_item)}"}
            
            if "text" not in content_item:
                return {"error": f"No 'text' field in content: {content_item}"}
            
            try:
                patient_data = json.loads(content_item["text"])
                
                # Validate the structure
                if not isinstance(patient_data, dict):
                    return {"error": f"Patient data is not a dictionary: {type(patient_data)}"}
                
                return patient_data
            except json.JSONDecodeError as e:
                return {"error": f"Failed to parse JSON: {str(e)}. Raw content: {content_item['text'][:200]}..."}
            
        except Exception as e:
            return {"error": f"Failed to fetch patient: {str(e)}"}

    async def find_products_with_intelligent_filtering(self, patient_data: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """
        Use LLM to intelligently filter products based on patient information
        
        Args:
            patient_data: Patient information dictionary
            limit: Maximum number of products to retrieve
            
        Returns:
            List of relevant products
        """
        print(f"\n🔍 INTELLIGENT PRODUCT FILTERING PROCESS")
        print(f"=" * 50)
        
        try:
            # Extract patient information
            patient_symptoms = patient_data.get("patient", {}).get("symptoms", [])
            patient_additional_info = patient_data.get("patient", {}).get("additional_info", {})
            patient_age = patient_data.get("patient", {}).get("age")
            patient_gender = patient_data.get("patient", {}).get("gender")
            
            print(f"📋 Patient Profile Analysis:")
            print(f"   Symptoms: {patient_symptoms}")
            print(f"   Additional Info Keys: {list(patient_additional_info.keys())}")
            print(f"   Age: {patient_age}, Gender: {patient_gender}")
            
            # Use LLM to generate intelligent search filters
            print(f"\n🧠 Generating LLM-based search filters...")
            filter_response = await self.filtering_chain.ainvoke({
                "patient_symptoms": ", ".join(patient_symptoms) if patient_symptoms else "No specific symptoms listed",
                "patient_additional_info": json.dumps(patient_additional_info, indent=2),
                "patient_age": patient_age or "Not specified",
                "patient_gender": patient_gender or "Not specified"
            })
            
            print(f"🤖 LLM Filter Response:")
            print(f"   Raw response: {filter_response}")
            
            # Parse the LLM response to get search terms
            try:
                # Clean the response - sometimes LLM returns with markdown code blocks
                clean_response = filter_response.strip()
                if clean_response.startswith("```json"):
                    clean_response = clean_response[7:]  # Remove ```json
                if clean_response.endswith("```"):
                    clean_response = clean_response[:-3]  # Remove ```
                clean_response = clean_response.strip()
                
                search_filters = json.loads(clean_response)
                print(f"✓ Successfully parsed search filters:")
                print(f"   Text search terms: {search_filters.get('text_search_terms', [])}")
                print(f"   Symptom keywords: {search_filters.get('symptom_keywords', [])}")
                print(f"   Category filters: {search_filters.get('category_filters', [])}")
                print(f"   Priority order: {search_filters.get('priority_order', [])}")
            except json.JSONDecodeError as e:
                print(f"⚠ Failed to parse LLM response as JSON: {e}")
                print(f"   Using fallback search filters based on patient symptoms")
                # Fallback to basic search if LLM response isn't valid JSON
                search_filters = {
                    "text_search_terms": patient_symptoms[:3] if patient_symptoms else [],
                    "symptom_keywords": patient_symptoms[:5] if patient_symptoms else [],
                    "category_filters": [],
                    "priority_order": ["symptoms", "product_description", "product_name"]
                }
                print(f"   Fallback filters: {search_filters}")
            
            # Build MongoDB queries based on LLM suggestions
            print(f"\n🔎 Executing product search with generated filters...")
            products = await self._execute_product_search(search_filters, limit)
            
            print(f"📊 Search Results Summary:")
            print(f"   Total products found: {len(products)}")
            if products:
                print(f"   Product names: {[p.get('product_name', 'Unknown') for p in products[:5]]}")
                if len(products) > 5:
                    print(f"   ... and {len(products) - 5} more")
            
            return products
            
        except Exception as e:
            print(f"❌ Error in intelligent filtering: {str(e)}")
            import traceback
            traceback.print_exc()
            # Fallback to basic product search
            print(f"🔄 Falling back to basic product search...")
            return await self._fallback_product_search(patient_data, limit)

    async def _execute_product_search(self, search_filters: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """
        Execute MongoDB search based on LLM-generated filters
        
        Args:
            search_filters: LLM-generated search filters
            limit: Maximum products to return
            
        Returns:
            List of products from MongoDB
        """
        print(f"\n📋 EXECUTING PRODUCT SEARCH STRATEGIES")
        print(f"=" * 40)
        
        if not self.tool_registry:
            print("❌ Tool registry not available")
            return []
        
        find_tool = self.tool_registry.get_tool("find")
        
        # First, let's check total products in database
        try:
            total_products_result = await find_tool.execute({
                "collection": "products",
                "filter": {},
                "limit": 1
            })
            if not total_products_result.is_error:
                total_data = json.loads(total_products_result.content[0]["text"])
                total_count = total_data.get("metadata", {}).get("total_in_collection", 0)
                print(f"📊 Total products in database: {total_count}")
            else:
                print(f"⚠ Could not get total product count")
        except Exception as e:
            print(f"⚠ Error getting total count: {e}")
        
        # Try multiple search strategies in order of priority
        search_strategies = []
        
        # Strategy 1: Search by symptom keywords in symptoms array
        if search_filters.get("symptom_keywords"):
            symptom_strategy = {
                "name": "Symptom-based search",
                "collection": "products",
                "filter": {
                    "$or": [
                        {"symptoms.symptom_name": {"$regex": keyword, "$options": "i"}}
                        for keyword in search_filters.get("symptom_keywords", [])
                    ]
                },
                "limit": limit // 2
            }
            search_strategies.append(symptom_strategy)
        
        # Strategy 2: Search by text terms in product name/description
        if search_filters.get("text_search_terms"):
            text_strategy = {
                "name": "Text-based search",
                "collection": "products", 
                "filter": {
                    "$or": [
                        {"product_name": {"$regex": term, "$options": "i"}}
                        for term in search_filters.get("text_search_terms", [])
                    ] + [
                        {"product_description": {"$regex": term, "$options": "i"}}
                        for term in search_filters.get("text_search_terms", [])
                    ]
                },
                "limit": limit // 2
            }
            search_strategies.append(text_strategy)
        
        # Strategy 3: Category-based search
        if search_filters.get("category_filters"):
            category_strategy = {
                "name": "Category-based search",
                "collection": "products",
                "filter": {
                    "$or": [
                        {"product_category": {"$regex": category, "$options": "i"}}
                        for category in search_filters.get("category_filters", [])
                    ]
                },
                "limit": limit // 3
            }
            search_strategies.append(category_strategy)
        
        print(f"🎯 Planned search strategies: {len(search_strategies)}")
        for i, strategy in enumerate(search_strategies, 1):
            print(f"   {i}. {strategy['name']}")
            print(f"      Filter keys: {list(strategy['filter'].keys())}")
        
        all_products = []
        seen_ids = set()
        
        # Execute each strategy
        for i, strategy in enumerate(search_strategies, 1):
            print(f"\n🔍 Executing Strategy {i}: {strategy['name']}")
            
            try:
                # Remove strategy-specific keys for API call
                api_params = {k: v for k, v in strategy.items() if k not in ['name']}
                result = await find_tool.execute(api_params)
                
                if not result.is_error:
                    data = json.loads(result.content[0]["text"])
                    products = data.get("results", [])
                    
                    print(f"   ✓ Found {len(products)} products")
                    
                    # Add unique products
                    new_products = 0
                    for product in products:
                        product_id = product.get("_id")
                        if product_id and product_id not in seen_ids:
                            all_products.append(product)
                            seen_ids.add(product_id)
                            new_products += 1
                    
                    print(f"   ✓ Added {new_products} new unique products")
                    if products:
                        sample_names = [p.get("product_name", "Unknown") for p in products[:3]]
                        print(f"   Sample products: {sample_names}")
                else:
                    print(f"   ❌ Search failed: {result.content[0].get('text', 'Unknown error') if result.content else 'No error details'}")
                            
            except Exception as e:
                print(f"   ❌ Strategy failed: {str(e)}")
                continue
        
        # If no products found with specific filters, get a general sample
        if not all_products:
            print(f"\n🔄 No products found with specific filters, trying fallback search...")
            try:
                result = await find_tool.execute({
                    "collection": "products",
                    "filter": {},
                    "limit": limit
                })
                if not result.is_error:
                    data = json.loads(result.content[0]["text"])
                    all_products = data.get("results", [])
                    print(f"   ✓ Fallback search found {len(all_products)} products")
                else:
                    print(f"   ❌ Fallback search failed")
            except Exception as e:
                print(f"   ❌ Fallback search failed: {str(e)}")
        
        final_products = all_products[:limit]
        print(f"\n📊 SEARCH SUMMARY:")
        print(f"   Total unique products found: {len(all_products)}")
        print(f"   Products returned (limit {limit}): {len(final_products)}")
        print(f"   Search efficiency: {len(final_products)}/{total_count if 'total_count' in locals() else '?'}")
        
        return final_products

    async def _fallback_product_search(self, patient_data: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """
        Fallback method for basic product search if intelligent filtering fails
        
        Args:
            patient_data: Patient information
            limit: Maximum products to return
            
        Returns:
            List of products
        """
        if not self.tool_registry:
            return []
        
        try:
            find_tool = self.tool_registry.get_tool("find")
            result = await find_tool.execute({
                "collection": "products",
                "filter": {},
                "limit": limit
            })
            
            if not result.is_error:
                data = json.loads(result.content[0]["text"])
                return data.get("results", [])
        except Exception as e:
            print(f"Fallback search failed: {str(e)}")
        
        return []

    async def analyze_product_symptom_match(self, patient_data: Dict[str, Any], product: Dict[str, Any]) -> SymptomMatch:
        """
        Use LLM to analyze how well a product matches patient symptoms
        
        Args:
            patient_data: Patient information
            product: Product information
            
        Returns:
            SymptomMatch object with scoring and analysis
        """
        try:
            patient_info = patient_data.get("patient", {})
            patient_symptoms = patient_info.get("symptoms", [])
            patient_additional_info = patient_info.get("additional_info", {})
            
            # Format product symptoms
            product_symptoms = []
            for symptom in product.get("symptoms", []):
                symptom_text = f"- {symptom.get('symptom_name', 'Unknown')}: {symptom.get('symptom_description', 'No description')}"
                product_symptoms.append(symptom_text)
            
            # Format product compositions
            product_compositions = []
            for comp in product.get("compositions", []):
                comp_text = f"- {comp.get('ingredient_name', 'Unknown')} ({comp.get('quantity', 'N/A')} {comp.get('ingredient_unit', 'units')})"
                product_compositions.append(comp_text)
            
            # Analyze with LLM
            match_analysis = await self.symptom_analysis_chain.ainvoke({
                "patient_symptoms": ", ".join(patient_symptoms) if patient_symptoms else "No specific symptoms reported",
                "patient_additional_info": json.dumps(patient_additional_info, indent=2),
                "product_name": product.get("product_name", "Unknown Product"),
                "product_description": product.get("product_description", "No description available"),
                "product_category": product.get("product_category", "Unknown Category"),
                "product_symptoms": "\n".join(product_symptoms) if product_symptoms else "No symptoms listed",
                "product_compositions": "\n".join(product_compositions) if product_compositions else "No compositions listed",
                "format_instructions": self.symptom_parser.get_format_instructions()
            })
            
            return match_analysis
            
        except Exception as e:
            print(f"Error analyzing symptom match: {str(e)}")
            # Return a default low-confidence match in case of error
            return SymptomMatch(
                similarity_score=0.1,
                reasoning=f"Error in analysis: {str(e)}",
                matched_symptoms=[],
                confidence="low"
            )

    async def generate_product_recommendations(self, patient_data: json, max_recommendations: int = 5) -> Dict[str, Any]:
        """
        Main workflow: Get patient, find products, analyze matches, and generate recommendations
        
        Args:
            patient_data: Patient data dictionary (not patient_id)
            max_recommendations: Maximum number of products to recommend
            
        Returns:
            Dictionary containing recommendations and analysis
        """
        try:
            # Extract patient_id from patient_data if available
            patient_id = patient_data.get("_id", "unknown")
            if isinstance(patient_id, dict) and "$oid" in patient_id:
                patient_id = patient_id["$oid"]
            
            # Step 1: Find relevant products using intelligent filtering
            print("Finding relevant products...")
            products = await self.find_products_with_intelligent_filtering(patient_data, limit=100)
            
            if not products:
                return {"error": "No products found in database"}
            
            print(f"Found {len(products)} products to analyze")
            
            # Step 3: Analyze each product for symptom match
            print("Analyzing product-symptom matches...")
            analyzed_products = []
            
            for i, product in enumerate(products[:50]):  # Limit to 50 products for performance
                print(f"Analyzing product {i+1}/{min(len(products), 50)}: {product.get('product_name', 'Unknown')}")
                
                match_analysis = await self.analyze_product_symptom_match(patient_data, product)
                
                recommendation = ProductRecommendation(
                    product_id=str(product.get("_id", "")),
                    product_name=product.get("product_name", "Unknown Product"),
                    recommendation_score=match_analysis.similarity_score,
                    symptom_match=match_analysis,
                    additional_factors={
                        "product_category": product.get("product_category", "Unknown"),
                        "cost_price": product.get("cost_price", 0),
                        "selling_price": product.get("selling_price", 0),
                        "branch_name": product.get("branch_name", "Unknown"),
                        "cpt_code": product.get("cpt_code", ""),
                        "compositions_count": len(product.get("compositions", []))
                    }
                )
                
                analyzed_products.append(recommendation)
            
            # Step 4: Sort by recommendation score and get top recommendations
            analyzed_products.sort(key=lambda x: x.recommendation_score, reverse=True)
            top_recommendations = analyzed_products[:max_recommendations]
            
            # Step 5: Generate final consultation report
            print("Generating consultation report...")
            patient_info = patient_data.get("patient", {})
            
            consultation_report = await self.recommendation_chain.ainvoke({
                "patient_name": patient_info.get("name", "Unknown"),
                "patient_age": patient_info.get("age", "Unknown"),
                "patient_gender": patient_info.get("gender", "Unknown"),
                "patient_symptoms": ", ".join(patient_info.get("symptoms", [])),
                "patient_additional_info": json.dumps(patient_info.get("additional_info", {}), indent=2),
                "analyzed_products": json.dumps([{
                    "product_name": rec.product_name,
                    "score": rec.recommendation_score,
                    "reasoning": rec.symptom_match.reasoning,
                    "confidence": rec.symptom_match.confidence,
                    "matched_symptoms": rec.symptom_match.matched_symptoms,
                    "category": rec.additional_factors.get("product_category"),
                    "price": rec.additional_factors.get("selling_price")
                } for rec in top_recommendations], indent=2)
            })
            
            # Step 6: Return comprehensive results
            return {
                "patient_info": {
                    "patient_id": str(patient_id),
                    "name": patient_info.get("name", "Unknown"),
                    "age": patient_info.get("age"),
                    "gender": patient_info.get("gender"),
                    "symptoms": patient_info.get("symptoms", []),
                    "additional_info": patient_info.get("additional_info", {})
                },
                "recommendations": [rec.dict() for rec in top_recommendations],
                "total_products_analyzed": len(analyzed_products),
                "consultation_report": consultation_report,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to generate recommendations: {str(e)}"}

    async def create_product_search_indexes(self) -> Dict[str, Any]:
        """
        Create MongoDB indexes for efficient product searching
        
        Returns:
            Dictionary with index creation results
        """
        if not self.tool_registry:
            return {"error": "Tool registry not initialized"}
        
        if not self.mongodb_client or self.mongodb_client.db is None:
            return {"error": "MongoDB client or database not initialized"}
        
        try:
            results = []
            
            # Index definitions for optimal product searching
            indexes_to_create = [
                {
                    "collection": "products",
                    "index": {"symptoms.symptom_name": 1},
                    "name": "symptom_name_index"
                },
                {
                    "collection": "products", 
                    "index": {"product_name": "text", "product_description": "text"},
                    "name": "text_search_index"
                },
                {
                    "collection": "products",
                    "index": {"product_category": 1},
                    "name": "category_index"
                },
                {
                    "collection": "products",
                    "index": {"compositions.ingredient_name": 1},
                    "name": "ingredient_index"
                }
            ]
            
            print(f"Creating {len(indexes_to_create)} indexes...")
            
            for index_def in indexes_to_create:
                try:
                    print(f"Creating index: {index_def['name']}")
                    
                    # Create index directly using MongoDB client
                    collection = self.mongodb_client.db[index_def["collection"]]
                    index_spec = index_def["index"]
                    
                    # Handle text indexes differently
                    if any(v == "text" for v in index_spec.values()):
                        # For text indexes, pass the spec directly
                        index_name = collection.create_index([(k, v) for k, v in index_spec.items()])
                    else:
                        # For regular indexes
                        index_name = collection.create_index([(k, v) for k, v in index_spec.items()])
                    
                    results.append({
                        "index_name": index_def["name"],
                        "success": True,
                        "message": f"Index created successfully: {index_name}",
                        "mongodb_index_name": str(index_name)
                    })
                    
                    print(f"✓ Created index: {index_def['name']} -> {index_name}")
                    
                except Exception as e:
                    error_msg = str(e)
                    results.append({
                        "index_name": index_def["name"],
                        "success": False,
                        "error": error_msg
                    })
                    print(f"❌ Failed to create index {index_def['name']}: {error_msg}")
            
            successful_count = sum(1 for r in results if r["success"])
            print(f"Index creation complete: {successful_count}/{len(indexes_to_create)} successful")
            
            return {
                "index_creation_results": results,
                "total_indexes": len(indexes_to_create),
                "successful_indexes": successful_count
            }
            
        except Exception as e:
            return {"error": f"Failed to create indexes: {str(e)}"}

    def get_patient_summary(self, patient_data: Dict[str, Any]) -> str:
        """
        Generate a summary of patient information for display
        
        Args:
            patient_data: Patient data dictionary
            
        Returns:
            Formatted patient summary string
        """
        try:
            patient_info = patient_data.get("patient", {})
            
            summary = f"""
PATIENT SUMMARY
================
Name: {patient_info.get('name', 'Unknown')}
Age: {patient_info.get('age', 'Unknown')}
Gender: {patient_info.get('gender', 'Unknown')}

SYMPTOMS:
{chr(10).join(f"- {symptom}" for symptom in patient_info.get('symptoms', [])) or "- No symptoms recorded"}

ADDITIONAL INFORMATION:
{chr(10).join(f"- {key}: {value}" for key, value in patient_info.get('additional_info', {}).items()) or "- No additional information"}

MEDICAL HISTORY:
{chr(10).join(f"- {item}" for item in patient_info.get('medical_history', [])) or "- No medical history recorded"}

MEDICATIONS:
{chr(10).join(f"- {item}" for item in patient_info.get('medications', [])) or "- No medications recorded"}
            """
            
            return summary.strip()
            
        except Exception as e:
            return f"Error generating patient summary: {str(e)}"

    async def get_product_details(self, product_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific product
        
        Args:
            product_id: MongoDB ObjectID of the product
            
        Returns:
            Dictionary containing detailed product information
        """
        if not self.tool_registry:
            return {"error": "Tool registry not initialized"}
        
        try:
            find_tool = self.tool_registry.get_tool("find")
            
            # Convert string ID to ObjectId for querying
            try:
                obj_id = ObjectId(product_id)
            except InvalidId:
                return {"error": f"Invalid product ID format: {product_id}"}
            
            result = await find_tool.execute({
                "collection": "products",
                "filter": {"_id": obj_id},
                "limit": 1
            })
            
            if result.is_error:
                return {"error": result.content[0].get("text", "Unknown error")}
            
            data = json.loads(result.content[0]["text"])
            products = data.get("results", [])
            
            if not products:
                return {"error": f"Product not found with ID: {product_id}"}
            
            return {"product": products[0]}
            
        except Exception as e:
            return {"error": f"Failed to fetch product details: {str(e)}"}
