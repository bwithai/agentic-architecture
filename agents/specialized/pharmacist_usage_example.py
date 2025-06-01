#!/usr/bin/env python3
"""
Pharmacist Agent Usage Example

This script demonstrates how to use the PharmacistAgent to:
1. Fetch patient data by ObjectID
2. Generate intelligent product recommendations based on symptoms
3. Create necessary database indexes for optimal performance
4. Analyze product-symptom matches using LLM capabilities

Example patient ID from the provided example: "683c01c39836522c266fafb0"
"""

import asyncio
import os
import json
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient

# Add the project root to Python path for proper imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import our agents and utilities
from agents.specialized.pharmacist_agent import PharmacistAgent
from mongodb.client import MongoDBClient

# Load environment variables
load_dotenv()


async def setup_sample_products(mongodb_client: MongoDBClient):
    """
    Setup sample products in the database for testing
    (This would typically be done separately, but included here for demonstration)
    """
    print("Setting up sample products...")
    
    # Sample products based on the document structure provided
    sample_products = [
        {
            "product_name": "Bali Belly Infusion - Rapid Relief",
            "product_description": "Rapid relief IV infusion for traveler's diarrhea and gastroenteritis",
            "product_category": "IV",
            "product_alias": "Traveler's Diarrhea – Gastroenteritis IV",
            "cost_price": 437500,
            "selling_price": 2190000,
            "cpt_code": "96365",
            "is_doctor": False,
            "branch_name": "Bali",
            "entity_name": "CSI",
            "compositions": [
                {"quantity": 1, "ingredient_name": "Diagit / Molagit Tab", "ingredient_unit": "Str"},
                {"quantity": 1, "ingredient_name": "Hyoscine", "ingredient_unit": "Vial"},
                {"quantity": 1, "ingredient_name": "Ketorolac (Optional)", "ingredient_unit": "Amp"},
                {"quantity": 1, "ingredient_name": "NaCl 500ml / Ringer Lactat 500ml", "ingredient_unit": "Kolf"},
                {"quantity": 1, "ingredient_name": "Omeprazole / Ranitidin / Pantoprazole", "ingredient_unit": "Amp"},
                {"quantity": 1, "ingredient_name": "Ondansetron 4mg/2ml", "ingredient_unit": "Amp"},
                {"quantity": 3, "ingredient_name": "Vitamin B Complex", "ingredient_unit": "cc"},
                {"quantity": 1, "ingredient_name": "Vometa / Ondansetron Tab (Optional K/P)", "ingredient_unit": "Str"}
            ],
            "symptoms": [
                {
                    "icd10_code": "R25.2",
                    "symptom_name": "Abdominal Cramps",
                    "symptom_description": "Aching or cramping pain in the abdominal area, potentially caused by digestive disorders or infection."
                },
                {
                    "icd10_code": "A09",
                    "symptom_name": "Diarrhea",
                    "symptom_description": "Frequent passage of loose or watery stools."
                },
                {
                    "icd10_code": "R14",
                    "symptom_name": "Flatulence",
                    "symptom_description": "Accumulation of gas in the gastrointestinal tract causing a feeling of fullness."
                },
                {
                    "icd10_code": "R11",
                    "symptom_name": "Nausea and vomiting",
                    "symptom_description": "Unpleasant sensation in the stomach that can result in the urge to vomit."
                }
            ]
        },
        {
            "product_name": "Sleep Recovery IV Therapy",
            "product_description": "IV therapy designed to help with sleep disorders and insomnia recovery",
            "product_category": "IV",
            "product_alias": "Insomnia Relief IV",
            "cost_price": 350000,
            "selling_price": 1750000,
            "cpt_code": "96365",
            "is_doctor": False,
            "branch_name": "Wellness Center",
            "entity_name": "CSI",
            "compositions": [
                {"quantity": 500, "ingredient_name": "Normal Saline", "ingredient_unit": "ml"},
                {"quantity": 2, "ingredient_name": "Magnesium Sulfate", "ingredient_unit": "amp"},
                {"quantity": 1, "ingredient_name": "Vitamin B Complex", "ingredient_unit": "vial"},
                {"quantity": 1, "ingredient_name": "Melatonin", "ingredient_unit": "tablet"}
            ],
            "symptoms": [
                {
                    "icd10_code": "G47.0",
                    "symptom_name": "Insomnia",
                    "symptom_description": "Difficulty falling asleep or staying asleep."
                },
                {
                    "icd10_code": "G47.9",
                    "symptom_name": "Sleep disorder",
                    "symptom_description": "General sleep pattern disruption affecting quality of rest."
                },
                {
                    "icd10_code": "R06.00",
                    "symptom_name": "Fatigue",
                    "symptom_description": "Extreme tiredness and lack of energy."
                }
            ]
        },
        {
            "product_name": "Digestive Health Support",
            "product_description": "Comprehensive digestive health support for headaches related to digestion",
            "product_category": "Oral",
            "product_alias": "Digestive Headache Relief",
            "cost_price": 150000,
            "selling_price": 750000,
            "cpt_code": "99213",
            "is_doctor": True,
            "branch_name": "Gastro Clinic",
            "entity_name": "CSI",
            "compositions": [
                {"quantity": 1, "ingredient_name": "Proton Pump Inhibitor", "ingredient_unit": "tablet"},
                {"quantity": 1, "ingredient_name": "Digestive Enzymes", "ingredient_unit": "capsule"},
                {"quantity": 1, "ingredient_name": "Probiotics", "ingredient_unit": "capsule"}
            ],
            "symptoms": [
                {
                    "icd10_code": "R51",
                    "symptom_name": "Headache",
                    "symptom_description": "Pain in the head or upper neck area."
                },
                {
                    "icd10_code": "K30",
                    "symptom_name": "Indigestion",
                    "symptom_description": "Discomfort in the stomach after eating."
                },
                {
                    "icd10_code": "K59.1",
                    "symptom_name": "Slow digestion",
                    "symptom_description": "Delayed gastric emptying causing digestive discomfort."
                }
            ]
        },
        {
            "product_name": "Mood Stabilizer Complex",
            "product_description": "Natural mood support for depression and low mood symptoms",
            "product_category": "Oral",
            "product_alias": "Depression Support",
            "cost_price": 200000,
            "selling_price": 1000000,
            "cpt_code": "99214",
            "is_doctor": True,
            "branch_name": "Mental Health Clinic",
            "entity_name": "CSI",
            "compositions": [
                {"quantity": 1, "ingredient_name": "St. John's Wort", "ingredient_unit": "capsule"},
                {"quantity": 1, "ingredient_name": "Omega-3 Fatty Acids", "ingredient_unit": "softgel"},
                {"quantity": 1, "ingredient_name": "Vitamin D3", "ingredient_unit": "tablet"},
                {"quantity": 1, "ingredient_name": "B-Complex Vitamins", "ingredient_unit": "tablet"}
            ],
            "symptoms": [
                {
                    "icd10_code": "F32.9",
                    "symptom_name": "Depression",
                    "symptom_description": "Persistent feelings of sadness and loss of interest."
                },
                {
                    "icd10_code": "R45.2",
                    "symptom_name": "Low mood",
                    "symptom_description": "Feeling down or experiencing negative emotions."
                },
                {
                    "icd10_code": "G47.0",
                    "symptom_name": "Sleep disturbance",
                    "symptom_description": "Disrupted sleep patterns affecting mental health."
                }
            ]
        }
    ]
    
    try:
        # Insert sample products
        db = mongodb_client.db
        if db is None:
            print("Database connection not available")
            return False
            
        # Clear existing products (optional - for testing)
        # db.products.delete_many({})
        
        # Insert new products
        result = db.products.insert_many(sample_products)
        print(f"Inserted {len(result.inserted_ids)} sample products")
        return True
        
    except Exception as e:
        print(f"Error setting up sample products: {str(e)}")
        return False


async def demonstrate_pharmacist_agent():
    """
    Main demonstration function
    """
    print("=== PHARMACIST AGENT DEMONSTRATION ===")
    print(f"Timestamp: {datetime.now()}\n")
    
    # Initialize MongoDB client
    mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    database_name = os.getenv("MONGODB_DATABASE", "default_database")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY not found in environment variables!")
        print("Please set your OpenAI API key in .env file or environment.")
        return
    
    print(f"Connecting to MongoDB: {mongodb_uri}")
    print(f"Database: {database_name}")
    
    try:
        # Setup MongoDB client
        mongodb_client = MongoDBClient(mongodb_uri, database_name)
        
        # Initialize the pharmacist agent
        print("\nInitializing Pharmacist Agent...")
        pharmacist = PharmacistAgent(
            openai_api_key=openai_api_key,
            mongodb_client=mongodb_client
        )
        
        # Step 1: Create database indexes for optimal performance
        print("\n=== STEP 1: Creating Database Indexes ===")
        index_results = await pharmacist.create_product_search_indexes()
        print("Index Creation Results:")
        print(json.dumps(index_results, indent=2))
        
        # Step 2: Example patient ID (from your provided example)
        patient_id = "683c01c39836522c266fafb0"  # Sanaullah's patient ID
        print(f"\n=== STEP 2: Fetching Patient Data ===")
        print(f"Patient ID: {patient_id}")
        
        patient_data = await pharmacist.get_patient_by_id(patient_id)
        
        if "error" in patient_data:
            print(f"Error fetching patient: {patient_data['error']}")
        
        # Display patient summary
        if "error" not in patient_data:
            print("\n=== PATIENT INFORMATION ===")
            patient_summary = pharmacist.get_patient_summary(patient_data)
            print(patient_summary)
        
        # Step 3: Generate product recommendations
        print(f"\n=== STEP 3: Generating Product Recommendations ===")
        print("This may take a few moments as we analyze products using AI...")
        
        recommendations = await pharmacist.generate_product_recommendations(
            patient_data=patient_data,
            max_recommendations=5
        )
        
        if "error" in recommendations:
            print(f"Error generating recommendations: {recommendations['error']}")
            return
        
        # Step 4: Display results
        print("\n=== RECOMMENDATION RESULTS ===")
        print(f"Total products analyzed: {recommendations['total_products_analyzed']}")
        print(f"Analysis timestamp: {recommendations['analysis_timestamp']}")
        
        print("\n=== TOP RECOMMENDATIONS ===")
        for i, rec in enumerate(recommendations['recommendations'], 1):
            print(f"\n{i}. {rec['product_name']}")
            print(f"   Recommendation Score: {rec['recommendation_score']:.2f}/1.00")
            print(f"   Confidence: {rec['symptom_match']['confidence']}")
            print(f"   Category: {rec['additional_factors'].get('product_category', 'Unknown')}")
            print(f"   Price: ${rec['additional_factors'].get('selling_price', 0):,}")
            print(f"   Reasoning: {rec['symptom_match']['reasoning']}")
            
            if rec['symptom_match']['matched_symptoms']:
                print(f"   Matched Symptoms: {', '.join(rec['symptom_match']['matched_symptoms'])}")
        
        print("\n=== PHARMACIST CONSULTATION REPORT ===")
        print(recommendations['consultation_report'])
        
        # Step 5: Demonstrate getting detailed product information
        if recommendations['recommendations']:
            top_product_id = recommendations['recommendations'][0]['product_id']
            print(f"\n=== STEP 4: Detailed Product Information ===")
            print(f"Getting details for top recommended product: {top_product_id}")
            
            product_details = await pharmacist.get_product_details(top_product_id)
            if "error" not in product_details:
                product = product_details['product']
                print(f"\nProduct Details:")
                print(f"Name: {product.get('product_name', 'Unknown')}")
                print(f"Description: {product.get('product_description', 'No description')}")
                print(f"Category: {product.get('product_category', 'Unknown')}")
                print(f"Cost Price: ${product.get('cost_price', 0):,}")
                print(f"Selling Price: ${product.get('selling_price', 0):,}")
                print(f"CPT Code: {product.get('cpt_code', 'N/A')}")
                print(f"Branch: {product.get('branch_name', 'Unknown')}")
                
                print("\nCompositions:")
                for comp in product.get('compositions', []):
                    print(f"  - {comp.get('ingredient_name', 'Unknown')}: "
                          f"{comp.get('quantity', 'N/A')} {comp.get('ingredient_unit', 'units')}")
                
                print("\nTarget Symptoms:")
                for symptom in product.get('symptoms', []):
                    print(f"  - {symptom.get('symptom_name', 'Unknown')}: "
                          f"{symptom.get('symptom_description', 'No description')}")
            else:
                print(f"Error getting product details: {product_details['error']}")
        
        # Step 6: Performance and recommendations summary
        print(f"\n=== PERFORMANCE SUMMARY ===")
        print(f"✓ Patient fetched successfully")
        print(f"✓ {recommendations['total_products_analyzed']} products analyzed")
        print(f"✓ {len(recommendations['recommendations'])} recommendations generated")
        print(f"✓ AI-powered symptom matching completed")
        print(f"✓ Consultation report generated")
        
        print(f"\n=== INDEXING RECOMMENDATIONS ===")
        if index_results.get('successful_indexes', 0) > 0:
            print(f"✓ {index_results['successful_indexes']}/{index_results['total_indexes']} indexes created successfully")
            print("  Database is optimized for fast product searches")
        else:
            print("⚠ Database indexes not created - search performance may be slower")
            
        print(f"\n=== NEXT STEPS ===")
        print("1. Review the consultation report with medical professionals")
        print("2. Consider patient's medical history and contraindications")
        print("3. Verify product availability and pricing")
        print("4. Document the recommendation in patient records")
        
    except Exception as e:
        print(f"Error in demonstration: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        print(f"\n=== DEMONSTRATION COMPLETED ===")
        print(f"End time: {datetime.now()}")


async def quick_test_single_patient(patient_id: str = None):
    """
    Quick test function for a single patient recommendation
    """
    print("=== QUICK PHARMACIST TEST ===")
    
    # Use default patient ID if none provided
    if not patient_id:
        patient_id = "683c01c39836522c266fafb0"
    
    mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    database_name = os.getenv("MONGODB_DATABASE", "default_database")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY required!")
        return
    
    try:
        mongodb_client = MongoDBClient(mongodb_uri, database_name)
        pharmacist = PharmacistAgent(openai_api_key=openai_api_key, mongodb_client=mongodb_client)
        
        print(f"Testing recommendations for patient: {patient_id}")
        recommendations = await pharmacist.generate_product_recommendations(patient_id, max_recommendations=3)
        
        if "error" in recommendations:
            print(f"Error: {recommendations['error']}")
        else:
            print(f"Success! Found {len(recommendations['recommendations'])} recommendations")
            for i, rec in enumerate(recommendations['recommendations'], 1):
                print(f"{i}. {rec['product_name']} (Score: {rec['recommendation_score']:.2f})")
                
    except Exception as e:
        print(f"Test failed: {str(e)}")


if __name__ == "__main__":
    print("Pharmacist Agent Usage Example")
    print("==============================")
    
    # Run the full demonstration
    print("\nChoose an option:")
    print("1. Full demonstration (recommended)")
    print("2. Quick test")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        custom_patient_id = input("Enter patient ID (or press Enter for default): ").strip()
        asyncio.run(quick_test_single_patient(custom_patient_id if custom_patient_id else None))
    else:
        asyncio.run(demonstrate_pharmacist_agent()) 