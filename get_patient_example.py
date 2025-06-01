#!/usr/bin/env python3
"""
Simple example of using GetPatientTool to retrieve a patient by MongoDB ObjectId
Usage: python get_patient_example.py [patient_id]
"""

import asyncio
import sys
from dotenv import load_dotenv

from mongodb.client import MongoDBClient
from agents.tools.patient.get_patient import GetPatientTool


async def get_patient_by_id(patient_id: str):
    """
    Retrieve a patient by their MongoDB ObjectId.
    
    Args:
        patient_id: MongoDB ObjectId as string (e.g., '6838a9a09e7ca8ddfcc6c1de')
    """
    
    # Load environment variables
    load_dotenv()
    
    # Setup MongoDB connection
    try:
        print(f"🔌 Connecting to MongoDB...")
        mongodb_client = MongoDBClient('mongodb://localhost:27017/kami')
        await mongodb_client.connect()
        print("✅ Connected to MongoDB successfully!")
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return None
    
    try:
        # Initialize the get patient tool
        get_tool = GetPatientTool(mongodb_client)
        
        print(f"🔍 Searching for patient with ID: {patient_id}")
        
        # Execute the tool
        result = await get_tool.execute({"patient_id": patient_id})
        
        if result.is_error:
            print(f"❌ Error: {result.content[0]['text']}")
            return None
        
        # Parse and display the result
        import json
        response_data = json.loads(result.content[0]['text'])
        
        if not response_data['metadata']['found']:
            print(f"⚠️ Patient not found with ID: {patient_id}")
            return None
        
        patient = response_data['patient']
        
        # Display patient information in a nice format
        print("\n" + "="*60)
        print("📋 PATIENT INFORMATION")
        print("="*60)
        print(f"🆔 Patient ID: {patient_id}")
        print(f"👤 Name: {patient.get('name', 'N/A')}")
        print(f"🎂 Age: {patient.get('age', 'N/A')}")
        print(f"⚧ Gender: {patient.get('gender', 'N/A')}")
        
        symptoms = patient.get('symptoms', [])
        if symptoms:
            print(f"🩺 Symptoms: {', '.join(symptoms)}")
        else:
            print("🩺 Symptoms: None reported")
        
        medical_history = patient.get('medical_history', [])
        if medical_history:
            print(f"📜 Medical History: {', '.join(medical_history)}")
        else:
            print("📜 Medical History: None reported")
        
        medications = patient.get('medications', [])
        if medications:
            print(f"💊 Medications: {', '.join(medications)}")
        else:
            print("💊 Medications: None reported")
        
        additional_info = patient.get('additional_info', {})
        if additional_info:
            print("ℹ️ Additional Information:")
            for key, value in additional_info.items():
                print(f"   • {key}: {value}")
        
        print(f"🕐 Created: {patient.get('timestamp', 'N/A')}")
        print(f"💬 QA Pairs: {patient.get('qa_pairs_count', 0)}")
        print(f"✅ Extraction Performed: {patient.get('extraction_performed', False)}")
        
        return patient
        
    except Exception as e:
        print(f"❌ Error retrieving patient: {e}")
        return None
    
    finally:
        # Clean up MongoDB connection
        if mongodb_client and mongodb_client.client:
            try:
                await mongodb_client.close()
                print("\n🔌 MongoDB connection closed.")
            except Exception as e:
                print(f"⚠️ Error closing MongoDB connection: {e}")


def main():
    """Main function to handle command line arguments and run the patient lookup."""
    
    # Check if patient ID was provided as command line argument
    if len(sys.argv) > 1:
        patient_id = sys.argv[1]
    else:
        # Use the example patient ID from the test, or ask for input
        patient_id = input("Enter patient ID (MongoDB ObjectId): ").strip()
    
    if not patient_id:
        print("❌ No patient ID provided")
        print("Usage: python get_patient_example.py [patient_id]")
        print("Example: python get_patient_example.py 6838a9a09e7ca8ddfcc6c1de")
        return
    
    print(f"🚀 Looking up patient: {patient_id}")
    
    # Run the async function
    patient = asyncio.run(get_patient_by_id(patient_id))
    
    if patient:
        print("\n✅ Patient lookup completed successfully!")
    else:
        print("\n❌ Patient lookup failed!")


if __name__ == "__main__":
    main() 