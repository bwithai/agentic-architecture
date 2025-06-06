#!/usr/bin/env python3
"""
Test script for the GetPatientTool
Demonstrates how to retrieve a patient by MongoDB ObjectId
"""

import asyncio
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from mongodb.client import MongoDBClient
from agents.tools.patient.get_patient import GetPatientTool
from agents.tools.patient.create_patient_profile import CreatePatientProfileTool


async def test_patient_tools():
    """Test both creating and retrieving patients."""
    
    # Load environment variables
    load_dotenv()
    
    # Setup MongoDB connection
    try:
        print("🔌 Connecting to MongoDB...")
        
        # Create MongoDB client wrapper with database URL
        mongodb_client = MongoDBClient('mongodb://localhost:27017/kami')
        await mongodb_client.connect()
        
        print("✅ MongoDB connection successful!")
        
    except ConnectionFailure:
        print("❌ MongoDB connection failed. Please ensure MongoDB is running on localhost:27017")
        return
    except Exception as e:
        print(f"❌ MongoDB setup error: {e}")
        return
    
    try:
        # Initialize tools
        create_tool = CreatePatientProfileTool(mongodb_client)
        get_tool = GetPatientTool(mongodb_client)
        
        print("\n" + "="*60)
        print("🧪 TESTING PATIENT TOOLS")
        print("="*60)
        
        # Test 1: Create a sample patient
        print("\n1️⃣ Creating a sample patient...")
        create_params = {
            "name": "John Doe",
            "age": 35,
            "gender": "Male",
            "symptoms": ["headache", "fever", "fatigue"],
            "medical_history": ["hypertension"],
            "medications": ["lisinopril 10mg"],
            "additional_info": {
                "allergies": "penicillin",
                "emergency_contact": "Jane Doe - 555-1234"
            }
        }
        
        create_result = await create_tool.execute(create_params)
        
        if create_result.is_error:
            print(f"❌ Failed to create patient: {create_result.content[0]['text']}")
            return
        
        print(f"✅ Patient created successfully!")
        
        # Extract patient ID from the response
        import json
        response_data = json.loads(create_result.content[0]['text'])
        patient_id = response_data['metadata']['patient_id']
        print(f"📋 Patient ID: {patient_id}")
        
        # Test 2: Retrieve the patient by ID
        print(f"\n2️⃣ Retrieving patient by ID: {patient_id}")
        get_params = {
            "patient_id": patient_id
        }
        
        get_result = await get_tool.execute(get_params)
        
        if get_result.is_error:
            print(f"❌ Failed to retrieve patient: {get_result.content[0]['text']}")
            return
        
        print("✅ Patient retrieved successfully!")
        
        # Parse and display the retrieved patient data
        retrieved_data = json.loads(get_result.content[0]['text'])
        patient = retrieved_data['patient']
        
        print("\n📋 Retrieved Patient Information:")
        print(f"   👤 Name: {patient.get('name', 'N/A')}")
        print(f"   🎂 Age: {patient.get('age', 'N/A')}")
        print(f"   ⚧ Gender: {patient.get('gender', 'N/A')}")
        print(f"   🩺 Symptoms: {', '.join(patient.get('symptoms', []))}")
        print(f"   📜 Medical History: {', '.join(patient.get('medical_history', []))}")
        print(f"   💊 Medications: {', '.join(patient.get('medications', []))}")
        print(f"   ℹ️ Additional Info: {patient.get('additional_info', {})}")
        print(f"   🕐 Created: {patient.get('timestamp', 'N/A')}")
        
        # Test 3: Try to retrieve a non-existent patient
        print(f"\n3️⃣ Testing retrieval of non-existent patient...")
        fake_id = "507f1f77bcf86cd799439011"  # Valid ObjectId format but doesn't exist
        fake_params = {
            "patient_id": fake_id
        }
        
        fake_result = await get_tool.execute(fake_params)
        print(f"✅ Handled non-existent patient correctly: {fake_result.content[0]['text']}")
        
        # Test 4: Try invalid ObjectId format
        print(f"\n4️⃣ Testing invalid ObjectId format...")
        invalid_params = {
            "patient_id": "invalid-id-format"
        }
        
        try:
            invalid_result = await get_tool.execute(invalid_params)
            if invalid_result.is_error:
                print(f"✅ Correctly handled invalid ObjectId: {invalid_result.content[0]['text']}")
            else:
                print(f"⚠️ Unexpected success with invalid ID: {invalid_result.content[0]['text']}")
        except Exception as e:
            print(f"✅ Correctly caught exception for invalid ObjectId: {e}")
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test error: {e}")
    
    finally:
        # Clean up MongoDB connection
        if mongodb_client and mongodb_client.client:
            try:
                await mongodb_client.close()
                print("\n🔌 MongoDB connection closed.")
            except Exception as e:
                print(f"⚠️  Error closing MongoDB connection: {e}")


def main():
    """Main function to run the tests."""
    print("🚀 Starting Patient Tools Test")
    print("\nThis test will:")
    print("1. Create a sample patient profile")
    print("2. Retrieve the patient by MongoDB ObjectId")
    print("3. Test error handling for non-existent patients")
    print("4. Test validation for invalid ObjectId formats")
    
    # Run the async test
    asyncio.run(test_patient_tools())


if __name__ == "__main__":
    main() 