#!/usr/bin/env python3
"""
Simple test to verify all imports work correctly.
"""

def test_imports():
    """Test all the key imports for the patient profile system."""
    try:
        print("🔧 Testing imports...")
        
        # Test base imports
        from agents.tools.patient.create_patient_profile import CreatePatientProfileTool
        print("✅ CreatePatientProfileTool import successful")
        
        from agents.tools.registry import ToolRegistry
        print("✅ ToolRegistry import successful")
        
        from agents.specialized.mongodb_agent import MongoDBChatBot
        print("✅ MongoDBChatBot import successful")
        
        from agents.base.base_agent import _create_patient_profile_chain
        print("✅ Patient profile chain import successful")
        
        print("\n🎉 All imports successful! The patient profile system is ready.")
        print("\nNext steps:")
        print("1. Set your OPENAI_API_KEY environment variable")
        print("2. Set your MONGODB_URI environment variable")
        print("3. Run: python main.py")
        print("4. Or test with: python test_patient_profile.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_imports() 