#!/usr/bin/env python3
"""
Enhanced Medical Consultation System - Streamlit Launcher

This script launches the Streamlit web interface for the enhanced medical consultation system.
The system provides a complete patient journey from consultation to product recommendations.

Usage:
    python run_streamlit.py

Requirements:
    - MongoDB running on localhost:27017
    - OPENAI_API_KEY environment variable set
    - All dependencies installed (pip install -r requirements.txt)
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all requirements are met before starting"""
    issues = []
    
    # Check if MongoDB is accessible (optional check)
    try:
        from pymongo import MongoClient
        client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=2000)
        client.admin.command('ping')
        print("✅ MongoDB connection: OK")
    except Exception as e:
        issues.append(f"⚠️  MongoDB connection issue: {e}")
    
    # Check OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        issues.append("❌ OPENAI_API_KEY environment variable not set")
    else:
        print("✅ OpenAI API key: Configured")
    
    # Check if Streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit: Installed")
    except ImportError:
        issues.append("❌ Streamlit not installed (pip install streamlit)")
    
    # Check if app.py exists
    if not Path("app.py").exists():
        issues.append("❌ app.py file not found")
    else:
        print("✅ app.py: Found")
    
    if issues:
        print("\n🚨 Issues found:")
        for issue in issues:
            print(f"   {issue}")
        print("\n💡 Please resolve these issues before starting the application.")
        return False
    
    return True

def main():
    """Main function to launch the Streamlit app"""
    print("🏥 Enhanced Medical Consultation System - Streamlit Launcher")
    print("=" * 70)
    
    print("\n🔍 Checking requirements...")
    if not check_requirements():
        sys.exit(1)
    
    print("\n🚀 Starting Streamlit application...")
    print("📱 The web interface will open in your default browser")
    print("🔗 URL: http://localhost:8501")
    print("\n⏹️  Press Ctrl+C to stop the application")
    print("=" * 70)
    
    try:
        # Launch Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false",
            "--theme.base", "light"
        ]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 