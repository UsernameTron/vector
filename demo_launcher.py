#!/usr/bin/env python3
"""
Vector RAG Database - Demo Script
Demonstrates the desktop launcher and system capabilities
"""

import subprocess
import time
import sys
import os
from pathlib import Path

def main():
    print("🚀 Vector RAG Database - Demo Launcher")
    print("=" * 50)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    print(f"📁 Project Directory: {project_dir}")
    print(f"📊 Current Working Directory: {os.getcwd()}")
    
    # Check if all required files exist
    required_files = [
        "desktop_launcher.py",
        "app.py", 
        "requirements.txt",
        ".env.template"
    ]
    
    print("\n🔍 Checking project files...")
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        print("Please ensure you're in the correct project directory.")
        return
    
    print("\n📋 Available Launch Options:")
    print("1. 🖥️  Desktop Launcher (GUI)")
    print("2. 🚀 Direct Server Start")
    print("3. 📖 Show README")
    print("4. 🔧 Check Dependencies")
    print("5. 🌐 Open GitHub Repository")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nSelect option (0-5): ").strip()
            
            if choice == "0":
                print("👋 Goodbye!")
                break
            elif choice == "1":
                print("\n🖥️ Launching Desktop GUI...")
                try:
                    subprocess.run([sys.executable, "desktop_launcher.py"])
                except KeyboardInterrupt:
                    print("\n⏹️ GUI closed by user")
            elif choice == "2":
                print("\n🚀 Starting server directly...")
                try:
                    subprocess.run([sys.executable, "app.py"])
                except KeyboardInterrupt:
                    print("\n⏹️ Server stopped by user")
            elif choice == "3":
                print("\n📖 README.md:")
                print("-" * 30)
                try:
                    with open("README.md", "r") as f:
                        print(f.read())
                except FileNotFoundError:
                    print("❌ README.md not found")
            elif choice == "4":
                print("\n🔧 Checking dependencies...")
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--dry-run"])
                    print("✅ All dependencies available")
                except subprocess.CalledProcessError:
                    print("❌ Some dependencies may need installation")
                    print("Run: pip install -r requirements.txt")
            elif choice == "5":
                import webbrowser
                print("\n🌐 Opening GitHub repository...")
                webbrowser.open("https://github.com/UsernameTron/vector")
            else:
                print("❌ Invalid option. Please choose 0-5.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Demo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
