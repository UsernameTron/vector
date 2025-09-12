#!/usr/bin/env python3
"""
🚀 Vector RAG Database - Desktop Launcher
Enhanced desktop application launcher with automatic browser opening
"""

import os
import sys
import subprocess
import time
import webbrowser
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import requests
from typing import Optional

class VectorRAGLauncher:
    def __init__(self):
        self.root = tk.Tk()
        self.server_process: Optional[subprocess.Popen] = None
        self.server_port = 5001
        self.server_url = f"http://localhost:{self.server_port}"
        self.app_mode = "production"  # Default to production mode
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the launcher UI"""
        self.root.title("🚀 Vector RAG Database Launcher")
        self.root.geometry("650x550")
        self.root.resizable(False, False)
        
        # Configure style for cyberpunk theme
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure custom colors (light blue cyberpunk theme)
        style.configure('Title.TLabel', foreground='#00BFFF', font=("Arial", 20, "bold"))
        style.configure('Subtitle.TLabel', foreground='#87CEEB', font=("Arial", 12, "italic"))
        style.configure('Cyberpunk.TButton', foreground='#00BFFF')
        
        # Set background color
        self.root.configure(bg='#0a0a0a')
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill="both", expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="🚀 Vector RAG Database", 
            style="Title.TLabel"
        )
        title_label.pack(pady=(0, 5))
        
        subtitle_label = ttk.Label(
            main_frame, 
            text="Advanced AI-Powered Document Intelligence System",
            style="Subtitle.TLabel"
        )
        subtitle_label.pack(pady=(0, 20))
        
        # Configuration frame
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        config_frame.pack(fill="x", pady=(0, 10))
        
        # Mode selection
        mode_label = ttk.Label(config_frame, text="Application Mode:")
        mode_label.pack(anchor="w")
        
        mode_frame = ttk.Frame(config_frame)
        mode_frame.pack(fill="x", pady=(5, 0))
        
        self.mode_var = tk.StringVar(value="production")
        modes = [
            ("Production", "production", "🚀 Full features with security"),
            ("Development", "development", "🛠️ Debug mode with verbose logging"), 
            ("Clean Architecture", "clean", "🏗️ Modern architecture with Swagger docs")
        ]
        
        for i, (label, value, desc) in enumerate(modes):
            rb = ttk.Radiobutton(mode_frame, text=f"{label} - {desc}", 
                               variable=self.mode_var, value=value,
                               command=self.on_mode_change)
            rb.pack(anchor="w", pady=2)
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="System Status", padding="10")
        status_frame.pack(fill="x", pady=(0, 20))
        
        self.status_label = ttk.Label(status_frame, text="Ready to start...", font=("Arial", 10))
        self.status_label.pack()
        
        # Progress bar
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate')
        self.progress.pack(fill="x", pady=(10, 0))
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill="x", pady=(0, 20))
        
        # Start button
        self.start_button = ttk.Button(
            buttons_frame, 
            text="🚀 Start Vector RAG System", 
            command=self.start_system,
            style="Cyberpunk.TButton"
        )
        self.start_button.pack(side="left", padx=(0, 10))
        
        # Open browser button (initially disabled)
        self.browser_button = ttk.Button(
            buttons_frame, 
            text="🌐 Open Web Interface", 
            command=self.open_browser,
            state="disabled",
            style="Cyberpunk.TButton"
        )
        self.browser_button.pack(side="left", padx=(0, 10))
        
        # Stop button (initially disabled)
        self.stop_button = ttk.Button(
            buttons_frame, 
            text="⏹️ Stop System", 
            command=self.stop_system,
            state="disabled",
            style="Cyberpunk.TButton"
        )
        self.stop_button.pack(side="left")
        
        # Information frame
        info_frame = ttk.LabelFrame(main_frame, text="Available Features", padding="10")
        info_frame.pack(fill="both", expand=True)
        
        features_text = """
🏠 Main Dashboard: Complete system overview and document management
📊 Vector Database: ChromaDB-powered document storage and retrieval
🔍 Smart Search: Semantic search with context-aware results
💬 Agent Chat: Interactive conversations with specialized AI agents

🎯 8 Specialized AI Agents Available:
• 📈 Research Agent - Market intelligence & data analysis
• 👔 CEO Agent - Strategic business decisions & leadership
• ⚡ Performance Agent - Metrics analysis & optimization
• 🎓 Coaching Agent - Training & development guidance  
• 💼 Business Intelligence Agent - Data insights & reporting
• 📞 Contact Center Director - Customer service management
• 🔍 Code Analyzer Agent - Code review and analysis
• 🎯 Triage Agent - Issue prioritization and routing

🚀 Advanced RAG Features (NEW):
• Dynamic Context Management - Intelligent context window optimization
• Query Intelligence & Expansion - Smart query processing and routing
• Multi-hop Graph Reasoning - Entity extraction and relationship mapping
• Advanced Retrieval Patterns - ColBERT, SPLADE, and hybrid retrieval
• Real-time Quality Monitoring - Confidence scoring and hallucination detection
• Embedding Optimization - Quantization and domain-specific embeddings
• Advanced Caching with Redis - Multi-level intelligent caching
• Comprehensive RAG Evaluation - 20+ metrics with A/B testing

🔧 Core Capabilities:
• Vector embeddings for semantic document understanding
• RAG (Retrieval-Augmented Generation) for context-aware responses
• Cyberpunk-themed UI with smooth animations
• Document upload and intelligent processing
• Real-time chat with AI agents using document context
• Comprehensive API endpoints for system integration

🌐 Web Interface Features:
• Beautiful light blue cyberpunk design
• Responsive layout with glowing effects
• Agent selection and specialized conversations
• Document management and search
• Real-time status updates and error handling
        """
        
        info_label = ttk.Label(info_frame, text=features_text, font=("Arial", 9), justify="left")
        info_label.pack(anchor="w", fill="both", expand=True)
        
        # Footer
        footer_label = ttk.Label(
            main_frame, 
            text="💡 Tip: The system will automatically open in your browser once started", 
            font=("Arial", 8, "italic"),
            foreground='#87CEEB'
        )
        footer_label.pack(pady=(10, 0))
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def update_status(self, message: str, show_progress: bool = False):
        """Update status label and progress bar"""
        self.status_label.config(text=message)
        if show_progress:
            self.progress.start(10)
        else:
            self.progress.stop()
        self.root.update()
        
    def check_dependencies(self) -> bool:
        """Check if required packages are installed"""
        try:
            import flask
            import chromadb
            import openai
            import requests
            self.update_status("✅ Dependencies verified")
            return True
        except ImportError as e:
            self.update_status(f"❌ Missing dependency: {e}")
            messagebox.showerror(
                "Dependencies Missing", 
                f"Required packages not found: {e}\n\nPlease run: pip install -r requirements.txt"
            )
            return False
            
    def check_environment(self) -> bool:
        """Check if environment is properly configured"""
        if not os.path.exists('.env'):
            if os.path.exists('.env.template'):
                messagebox.showwarning(
                    "Environment Setup Required",
                    "Please copy .env.template to .env and configure your OpenAI API key:\n\n"
                    "cp .env.template .env\n\n"
                    "Then edit .env and add your OPENAI_API_KEY"
                )
            else:
                messagebox.showerror(
                    "Environment Missing",
                    "No .env file found. Please create one with your OPENAI_API_KEY"
                )
            return False
            
        return True
        
    def start_system(self):
        """Start the Vector RAG Database System"""
        if not self.check_dependencies():
            return
            
        if not self.check_environment():
            return
            
        self.start_button.config(state="disabled")
        self.update_status("🔧 Checking system dependencies...", True)
        
        # Start server in a separate thread
        threading.Thread(target=self._start_server_thread, daemon=True).start()
        
    def _start_server_thread(self):
        """Start the Flask server in a separate thread"""
        try:
            self.update_status("🚀 Starting Vector RAG Database server...")
            
            # Use the new unified application
            app_files = ["app_unified.py", "app_production.py", "app_demo.py", "app.py"]
            app_to_use = None
            
            for app_file in app_files:
                if os.path.exists(app_file):
                    app_to_use = app_file
                    break
                    
            if not app_to_use:
                self.update_status("❌ No Flask application file found")
                self._reset_ui()
                return
                
            try:
                # Set environment variables for Flask
                env = os.environ.copy()
                env['FLASK_ENV'] = self.app_mode
                env['FLASK_APP'] = app_to_use
                
                # Prepare command arguments based on app type
                if app_to_use == "app_unified.py":
                    # Use unified app with mode and port arguments
                    cmd = [
                        sys.executable, app_to_use, 
                        "--mode", self.app_mode,
                        "--host", "127.0.0.1",
                        "--port", str(self.server_port)
                    ]
                else:
                    # Legacy app files
                    cmd = [sys.executable, app_to_use]
                
                # Start the server
                self.server_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    env=env,
                    cwd=os.getcwd()
                )
                
                # Give server time to start
                self.update_status("⏳ Starting server... (this may take a moment)")
                time.sleep(7)  # Vector DB initialization takes time
                
                # Check if server is responding
                if self._check_server_health():
                    self.update_status("✅ Vector RAG Database started successfully")
                    # Update UI on successful start
                    self.root.after(0, self._on_server_started)
                else:
                    self.update_status("❌ Server failed to start or is not responding")
                    if self.server_process:
                        self.server_process.terminate()
                        self.server_process = None
                    self._reset_ui()
                    
            except Exception as e:
                self.update_status(f"❌ Failed to start server: {e}")
                self._reset_ui()
                
        except Exception as e:
            self.update_status(f"❌ Error starting system: {e}")
            self._reset_ui()
            
    def _check_server_health(self) -> bool:
        """Check if the server is responding"""
        health_endpoints = [
            f"{self.server_url}/api/health",
            f"{self.server_url}/",
            f"{self.server_url}/api/agents"
        ]
        
        for endpoint in health_endpoints:
            try:
                response = requests.get(endpoint, timeout=5)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                continue
                
        return False
        
    def _on_server_started(self):
        """Handle successful server start"""
        self.browser_button.config(state="normal")
        self.stop_button.config(state="normal")
        self.update_status("🎉 Vector RAG Database is ready! Opening web interface...")
        
        # Automatically open browser after a short delay
        self.root.after(2000, self.open_browser)  # Wait 2 seconds then open browser
        
    def open_browser(self):
        """Open the web interface in the default browser"""
        try:
            # Try the main endpoint
            try:
                response = requests.get(self.server_url, timeout=3)
                if response.status_code == 200:
                    webbrowser.open(self.server_url)
                    self.update_status(f"🌐 Opened Vector RAG Database interface: {self.server_url}")
                    return
            except requests.exceptions.RequestException:
                pass
                
            # Fallback - just open the base URL anyway
            webbrowser.open(self.server_url)
            self.update_status(f"🌐 Opened web interface: {self.server_url}")
                
        except Exception as e:
            self.update_status(f"❌ Failed to open browser: {e}")
            messagebox.showwarning(
                "Browser Error", 
                f"Could not open browser automatically.\n\nPlease manually navigate to: {self.server_url}"
            )
            
    def stop_system(self):
        """Stop the Vector RAG Database System"""
        try:
            self.update_status("⏹️ Stopping Vector RAG Database...", True)
            
            if self.server_process:
                self.server_process.terminate()
                try:
                    self.server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.server_process.kill()
                    self.server_process.wait()
                self.server_process = None
                
            self._reset_ui()
            self.update_status("✅ System stopped successfully")
            
        except Exception as e:
            self.update_status(f"❌ Error stopping system: {e}")
            self._reset_ui()
            
    def on_mode_change(self):
        """Handle application mode change"""
        self.app_mode = self.mode_var.get()
        self.update_status(f"Mode changed to: {self.app_mode}")
        
        # Use consistent port 5001 for unified app regardless of mode
        self.server_port = 5001
        self.server_url = f"http://localhost:{self.server_port}"
        
    def _reset_ui(self):
        """Reset UI to initial state"""
        self.start_button.config(state="normal")
        self.browser_button.config(state="disabled")
        self.stop_button.config(state="disabled")
        self.progress.stop()
        
    def on_closing(self):
        """Handle window closing event"""
        if self.server_process:
            result = messagebox.askyesno(
                "Confirm Exit", 
                "The Vector RAG Database is still running.\n\nDo you want to stop it and exit?"
            )
            if result:
                self.stop_system()
                time.sleep(1)  # Give time for cleanup
                self.root.destroy()
        else:
            self.root.destroy()
            
    def run(self):
        """Run the launcher application"""
        self.root.mainloop()


def main():
    """Main entry point"""
    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check if we're in the right directory
    if not os.path.exists("requirements.txt") or not os.path.exists("app.py"):
        print("❌ Error: Please run this launcher from the Vector RAG Database project directory")
        print(f"Current directory: {os.getcwd()}")
        print("Expected files: requirements.txt, app.py")
        input("Press Enter to exit...")
        return
        
    # Create and run the launcher
    launcher = VectorRAGLauncher()
    launcher.run()


if __name__ == "__main__":
    main()
