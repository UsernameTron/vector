Setting Up the Vector Repository on macOS

This step-by-step guide will help you clone and run the UsernameTron/vector repository on a MacBook. It covers all prerequisites, installation steps, running the application, verification, and common macOS issues.

1. Prerequisites

Before you begin, make sure your Mac has the following tools and requirements:

Git: Ensure Git is installed so you can clone the repository. On macOS, Git is often installed with Xcode Command Line Tools. You can install these by running:

xcode-select --install


(If you already have Git, this step is not needed.)

Python 3.7 or higher: The project requires Python 3.7+
GitHub
. macOS does not come with a recent Python by default, so install Python 3 if you haven’t already:

The easiest way is via Homebrew: brew install python3, which will provide python3 and pip3.

Alternatively, download an installer from the official Python website.

Pip and venv: These come with modern Python 3. Ensure you have pip (the Python package installer) available. Also, you'll use Python's built-in virtual environment module (venv) to avoid installing packages globally.

OpenAI API Key: Sign up at OpenAI and get an API key if you want the AI agents to function. The application requires a valid OpenAI API key in its configuration
GitHub
. Keep this key ready; you will add it to the environment settings later.

Homebrew (optional): Homebrew can help install any additional system dependencies if needed (for example, installing a Rust compiler or other libraries if a Python package needs to build native code). This isn't required for all users, but having Homebrew can simplify troubleshooting package installations on Mac.

2. Cloning the Repository

First, clone the UsernameTron/vector repository from GitHub to your local machine:

# Open a terminal and navigate to the directory where you want the project
cd ~/<desired-folder>

# Clone the repository using Git
git clone https://github.com/UsernameTron/vector.git

# Move into the project directory
cd vector


This will download the repository into a folder named vector. All the project’s files should now be present in this folder.

3. Setting Up a Python Virtual Environment

It’s recommended to use a Python virtual environment for this project. A virtual environment isolates project-specific packages so they don’t conflict with other Python projects or the system Python.

Create and activate a virtual environment:

# Create a virtual environment named "venv" (using python3)
python3 -m venv venv

# Activate the virtual environment (for bash/zsh shell)
source venv/bin/activate


After activation, your shell prompt will change (often prefixing with (venv)) indicating that the virtual environment is active. All Python packages you install next will be confined to this environment
GitHub
. (On macOS, using a venv also helps avoid permission issues since you won’t be installing packages system-wide.)

💡 Tip: If you get a “command not found” error for python3, ensure Python 3 is installed and accessible (you might need to open a new terminal or adjust your PATH). If source venv/bin/activate doesn’t work, make sure you are in the vector directory and that the venv folder was created successfully.

4. Installing Project Dependencies

With the virtual environment active (or using your system Python if you chose not to use a venv), install the required Python packages:

pip install -r requirements.txt


This command reads the requirements.txt file and installs all necessary libraries for the project (Flask, ChromaDB, OpenAI SDK, etc.)
GitHub
. It may take a few minutes, as some packages are large. You should see output as pip downloads and installs each dependency.

Potential macOS Installation Issues and Solutions:

Compiler Tools: If pip encounters errors building packages (e.g. failures involving C/C++ or Rust compilers), make sure you have Xcode command-line tools installed (xcode-select --install as mentioned earlier). Many Python packages with native extensions (like cryptography or orjson) will compile on installation if no pre-built wheel is available for your Mac architecture.

Faiss Library: The requirements include faiss-cpu, which can be tricky to install on Mac (especially on Apple Silicon M1/M2). If the installation fails at faiss-cpu, you have a couple of options:

Use Conda (if available): You can skip installing faiss-cpu via pip and install it via Conda (e.g., conda install -c conda-forge faiss-cpu) which often has a pre-built package for Mac.

Skip/Defer Faiss: Since the Vector database uses ChromaDB by default, you can temporarily remove or comment out the faiss-cpu line in requirements.txt and run pip install -r requirements.txt again to install the rest. This will allow other dependencies to install. The application can run using ChromaDB’s default settings without Faiss. (If you do this, you can try installing faiss-cpu later once everything else is set up.)

Build from source: Advanced users can attempt to compile faiss from source, but this is usually not necessary for running the app.

Permissions: If you see permission denied errors during pip install (for example, errors writing to system directories), it means pip tried to install globally without permissions. This shouldn’t happen when using a venv (everything installs locally under the project), but if you’re not using a venv and encounter this, do not use sudo with pip. Instead, activate a virtual environment or use the --user flag (e.g., pip install --user -r requirements.txt) to install packages to your home directory. The better solution is to use the virtual environment as shown above to avoid these issues.

Install Dependencies Individually: If a particular package is failing to install via the requirements file, you can try installing the main requirements one by one as a workaround
GitHub
. For example:

pip install flask flask-cors chromadb openai
pip install -r requirements.txt


Installing some core packages first can sometimes bypass issues with one problematic library.

When pip install finishes, you should have all required packages ready. If everything succeeds, you can proceed to the next step.

5. Configuration – Setting Up Environment Variables

Before running the app, you need to configure some environment variables. The project comes with a template file named .env.template in the repository. This file contains all the necessary configuration keys.

Set up your .env file:

Copy the template to a new file named .env in the project directory:

cp .env.template .env


This creates a file .env with default configurations
GitHub
.

Open the .env file in a text editor (e.g., use nano .env or open it in VS Code/TextEdit) and fill in the required values. At a minimum, set these variables:

OPENAI_API_KEY – Insert your actual OpenAI API key here (replace the placeholder in the file)
GitHub
. This key enables the AI agent functionality.

FLASK_SECRET_KEY, JWT_SECRET_KEY, ENCRYPTION_SECRET, ENCRYPTION_SALT – These are used for security (session encryption, JWT auth, etc.). For local testing, you can put any random strings here, but it’s recommended to generate secure values. The .env.template file includes instructions on how to generate secure keys using Python
GitHub
. For example, you can run python -c "import secrets; print(secrets.token_urlsafe(32))" to get a random 32-character string for each secret.

FLASK_ENV – By default this is set to development in the template. That’s fine for running locally. (In development mode, you’ll get debug output. If you set it to production, remember to also adjust other settings accordingly.)

FLASK_PORT – The template might set this to 5001. You can leave it as is. In development mode the app may still use port 5000 (see note in next section), but having this set is okay.

CHROMA_PERSIST_DIRECTORY – This is the folder where the vector database (ChromaDB) will store its data. By default it’s ./chroma_db (a folder in the project). You can leave it or change it to another path. The first time you run the app, it will create this folder.

(Any other variables in .env can typically remain at their default for a local run. Optional ones like Redis or logging config can be ignored unless needed.)

Save the .env file after editing.

Ensure that the .env file is in the same directory as the application scripts (it should be, since you copied it in place). The application will automatically load this file at runtime to read your configuration.

💡 Troubleshooting .env: If you later run the app and see errors about missing secrets or API keys, double-check that you updated the .env file correctly and that you launched the app from the project directory. The app uses python-dotenv to load this file on startup
GitHub
, so it needs to be present.

6. Running the Application Locally

With dependencies installed and the environment configured, you are ready to start the application. There are two primary ways to run the Vector app locally:

A. Run via the Flask Application (command-line)

This is the straightforward way to run the server using Python:

Activate the virtual environment if you haven’t already (from Step 3). For example:

source venv/bin/activate


(Skip if you are already in the virtual environment or chose not to use one.)

Start the Flask app by running the main application script. The repository provides a few entry-point scripts; the simplest is app.py (the basic Flask app):

python app.py


This will launch the Flask development server. In your terminal, you should see logs indicating the server is starting up (and it may print something like “Running on http://127.0.0.1:5001/” along with a Werkzeug server notice).

Note: By default, the app is configured to run on port 5001 (especially in production mode)
GitHub
. However, some documentation and the quick-start guide indicate port 5000 for the basic app
GitHub
GitHub
. Check your terminal output after running python app.py – it will tell you which port it’s listening on. In development mode it may use 5000, but if not, assume 5001. The key is to use whatever port the console says it started on.

Once the server initializes, open your web browser to http://localhost:5000 (or http://localhost:5001 if the terminal indicated port 5001). You should see the Vector RAG Database web interface load in the browser
GitHub
. This interface has a cyberpunk-themed dashboard with a list of AI agents, a chat area, and options to upload documents.

If the page loads with the header “Vector RAG Database” and shows the agent cards (Research Agent, CEO Agent, etc.), the application is up and running.

B. Run via the Desktop Launcher (GUI method)

The repository also includes a convenience GUI launcher for those who prefer a one-click startup:

One-time setup: Make sure the launcher script is executable. In the project directory run:

chmod +x desktop_launcher.py launch.sh


This gives execute permission to the launcher scripts (needed on macOS/Linux)
GitHub
.

Start via launcher: You have two options:

Run the shell launcher: ./launch.sh – This will perform some setup checks and then launch the GUI.

Or, run the Python launcher directly: python3 desktop_launcher.py – This opens a windowed application.

The launcher provides a GUI with a Start button. It will automatically start the server in the selected mode (by default “Production” mode on port 5001) and open your web browser to the app once ready
GitHub
GitHub
.

Using the GUI, you can also Stop the server and see basic status messages. It’s essentially doing the same things as the manual steps (starting the Flask app), but with a user-friendly interface.

For most purposes, the command-line method (Option A) is sufficient, but the GUI can be convenient. Either way, the end result is the server running locally.

7. Verifying the Application is Running

After starting the application, you’ll want to confirm everything works:

Web Interface Check: As mentioned, open the URL http://localhost:5000 (or 5001) in your browser. The interface should load. You should see the AI Agent Command Center with the six specialized agents listed (Research Agent, CEO Agent, etc.), a chat panel, and an upload section. This confirms the front-end is working.

Health Endpoint: The app provides a health check endpoint. You can test it by visiting http://localhost:5000/health or using curl:

curl http://localhost:5000/health


This should return a JSON response with a status (e.g. {"status":"healthy", ... }). The documentation indicates a health endpoint for sanity checking
GitHub
. If you get a healthy status, it means the backend is responding correctly.

Functionality Test: Try a simple interaction:

Click on one of the agents in the web UI (for example, the Research Agent), type a question in the chat box, and send it. The request will go to the backend and the agent (powered by OpenAI) should respond. The first query might take a moment as it initializes the AI model.

If you have no OpenAI API key or there’s a configuration issue, the app might show an error saying the agent is not available or you might see error logs in the terminal indicating a missing API key. In that case, re-check your .env file configuration.

You can also try uploading a small text document in the “Documents” section of the interface and then ask a question that the document’s content would answer, to verify the retrieval-augmented generation (RAG) flow is working.

Terminal Logs: Keep an eye on the terminal where the app is running. It will log requests and any errors. For example, when the app starts, you should see logs like Starting Vector RAG Database Application and messages about initializing the vector database and agents. On each query or action, you’ll see log output (which is useful for debugging if something isn’t working).

If the UI loads and you can engage with the agents or see the healthy status, then your local instance of Vector is running successfully!

8. Common macOS Troubleshooting Tips

Even with the above steps, you might encounter some hiccups specific to macOS. Here are some troubleshooting tips for common issues:

“Permission denied” errors on the chroma_db directory: The vector database (ChromaDB) writes to a folder (by default ./chroma_db). If you see an error like “Vector database initialization failed: [Errno 13] Permission denied” in the logs
GitHub
, it means the app couldn’t write to that directory. This can happen if the directory has wrong permissions or was created by another user. To fix:

Ensure the folder exists and is writable:

ls -ld chroma_db


If the owner or permissions look wrong, adjust them:

chmod -R 755 chroma_db


This gives the owner (you) read/write/execute and everyone else read/execute, which is usually sufficient
GitHub
.

Alternatively, you can configure a different directory for the vector DB. For instance, set CHROMA_PERSIST_DIRECTORY=$HOME/vector_data (and create that folder). You can export this in your shell or add it to the .env file
GitHub
.

Application won’t start due to missing modules: If you try to run the app and get Python import errors (e.g., “No module named 'chromadb'” or similar), it means the dependencies weren’t installed in the environment the app is using
GitHub
. Make sure you activated the correct virtual environment, and rerun the pip install -r requirements.txt. If you see only one or two missing, installing those individually as mentioned earlier can resolve it
GitHub
.

OpenAI API key issues: If the agents are not responding or you see errors about the OpenAI client, double-check that the OPENAI_API_KEY is set in .env and that it’s a valid key. The production app may log “Valid OpenAI API key not found” if the key is missing or incorrect
GitHub
. You must have internet access for the AI calls to work. If you don’t have a key or connectivity, the rest of the app (document upload/search) might work but the AI responses will not.

Mac M1/M2 specific issues: Most Python packages should install fine on Apple Silicon now, but if you hit a segmentation fault or illegal instruction error when running, it could be due to an incompatible binary. In such rare cases, running the app under Rosetta (translating to x86) might help:

You can install an x86 version of Python via Homebrew and use that, or run the Python process using Rosetta (e.g., arch -x86_64 python app.py). This is generally not needed, but it’s a last resort if some dependency isn’t M1-native and crashes.

Always try upgrading pip (pip install --upgrade pip) and reinstalling requirements first, as many issues have been resolved in newer package versions.

Firewall or Port issues: macOS might show a firewall prompt when the Flask server starts, asking if you want to allow incoming connections (especially if you have macOS firewall enabled). Click “Allow” so that your browser can talk to the local server. If you accidentally clicked “Deny,” go to System Preferences > Security & Privacy > Firewall and enable the connection for Python. Alternatively, since it’s all localhost traffic, you could temporarily disable the firewall while testing.

Using a different port: If port 5000 (or 5001) is already in use by another process on your system, the app might not start or will error out. You can edit the port in the .env (change FLASK_PORT) or modify the run command. For example, to run on a different port, you might manually start Flask on a specified port:

python app.py  # default as coded (5000/5001)


If needed, you can change the code in app.py at the very bottom where app.run(... port=5001 ...) to a different number. But normally this isn’t necessary unless you have a conflict.

Testing endpoints via curl: In addition to the health check, you can test other API endpoints with curl for deeper troubleshooting. For example:

# List agents via API
curl http://localhost:5000/api/agents 


should return a JSON listing the agent names and descriptions. This is useful to ensure the backend is functioning (especially if the front-end isn’t showing something due to a browser issue).

If you run into other issues, check the repository’s TROUBLESHOOTING.md for more detailed fixes to specific problems
GitHub
. The project’s documentation covers many edge cases and solutions. And of course, make sure to read any error messages in the terminal – they often hint at what’s wrong (missing dependency, bad config, etc.) and can guide you to the fix.

By following this guide, you should have the Vector RAG Database up and running on your MacBook. You’ve cloned the repository, set up a proper environment, installed all dependencies, configured your secrets, and launched the application. Now you can experiment with uploading documents and interacting with the specialized AI agents. Happy coding! 🎉
GitHub