#!/bin/bash

# Create proper macOS app bundle for Vector RAG Database

APP_NAME="Vector RAG Database"
APP_DIR="$HOME/Desktop/Vector RAG Database.app"
BUNDLE_ID="com.usernameTron.vectorrag"

echo "🔧 Creating macOS App Bundle: $APP_NAME"

# Remove existing app if it exists
if [ -d "$APP_DIR" ]; then
    echo "Removing existing app..."
    rm -rf "$APP_DIR"
fi

# Create app bundle structure
echo "📁 Creating bundle structure..."
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"

# Create Info.plist
echo "📝 Creating Info.plist..."
cat > "$APP_DIR/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>Vector RAG Database</string>
    <key>CFBundleIdentifier</key>
    <string>$BUNDLE_ID</string>
    <key>CFBundleName</key>
    <string>$APP_NAME</string>
    <key>CFBundleDisplayName</key>
    <string>$APP_NAME</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleSignature</key>
    <string>VRAg</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSRequiresAquaSystemAppearance</key>
    <false/>
</dict>
</plist>
EOF

# Create the executable script
echo "🚀 Creating executable script..."
cat > "$APP_DIR/Contents/MacOS/Vector RAG Database" << 'EOF'
#!/bin/bash

# Vector RAG Database Launcher
export TK_SILENCE_DEPRECATION=1

# Project directory
PROJECT_DIR="/Users/cpconnor/projects/vector-rag-database"

# Function to show error dialog
show_error() {
    osascript -e "display dialog \"$1\" buttons {\"OK\"} default button \"OK\" with icon stop with title \"Vector RAG Database\""
}

# Function to show info dialog
show_info() {
    osascript -e "display dialog \"$1\" buttons {\"OK\"} default button \"OK\" with icon note with title \"Vector RAG Database\""
}

# Check if project exists
if [ ! -d "$PROJECT_DIR" ]; then
    show_error "Vector RAG Database project not found at: $PROJECT_DIR"
    exit 1
fi

# Navigate to project directory
cd "$PROJECT_DIR" || {
    show_error "Cannot access project directory"
    exit 1
}

# Check if already running
if curl -s http://localhost:5001/api/health > /dev/null 2>&1; then
    show_info "Vector RAG Database is already running!\n\nOpening web interface..."
    open http://localhost:5001
    exit 0
fi

# Start the application
show_info "Starting Vector RAG Database...\n\nThis may take a moment."

# Start the demo server in background
nohup python3 app_demo.py > /tmp/vector_rag.log 2>&1 &

# Wait for server to start
for i in {1..15}; do
    if curl -s http://localhost:5001/api/health > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

# Check if server started successfully
if curl -s http://localhost:5001/api/health > /dev/null 2>&1; then
    show_info "🚀 Vector RAG Database launched successfully!\n\n• Web interface: http://localhost:5001\n• 6 AI Agents ready\n• Cyberpunk interface loaded"
    sleep 2
    open http://localhost:5001
else
    show_error "Failed to start Vector RAG Database server.\n\nCheck log: /tmp/vector_rag.log"
    exit 1
fi
EOF

# Make executable
chmod +x "$APP_DIR/Contents/MacOS/Vector RAG Database"

# Set proper file attributes
echo "🎯 Setting file attributes..."
SetFile -a B "$APP_DIR" 2>/dev/null || true

# Register with Launch Services
echo "📱 Registering with Launch Services..."
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$APP_DIR"

# Touch to update modification time
touch "$APP_DIR"

echo "✅ App bundle created successfully!"
echo "📍 Location: $APP_DIR"
echo "🎉 You can now double-click the app to launch Vector RAG Database!"

# Refresh Finder
killall Finder 2>/dev/null || true

echo "🔄 Finder refreshed. The app should now appear with an application icon."
