// Vector RAG Database - Frontend JavaScript Application

class VectorRAGApp {
    constructor() {
        this.currentAgent = null;
        this.agents = {};
        this.isLoading = false;
        this.selectedFiles = [];
        
        this.init();
    }
    
    async init() {
        this.setupEventListeners();
        await this.loadAgents();
        await this.loadDocuments();
        this.updateStatus();
    }
    
    setupEventListeners() {
        // Chat functionality
        document.getElementById('sendButton').addEventListener('click', () => this.sendMessage());
        document.getElementById('messageInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Clear chat
        document.getElementById('clearChat').addEventListener('click', () => this.clearChat());
        
        // Document management
        document.getElementById('uploadDoc').addEventListener('click', () => this.uploadDocument());
        document.getElementById('uploadFiles').addEventListener('click', () => this.uploadFiles());
        document.getElementById('searchDocs').addEventListener('click', () => this.searchDocuments());
        
        // Search on enter
        document.getElementById('searchQuery').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.searchDocuments();
            }
        });
        
        // File upload drag and drop
        this.setupFileUpload();
        
        // Ensure paste functionality for document content
        document.getElementById('docContent').addEventListener('paste', (e) => {
            // Allow default paste behavior
            console.log('Paste event detected in document content');
        });
    }
    
    async loadAgents() {
        try {
            const response = await fetch('/api/agents');
            const data = await response.json();
            
            // Backend returns agents as direct object, not nested in agents property
            this.agents = data;
            
            console.log('Loaded agents:', this.agents);
            this.renderAgents();
        } catch (error) {
            console.error('Error loading agents:', error);
            this.showError('Failed to load AI agents');
        }
    }
    
    renderAgents() {
        const agentGrid = document.getElementById('agentGrid');
        if (!agentGrid) {
            console.error('Agent grid element not found!');
            return;
        }
        
        agentGrid.innerHTML = '';
        
        console.log('Rendering agents:', Object.keys(this.agents));
        
        const agentIcons = {
            'research': 'fas fa-microscope',
            'ceo': 'fas fa-crown',
            'performance': 'fas fa-chart-line',
            'coaching': 'fas fa-user-graduate',
            'business_intelligence': 'fas fa-brain',
            'contact_center': 'fas fa-headset'
        };
        
        Object.entries(this.agents).forEach(([key, agent]) => {
            const agentCard = document.createElement('div');
            agentCard.className = 'agent-card';
            agentCard.dataset.agentKey = key;
            
            agentCard.innerHTML = `
                <div class="agent-header">
                    <div class="agent-icon">
                        <i class="${agentIcons[key] || 'fas fa-robot'}"></i>
                    </div>
                    <div class="agent-info">
                        <h3>${agent.name}</h3>
                        <div class="role">${agent.role}</div>
                    </div>
                </div>
                <p class="agent-description">${agent.description}</p>
                <ul class="agent-capabilities">
                    ${(agent.capabilities || []).map(cap => `<li>${cap}</li>`).join('')}
                </ul>
            `;
            
            agentCard.addEventListener('click', () => this.selectAgent(key, agent));
            agentGrid.appendChild(agentCard);
        });
    }
    
    selectAgent(agentKey, agent) {
        // Update UI
        document.querySelectorAll('.agent-card').forEach(card => {
            card.classList.remove('active');
        });
        
        const selectedCard = document.querySelector(`[data-agent-key="${agentKey}"]`);
        selectedCard.classList.add('active');
        
        // Update current agent
        this.currentAgent = agentKey;
        document.getElementById('currentAgentName').textContent = agent.name;
        
        // Enable chat input
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        
        messageInput.disabled = false;
        sendButton.disabled = false;
        messageInput.placeholder = `Chat with ${agent.name}...`;
        messageInput.focus();
        
        // Clear welcome message if present
        const chatMessages = document.getElementById('chatMessages');
        if (chatMessages.querySelector('.welcome-message')) {
            chatMessages.innerHTML = '';
        }
    }
    
    async sendMessage() {
        if (!this.currentAgent || this.isLoading) return;
        
        const messageInput = document.getElementById('messageInput');
        const message = messageInput.value.trim();
        
        if (!message) return;
        
        // Clear input and show loading
        messageInput.value = '';
        this.setLoading(true);
        
        // Add user message to chat
        this.addMessage('user', message);
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    agent: this.currentAgent,
                    message: message 
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.addMessage('agent', data.response, this.agents[this.currentAgent].name);
            } else {
                this.addMessage('agent', `Error: ${data.error}`, 'System');
            }
        } catch (error) {
            console.error('Error sending message:', error);
            this.addMessage('agent', 'Sorry, I encountered an error processing your request.', 'System');
        } finally {
            this.setLoading(false);
        }
    }
    
    addMessage(type, content, agentName = null) {
        const chatMessages = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        const timestamp = new Date().toLocaleTimeString();
        const sender = type === 'user' ? 'You' : agentName || 'Agent';
        
        messageDiv.innerHTML = `
            <div class="message-content">${this.formatMessage(content)}</div>
            <div class="message-meta">${sender} • ${timestamp}</div>
        `;
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    formatMessage(content) {
        // Basic markdown-like formatting
        return content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>');
    }
    
    clearChat() {
        const chatMessages = document.getElementById('chatMessages');
        chatMessages.innerHTML = `
            <div class="welcome-message">
                <i class="fas fa-brain"></i>
                <h4>Chat Cleared</h4>
                <p>Continue your conversation with ${this.currentAgent ? this.agents[this.currentAgent].name : 'your selected agent'}.</p>
            </div>
        `;
    }
    
    async uploadDocument() {
        const title = document.getElementById('docTitle').value.trim();
        const content = document.getElementById('docContent').value.trim();
        const source = document.getElementById('docSource').value.trim() || 'user_upload';
        
        if (!title || !content) {
            this.showError('Please provide both title and content for the document.');
            return;
        }
        
        this.setLoading(true);
        
        try {
            // Send as JSON to match backend expectations
            const requestBody = {
                title: title,
                content: content,
                source: source
            };
            
            const response = await fetch('/api/documents', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.showSuccess('Document uploaded successfully!');
                // Clear form
                document.getElementById('docTitle').value = '';
                document.getElementById('docContent').value = '';
                document.getElementById('docSource').value = '';
                // Reload documents
                await this.loadDocuments();
            } else {
                this.showError(`Failed to upload document: ${data.error}`);
            }
        } catch (error) {
            console.error('Error uploading document:', error);
            this.showError('Failed to upload document.');
        } finally {
            this.setLoading(false);
        }
    }
    
    async searchDocuments() {
        const query = document.getElementById('searchQuery').value.trim();
        
        if (!query) {
            this.showError('Please enter a search query.');
            return;
        }
        
        this.setLoading(true);
        
        try {
            const response = await fetch('/api/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query, limit: 10 })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.displaySearchResults(data.results);
            } else {
                this.showError(`Search failed: ${data.error}`);
            }
        } catch (error) {
            console.error('Error searching documents:', error);
            this.showError('Search failed.');
        } finally {
            this.setLoading(false);
        }
    }
    
    displaySearchResults(results) {
        const searchResults = document.getElementById('searchResults');
        
        if (results.length === 0) {
            searchResults.innerHTML = '<p style="color: var(--text-muted); text-align: center; padding: 1rem;">No results found.</p>';
            return;
        }
        
        searchResults.innerHTML = results.map(result => `
            <div class="search-result" onclick="app.showDocumentDetails('${result.id}')">
                <div class="doc-title">${result.metadata.title}</div>
                <div class="doc-meta">
                    Source: ${result.metadata.source} • 
                    Added: ${new Date(result.metadata.timestamp).toLocaleDateString()}
                    ${result.distance ? ` • Similarity: ${(1 - result.distance).toFixed(2)}` : ''}
                </div>
                <div class="doc-preview">${result.content.substring(0, 200)}...</div>
            </div>
        `).join('');
    }
    
    async loadDocuments() {
        try {
            const response = await fetch('/api/documents');
            if (response.ok) {
                const documents = await response.json();
                this.displayDocuments(documents);
            } else {
                // Fallback to status check
                const statusResponse = await fetch('/api/status');
                if (statusResponse.ok) {
                    const status = await statusResponse.json();
                    const documentList = document.getElementById('documentList');
                    if (documentList) {
                        const docCount = status.database_status?.document_count || 0;
                        documentList.innerHTML = `
                            <div style="color: var(--text-muted); text-align: center; padding: 1rem;">
                                <i class="fas fa-database" style="font-size: 2rem; margin-bottom: 0.5rem;"></i>
                                <p>Vector database contains ${docCount} documents</p>
                                <p>Upload documents using the form above</p>
                            </div>
                        `;
                    }
                }
            }
        } catch (error) {
            console.error('Error loading documents:', error);
            const documentList = document.getElementById('documentList');
            if (documentList) {
                documentList.innerHTML = `
                    <div style="color: var(--text-muted); text-align: center; padding: 1rem;">
                        <i class="fas fa-exclamation-triangle" style="font-size: 2rem; margin-bottom: 0.5rem; color: var(--warning-color);"></i>
                        <p>Could not load documents</p>
                        <p>Upload documents using the form above</p>
                    </div>
                `;
            }
        }
    }
    
    displayDocuments(documents) {
        const documentList = document.getElementById('documentList');
        
        if (!documents || !Array.isArray(documents) || documents.length === 0) {
            if (documentList) {
                documentList.innerHTML = '<p style="color: var(--text-muted); text-align: center; padding: 1rem;">No documents in knowledge base.</p>';
            }
            return;
        }

        if (documentList) {
            documentList.innerHTML = documents.map(doc => `
                <div class="doc-item" onclick="app.showDocumentDetails('${doc.id}')">
                    <div class="doc-title">${doc.metadata?.title || 'Untitled'}</div>
                    <div class="doc-meta">
                        Source: ${doc.metadata?.source || 'Unknown'} • 
                        Added: ${doc.metadata?.timestamp ? new Date(doc.metadata.timestamp).toLocaleDateString() : 'Unknown date'} •
                        Length: ${doc.metadata?.content_length || 0} characters
                    </div>
                    <div class="doc-preview">${doc.content_preview || ''}</div>
                </div>
            `).join('');
        }
    }
    
    showDocumentDetails(docId) {
        // This could be expanded to show a modal with full document details
        console.log('Show document details for:', docId);
        // For now, just highlight the document
        document.querySelectorAll('.doc-item, .search-result').forEach(item => {
            item.style.border = '1px solid var(--border-color)';
        });
        event.target.closest('.doc-item, .search-result').style.border = '2px solid var(--primary-blue)';
    }
    
    async updateStatus() {
        try {
            const response = await fetch('/health');
            const status = await response.json();
            
            const statusDot = document.getElementById('statusDot');
            const statusText = document.getElementById('statusText');
            
            if (response.ok && status.status === 'healthy') {
                if (statusDot) {
                    statusDot.style.background = 'var(--success-color)';
                    statusDot.style.boxShadow = '0 0 10px var(--success-color)';
                }
                if (statusText) {
                    statusText.textContent = `Online • ${status.agents_available || 0} Agents • Vector DB ${status.vector_db_available ? 'Ready' : 'Offline'}`;
                }
            } else {
                if (statusDot) {
                    statusDot.style.background = 'var(--error-color)';
                    statusDot.style.boxShadow = '0 0 10px var(--error-color)';
                }
                if (statusText) {
                    statusText.textContent = 'System Error';
                }
            }
        } catch (error) {
            console.error('Error checking status:', error);
            const statusDot = document.getElementById('statusDot');
            const statusText = document.getElementById('statusText');
            if (statusDot) {
                statusDot.style.background = 'var(--warning-color)';
                statusDot.style.boxShadow = '0 0 10px var(--warning-color)';
            }
            if (statusText) {
                statusText.textContent = 'Connection Error';
            }
        }
    }
    
    setLoading(loading) {
        this.isLoading = loading;
        const loadingOverlay = document.getElementById('loadingOverlay');
        loadingOverlay.style.display = loading ? 'flex' : 'none';
        
        // Disable/enable inputs
        const inputs = ['messageInput', 'sendButton', 'uploadDoc', 'searchDocs'];
        inputs.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.disabled = loading;
            }
        });
    }
    
    showError(message) {
        this.showNotification(message, 'error');
    }
    
    showSuccess(message) {
        this.showNotification(message, 'success');
    }
    
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'error' ? 'var(--error-color)' : type === 'success' ? 'var(--success-color)' : 'var(--primary-blue)'};
            color: var(--bg-primary);
            padding: 1rem 1.5rem;
            border-radius: 8px;
            box-shadow: var(--glow-primary);
            z-index: 1001;
            animation: slideIn 0.3s ease-out;
            max-width: 300px;
            word-wrap: break-word;
        `;
        
        notification.innerHTML = `
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <i class="fas ${type === 'error' ? 'fa-exclamation-triangle' : type === 'success' ? 'fa-check-circle' : 'fa-info-circle'}"></i>
                <span>${message}</span>
            </div>
        `;
        
        // Add CSS animation
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideIn {
                from {
                    transform: translateX(100%);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }
        `;
        document.head.appendChild(style);
        
        document.body.appendChild(notification);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            notification.style.animation = 'slideIn 0.3s ease-out reverse';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 5000);
        
        // Click to dismiss
        notification.addEventListener('click', () => {
            notification.style.animation = 'slideIn 0.3s ease-out reverse';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        });
    }
    
    setupFileUpload() {
        const fileInput = document.getElementById('docFile');
        const fileLabel = document.querySelector('.file-upload-label');
        
        // File input change handler
        fileInput.addEventListener('change', (e) => {
            this.handleFileSelection(e.target.files);
        });
        
        // Drag and drop handlers
        fileLabel.addEventListener('dragover', (e) => {
            e.preventDefault();
            fileLabel.classList.add('drag-over');
        });
        
        fileLabel.addEventListener('dragleave', (e) => {
            e.preventDefault();
            fileLabel.classList.remove('drag-over');
        });
        
        fileLabel.addEventListener('drop', (e) => {
            e.preventDefault();
            fileLabel.classList.remove('drag-over');
            this.handleFileSelection(e.dataTransfer.files);
        });
    }
    
    handleFileSelection(files) {
        this.selectedFiles = Array.from(files);
        this.displaySelectedFiles();
    }
    
    displaySelectedFiles() {
        const fileUploadSection = document.getElementById('fileUploadSection');
        
        // Remove existing file list
        const existingList = fileUploadSection.querySelector('.file-list');
        if (existingList) {
            existingList.remove();
        }
        
        if (this.selectedFiles.length === 0) return;
        
        // Create file list
        const fileList = document.createElement('div');
        fileList.className = 'file-list';
        
        this.selectedFiles.forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            
            fileItem.innerHTML = `
                <div>
                    <div class="file-name">${file.name}</div>
                    <div class="file-size">${this.formatFileSize(file.size)}</div>
                </div>
                <button class="file-remove" onclick="app.removeFile(${index})">
                    <i class="fas fa-times"></i>
                </button>
            `;
            
            fileList.appendChild(fileItem);
        });
        
        // Insert file list before the upload button
        const uploadButton = document.getElementById('uploadFiles');
        fileUploadSection.insertBefore(fileList, uploadButton);
    }
    
    removeFile(index) {
        this.selectedFiles.splice(index, 1);
        this.displaySelectedFiles();
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    switchUploadMode(mode) {
        const fileTab = document.getElementById('fileTab');
        const textTab = document.getElementById('textTab');
        const fileSection = document.getElementById('fileUploadSection');
        const textSection = document.getElementById('textUploadSection');
        
        if (mode === 'file') {
            fileTab.classList.add('active');
            textTab.classList.remove('active');
            fileSection.style.display = 'block';
            textSection.style.display = 'none';
        } else {
            fileTab.classList.remove('active');
            textTab.classList.add('active');
            fileSection.style.display = 'none';
            textSection.style.display = 'block';
        }
    }
    
    async uploadFiles() {
        if (!this.selectedFiles || this.selectedFiles.length === 0) {
            this.showError('Please select files to upload.');
            return;
        }
        
        this.setLoading(true);
        const source = document.getElementById('fileSource').value.trim() || 'file_upload';
        
        try {
            const uploadPromises = this.selectedFiles.map(file => this.uploadSingleFile(file, source));
            const results = await Promise.all(uploadPromises);
            
            const successCount = results.filter(r => r.success).length;
            const failCount = results.length - successCount;
            
            if (successCount > 0) {
                this.showSuccess(`Successfully uploaded ${successCount} file(s)${failCount > 0 ? `, ${failCount} failed` : ''}`);
                
                // Clear form
                document.getElementById('docFile').value = '';
                document.getElementById('fileSource').value = '';
                this.selectedFiles = [];
                this.displaySelectedFiles();
                
                // Reload documents
                await this.loadDocuments();
            } else {
                this.showError('All file uploads failed.');
            }
        } catch (error) {
            console.error('Error uploading files:', error);
            this.showError('Failed to upload files.');
        } finally {
            this.setLoading(false);
        }
    }
    
    async uploadSingleFile(file, source) {
        try {
            const content = await this.readFileAsText(file);
            const title = file.name.replace(/\.[^/.]+$/, ""); // Remove file extension
            
            const requestBody = {
                title: title,
                content: content,
                source: source
            };
            
            const response = await fetch('/api/documents', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });
            
            const data = await response.json();
            return { success: response.ok, data, filename: file.name };
        } catch (error) {
            console.error(`Error uploading ${file.name}:`, error);
            return { success: false, error, filename: file.name };
        }
    }
    
    readFileAsText(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = (e) => reject(e);
            reader.readAsText(file);
        });
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new VectorRAGApp();
    
    // Update status every 30 seconds
    setInterval(() => {
        window.app.updateStatus();
    }, 30000);
});
