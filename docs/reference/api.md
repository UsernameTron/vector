# API Reference

## Base URL

```
Production: http://localhost:5001
Development: http://localhost:5000
```

## Response Format

All API responses follow this structure:

```json
{
  "status": "success|error",
  "data": {},
  "message": "Human-readable message",
  "timestamp": "ISO 8601 timestamp",
  "errors": [],  // Optional
  "pagination": {}  // Optional
}
```

## Endpoints

### Health Check

**GET** `/health`

Check application health and status.

**Response:**
```json
{
  "status": "healthy",
  "agents_available": 8,
  "vector_db_available": true,
  "mode": "production",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Agents

#### List Available Agents

**GET** `/api/agents`

Get list of all available AI agents.

**Response:**
```json
{
  "status": "success",
  "data": [
    {
      "type": "research",
      "name": "Research Agent",
      "description": "Deep analysis and information synthesis",
      "capabilities": ["analysis", "research", "synthesis"]
    }
  ]
}
```

### Chat

#### Send Chat Message

**POST** `/api/chat`

Chat with a specific AI agent.

**Request Body:**
```json
{
  "agent_type": "research|ceo|performance|coaching|code_analyzer|triage|bi|contact_center",
  "query": "Your question here",
  "context": [],  // Optional previous messages
  "use_rag": true  // Optional, default true
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "response": "Agent's response",
    "agent_type": "research",
    "context_used": ["doc1", "doc2"],
    "confidence": 0.95
  }
}
```

### Documents

#### List Documents

**GET** `/api/documents`

Get all documents in the vector database.

**Query Parameters:**
- `limit`: Maximum number of documents (default: 100)
- `offset`: Pagination offset (default: 0)
- `source`: Filter by source

**Response:**
```json
{
  "status": "success",
  "data": [
    {
      "id": "uuid",
      "title": "Document Title",
      "content_preview": "First 200 characters...",
      "metadata": {
        "source": "upload",
        "timestamp": "2024-01-01T12:00:00Z",
        "content_length": 1234
      }
    }
  ],
  "pagination": {
    "total": 100,
    "limit": 10,
    "offset": 0
  }
}
```

#### Get Document

**GET** `/api/documents/:id`

Get a specific document by ID.

**Response:**
```json
{
  "status": "success",
  "data": {
    "id": "uuid",
    "title": "Document Title",
    "content": "Full document content",
    "metadata": {},
    "embedding_status": "complete"
  }
}
```

#### Create Document

**POST** `/api/documents`

Add a new document to the vector database.

**Request Body:**
```json
{
  "content": "Document content",
  "title": "Document Title",
  "source": "manual",
  "metadata": {}  // Optional
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "id": "uuid",
    "message": "Document created successfully"
  }
}
```

#### Update Document

**PUT** `/api/documents/:id`

Update an existing document.

**Request Body:**
```json
{
  "content": "Updated content",
  "title": "Updated Title",
  "metadata": {}
}
```

#### Delete Document

**DELETE** `/api/documents/:id`

Remove a document from the database.

**Response:**
```json
{
  "status": "success",
  "message": "Document deleted successfully"
}
```

### Search

#### Vector Search

**POST** `/api/search`

Perform similarity search on documents.

**Request Body:**
```json
{
  "query": "Search query",
  "top_k": 5,  // Number of results
  "filter": {  // Optional metadata filters
    "source": "upload",
    "agent_type": "research"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "results": [
      {
        "id": "uuid",
        "content": "Relevant content",
        "score": 0.95,
        "metadata": {}
      }
    ],
    "query": "Search query",
    "count": 5
  }
}
```

### File Upload

#### Upload CSV

**POST** `/api/upload/csv`

Upload and process CSV file.

**Request:** Multipart form data
- `file`: CSV file
- `source`: Optional source identifier

**Response:**
```json
{
  "status": "success",
  "data": {
    "documents_processed": 100,
    "documents_stored": 100,
    "parsing_stats": {
      "rows": 100,
      "columns": 10,
      "errors": 0
    }
  }
}
```

#### Upload Excel

**POST** `/api/upload/excel`

Upload and process Excel file.

**Request:** Multipart form data
- `file`: Excel file (.xlsx, .xls)
- `sheet_name`: Optional specific sheet

**Response:** Similar to CSV upload

#### Upload PDF

**POST** `/api/upload/pdf`

Upload and process PDF file.

**Request:** Multipart form data
- `file`: PDF file
- `extract_images`: Optional boolean

### Analytics

#### Agent Usage Stats

**GET** `/api/analytics/agents`

Get usage statistics for agents.

**Response:**
```json
{
  "status": "success",
  "data": {
    "total_queries": 1000,
    "by_agent": {
      "research": 300,
      "ceo": 200
    },
    "avg_response_time": 1.5,
    "success_rate": 0.98
  }
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Invalid API key |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource doesn't exist |
| 413 | Payload Too Large - File too big |
| 429 | Too Many Requests - Rate limited |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

## Rate Limiting

Default limits:
- 100 requests per minute per IP
- 10 concurrent requests per IP
- 50MB max file upload size

## Authentication

### API Key Authentication

Include API key in headers:
```
X-API-Key: your-api-key-here
```

### JWT Token (Clean Architecture Mode)

```
Authorization: Bearer your-jwt-token
```

## WebSocket Events

### Connection

```javascript
const ws = new WebSocket('ws://localhost:5001/ws');
```

### Events

**chat:message**
```json
{
  "type": "chat:message",
  "data": {
    "agent": "research",
    "message": "Response text",
    "timestamp": "2024-01-01T12:00:00Z"
  }
}
```

**document:processed**
```json
{
  "type": "document:processed",
  "data": {
    "id": "uuid",
    "status": "complete",
    "chunks": 10
  }
}
```

## Code Examples

### Python

```python
import requests

# Chat with agent
response = requests.post(
    "http://localhost:5001/api/chat",
    json={
        "agent_type": "research",
        "query": "What is machine learning?"
    }
)
print(response.json())
```

### JavaScript

```javascript
// Upload file
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:5001/api/upload/csv', {
    method: 'POST',
    body: formData
})
.then(res => res.json())
.then(data => console.log(data));
```

### cURL

```bash
# Search documents
curl -X POST http://localhost:5001/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "vector databases",
    "top_k": 5
  }'
```