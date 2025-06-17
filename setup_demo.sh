#!/bin/bash

# Vector RAG Database - Quick Demo Setup
# This script sets up the application with sample data for demonstration

echo "🎬 Setting up Vector RAG Database Demo"
echo "======================================"

# Create sample .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️  Creating demo .env file..."
    cat > .env << EOF
# Demo Environment Variables for Vector RAG Database
OPENAI_API_KEY=your_openai_api_key_here
FLASK_ENV=development
FLASK_DEBUG=True
CHROMA_PERSIST_DIRECTORY=./chroma_db
DEFAULT_COLLECTION_NAME=vector_rag_collection
EOF
    echo "✅ Created .env file - Please add your OpenAI API key"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create chroma_db directory
mkdir -p chroma_db

# Create sample documents for demo
echo "📄 Creating sample knowledge base documents..."
cat > sample_data.py << 'EOF'
import requests
import json

# Sample documents to populate the knowledge base
sample_docs = [
    {
        "title": "Company Performance Metrics Q1 2025",
        "content": """Our Q1 2025 performance shows strong growth across all key metrics:

Revenue: $2.4M (up 15% from Q4 2024)
Customer Acquisition: 1,200 new customers (up 25%)
Customer Satisfaction: 4.7/5.0 (industry leading)
Employee Satisfaction: 4.5/5.0
Market Share: 12% (up from 10%)

Key achievements:
- Launched new AI-powered customer service platform
- Expanded to 3 new geographic markets
- Reduced customer churn by 30%
- Improved operational efficiency by 20%

Challenges:
- Higher customer acquisition costs
- Increased competition in core markets
- Technical debt in legacy systems""",
        "source": "quarterly_report"
    },
    {
        "title": "AI Agent Implementation Best Practices",
        "content": """Best practices for implementing AI agents in enterprise environments:

1. Define Clear Objectives
   - Establish specific use cases and success metrics
   - Align with business goals and user needs
   - Set realistic expectations for AI capabilities

2. Data Quality and Preparation
   - Ensure high-quality training data
   - Implement proper data governance
   - Regular data validation and updates

3. Human-in-the-Loop Design
   - Maintain human oversight for critical decisions
   - Provide clear escalation paths
   - Regular performance monitoring and feedback

4. Security and Privacy
   - Implement robust authentication and authorization
   - Encrypt sensitive data in transit and at rest
   - Regular security audits and compliance checks

5. Continuous Learning and Improvement
   - Monitor agent performance metrics
   - Collect user feedback for improvements
   - Regular model updates and retraining""",
        "source": "technical_documentation"
    },
    {
        "title": "Customer Service Excellence Framework",
        "content": """Our customer service excellence framework focuses on delivering exceptional experiences:

Core Principles:
1. Customer-Centric Approach
   - Put customer needs first in all interactions
   - Personalize service based on customer history
   - Proactive communication and follow-up

2. First Call Resolution (FCR)
   - Target: 85% FCR rate
   - Empower agents with comprehensive tools
   - Knowledge base integration for quick answers

3. Response Time Standards
   - Phone: Answer within 3 rings
   - Email: Respond within 2 hours
   - Chat: Respond within 30 seconds
   - Social Media: Respond within 1 hour

4. Quality Assurance
   - Monitor 10% of all interactions
   - Customer satisfaction surveys after each interaction
   - Regular coaching and training sessions

5. Technology Integration
   - CRM integration for complete customer view
   - AI-powered sentiment analysis
   - Automated routing based on expertise

Key Metrics:
- CSAT: 4.7/5.0
- NPS: 72 (excellent)
- Average Handle Time: 4.2 minutes
- Agent Utilization: 78%""",
        "source": "operations_manual"
    },
    {
        "title": "Leadership Development Program 2025",
        "content": """Our comprehensive leadership development program for 2025:

Program Overview:
- Duration: 12 months
- Participants: 50 high-potential employees
- Investment: $500K total program budget
- ROI Target: 300% within 24 months

Core Modules:
1. Strategic Thinking and Vision (Month 1-2)
   - Market analysis and competitive positioning
   - Long-term strategic planning
   - Vision communication and alignment

2. People Leadership (Month 3-4)
   - Team building and motivation
   - Performance management
   - Conflict resolution and mediation

3. Change Management (Month 5-6)
   - Leading organizational transformation
   - Change communication strategies
   - Resistance management

4. Data-Driven Decision Making (Month 7-8)
   - Analytics and business intelligence
   - KPI development and monitoring
   - Evidence-based leadership

5. Innovation and Growth (Month 9-10)
   - Creative problem solving
   - Innovation frameworks
   - Growth strategy development

6. Executive Presence (Month 11-12)
   - Communication and presentation skills
   - Stakeholder management
   - Board-level reporting

Success Metrics:
- 360-degree feedback improvement: 25%
- Promotion rate: 40% within 18 months
- Employee engagement scores: +15%
- Business impact measurement""",
        "source": "hr_development"
    },
    {
        "title": "Business Intelligence Dashboard Requirements",
        "content": """Requirements for our new executive BI dashboard:

Executive Summary Dashboard:
- Real-time revenue tracking
- Customer acquisition funnel
- Key performance indicators (KPIs)
- Predictive analytics for next quarter

Sales Performance:
- Sales by region, product, and salesperson
- Pipeline analysis and forecasting
- Win/loss analysis
- Customer lifetime value (CLV)

Operations Metrics:
- Service level agreements (SLA) performance
- Resource utilization rates
- Quality metrics and defect rates
- Cost per unit/transaction

Financial Analytics:
- P&L analysis with drill-down capabilities
- Cash flow projections
- Budget vs. actual reporting
- Profitability analysis by segment

Customer Analytics:
- Customer satisfaction trends
- Churn analysis and prediction
- Support ticket analysis
- Usage patterns and engagement

Technical Requirements:
- Real-time data processing (< 5 minute latency)
- Mobile-responsive design
- Role-based access control
- Export capabilities (PDF, Excel, PowerPoint)
- API integration with existing systems
- Automated alert system for threshold breaches

Data Sources:
- CRM system (Salesforce)
- ERP system (SAP)
- Customer support platform
- Web analytics (Google Analytics)
- Financial systems""",
        "source": "project_requirements"
    }
]

def upload_sample_data():
    base_url = "http://localhost:5000"
    
    print("Uploading sample documents to knowledge base...")
    
    for i, doc in enumerate(sample_docs, 1):
        try:
            response = requests.post(
                f"{base_url}/api/documents",
                json=doc,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                print(f"✅ Uploaded document {i}: {doc['title']}")
            else:
                print(f"❌ Failed to upload document {i}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print(f"❌ Could not connect to server. Make sure the app is running on {base_url}")
            return False
        except Exception as e:
            print(f"❌ Error uploading document {i}: {str(e)}")
    
    return True

if __name__ == "__main__":
    upload_sample_data()
EOF

echo ""
echo "🌟 Demo setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Add your OpenAI API key to the .env file"
echo "2. Start the application: python app.py"
echo "3. Open http://localhost:5000 in your browser"
echo "4. Run 'python sample_data.py' to populate with demo data"
echo ""
echo "🎯 Demo scenarios to try:"
echo "• Ask the Research Agent about Q1 performance metrics"
echo "• Consult the CEO Agent for strategic planning advice"
echo "• Query the Performance Agent about KPI optimization"
echo "• Chat with the Coaching Agent about leadership development"
echo "• Ask the BI Agent about dashboard requirements"
echo "• Consult the Contact Center Director about service excellence"
echo ""
echo "🚀 Enjoy exploring your Vector RAG Database!"
