"""
AI Agents Module
Specialized AI agents for different business functions
"""

import openai
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base class for all AI agents"""
    
    def __init__(self, vector_db, name: str, role: str, description: str, capabilities: List[str]):
        self.vector_db = vector_db
        self.name = name
        self.role = role
        self.description = description
        self.capabilities = capabilities
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def get_context(self, query: str, limit: int = 3) -> str:
        """Retrieve relevant context from vector database"""
        try:
            results = self.vector_db.search(query, limit)
            context = "\n\n".join([f"Document: {result['metadata'].get('title', 'Untitled')}\nContent: {result['content']}" for result in results])
            return context
        except Exception as e:
            logger.error(f"Error getting context: {str(e)}")
            return ""
    
    def generate_response(self, system_prompt: str, user_query: str, context: str = "") -> str:
        """Generate response using OpenAI GPT"""
        try:
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            if context:
                messages.append({
                    "role": "system", 
                    "content": f"Relevant context from knowledge base:\n{context}"
                })
            
            messages.append({"role": "user", "content": user_query})
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I apologize, but I encountered an error processing your request: {str(e)}"
    
    @abstractmethod
    def process_query(self, query: str) -> str:
        """Process a user query and return a response"""
        pass

class ResearchAgent(BaseAgent):
    """AI Agent specialized in research and data analysis"""
    
    def __init__(self, vector_db):
        super().__init__(
            vector_db=vector_db,
            name="Research Agent",
            role="Senior Research Analyst",
            description="Specializes in conducting thorough research, data analysis, and providing insights from various sources.",
            capabilities=[
                "Document analysis and synthesis",
                "Market research and competitive analysis", 
                "Data interpretation and visualization",
                "Trend identification and forecasting",
                "Academic and business research"
            ]
        )
    
    def process_query(self, query: str) -> str:
        """Process research-related queries"""
        system_prompt = f"""You are a {self.role} with expertise in research and data analysis. 
        Your capabilities include: {', '.join(self.capabilities)}.
        
        Provide thorough, evidence-based responses with clear analysis and actionable insights.
        If you reference any data or sources, be specific about their reliability and relevance.
        Structure your responses with clear headings and bullet points when appropriate."""
        
        context = self.get_context(query)
        return self.generate_response(system_prompt, query, context)

class CEOAgent(BaseAgent):
    """AI Agent with CEO-level strategic thinking and leadership"""
    
    def __init__(self, vector_db):
        super().__init__(
            vector_db=vector_db,
            name="CEO Agent",
            role="Chief Executive Officer",
            description="Provides strategic leadership, high-level decision making, and executive insights.",
            capabilities=[
                "Strategic planning and vision setting",
                "Executive decision making",
                "Business model optimization",
                "Stakeholder management",
                "Corporate governance and leadership"
            ]
        )
    
    def process_query(self, query: str) -> str:
        """Process strategic and leadership queries"""
        system_prompt = f"""You are a {self.role} with extensive experience in strategic leadership and executive management.
        Your capabilities include: {', '.join(self.capabilities)}.
        
        Provide strategic, high-level insights with focus on business impact, long-term vision, and stakeholder value.
        Consider multiple perspectives including financial, operational, and market factors.
        Communicate with executive presence and clarity."""
        
        context = self.get_context(query)
        return self.generate_response(system_prompt, query, context)

class PerformanceAgent(BaseAgent):
    """AI Agent focused on performance optimization and metrics"""
    
    def __init__(self, vector_db):
        super().__init__(
            vector_db=vector_db,
            name="Performance Agent",
            role="Performance Optimization Specialist",
            description="Analyzes performance metrics, identifies optimization opportunities, and drives efficiency improvements.",
            capabilities=[
                "Performance metrics analysis",
                "KPI development and tracking",
                "Process optimization",
                "Efficiency improvement strategies",
                "Benchmarking and comparative analysis"
            ]
        )
    
    def process_query(self, query: str) -> str:
        """Process performance and optimization queries"""
        system_prompt = f"""You are a {self.role} specialized in performance optimization and metrics analysis.
        Your capabilities include: {', '.join(self.capabilities)}.
        
        Focus on quantifiable improvements, measurable outcomes, and data-driven recommendations.
        Provide specific metrics, benchmarks, and actionable optimization strategies.
        Consider both short-term gains and long-term performance sustainability."""
        
        context = self.get_context(query)
        return self.generate_response(system_prompt, query, context)

class CoachingAgent(BaseAgent):
    """AI Agent specialized in coaching and development"""
    
    def __init__(self, vector_db):
        super().__init__(
            vector_db=vector_db,
            name="Coaching Agent",
            role="Executive Coach & Development Specialist",
            description="Provides personalized coaching, skill development guidance, and leadership mentoring.",
            capabilities=[
                "Leadership coaching and mentoring",
                "Skill assessment and development planning",
                "Career guidance and progression",
                "Team development and dynamics",
                "Communication and interpersonal skills"
            ]
        )
    
    def process_query(self, query: str) -> str:
        """Process coaching and development queries"""
        system_prompt = f"""You are a {self.role} with expertise in human development and coaching.
        Your capabilities include: {', '.join(self.capabilities)}.
        
        Provide supportive, constructive guidance with focus on growth and development.
        Use coaching methodologies like asking powerful questions and providing actionable feedback.
        Tailor your approach to individual needs and learning styles."""
        
        context = self.get_context(query)
        return self.generate_response(system_prompt, query, context)

class BusinessIntelligenceAgent(BaseAgent):
    """AI Agent specialized in business intelligence and analytics"""
    
    def __init__(self, vector_db):
        super().__init__(
            vector_db=vector_db,
            name="Business Intelligence Agent",
            role="Business Intelligence Director",
            description="Analyzes business data, generates insights, and provides intelligence for strategic decision making.",
            capabilities=[
                "Data analytics and visualization",
                "Business intelligence reporting",
                "Predictive modeling and forecasting",
                "Market intelligence and trends",
                "Dashboard development and KPI tracking"
            ]
        )
    
    def process_query(self, query: str) -> str:
        """Process business intelligence queries"""
        system_prompt = f"""You are a {self.role} with expertise in business intelligence and data analytics.
        Your capabilities include: {', '.join(self.capabilities)}.
        
        Provide data-driven insights with clear visualizations and actionable recommendations.
        Focus on business impact, trends, and predictive analysis.
        Present complex data in accessible formats with clear takeaways for stakeholders."""
        
        context = self.get_context(query)
        return self.generate_response(system_prompt, query, context)

class ContactCenterDirectorAgent(BaseAgent):
    """AI Agent specialized in contact center operations and customer service"""
    
    def __init__(self, vector_db):
        super().__init__(
            vector_db=vector_db,
            name="Contact Center Director Agent",
            role="Contact Center Director",
            description="Manages contact center operations, customer experience optimization, and service quality assurance.",
            capabilities=[
                "Contact center operations management",
                "Customer experience optimization",
                "Quality assurance and monitoring",
                "Agent training and development",
                "Service level management and metrics"
            ]
        )
    
    def process_query(self, query: str) -> str:
        """Process contact center and customer service queries"""
        system_prompt = f"""You are a {self.role} with extensive experience in contact center operations and customer service.
        Your capabilities include: {', '.join(self.capabilities)}.
        
        Focus on customer satisfaction, operational efficiency, and service quality.
        Provide practical solutions for contact center challenges and customer experience improvements.
        Consider metrics like CSAT, NPS, FCR, and AHT in your recommendations."""
        
        context = self.get_context(query)
        return self.generate_response(system_prompt, query, context)
