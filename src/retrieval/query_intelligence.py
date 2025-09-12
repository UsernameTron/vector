"""
Query Intelligence and Expansion Framework

This module provides sophisticated query analysis and expansion capabilities:
- Query classification and intent detection
- Automatic query expansion with synonyms and domain terms
- Agent-specific query optimization
- Multi-strategy query generation
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, Counter
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries for classification"""
    FACTUAL = "factual"                 # Direct fact lookup
    ANALYTICAL = "analytical"           # Analysis and insights
    COMPARATIVE = "comparative"         # Comparisons between entities
    PROCEDURAL = "procedural"           # How-to and process questions
    TEMPORAL = "temporal"               # Time-based questions
    CAUSAL = "causal"                   # Cause-effect relationships
    EXPLORATORY = "exploratory"        # Open-ended exploration
    DEFINITIONAL = "definitional"       # Definitions and explanations


class QueryIntent(Enum):
    """Intent categories for different query purposes"""
    INFORMATION_SEEKING = "information_seeking"
    PROBLEM_SOLVING = "problem_solving"
    DECISION_MAKING = "decision_making"
    MONITORING = "monitoring"
    OPTIMIZATION = "optimization"
    PLANNING = "planning"
    ANALYSIS = "analysis"
    RESEARCH = "research"


class ExpansionStrategy(Enum):
    """Query expansion strategies"""
    SYNONYM_BASED = "synonym_based"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    DOMAIN_SPECIFIC = "domain_specific"
    CONTEXTUAL = "contextual"
    HIERARCHICAL = "hierarchical"
    TEMPORAL = "temporal"


@dataclass
class QueryAnalysis:
    """Comprehensive query analysis results"""
    original_query: str
    query_type: QueryType
    intent: QueryIntent
    confidence: float
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    temporal_indicators: List[str] = field(default_factory=list)
    complexity_score: float = 0.0
    domain_indicators: List[str] = field(default_factory=list)
    question_words: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpandedQuery:
    """Expanded query with multiple variations"""
    original_query: str
    expanded_terms: List[str]
    synonym_expansions: List[str]
    domain_expansions: List[str]
    semantic_variations: List[str]
    combined_query: str
    expansion_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryRoute:
    """Routing information for agent-specific optimization"""
    target_agent: str
    optimized_query: str
    context_hints: List[str]
    retrieval_strategy: str
    confidence: float


class QueryClassifier:
    """Advanced query classification system"""
    
    def __init__(self):
        # Define classification patterns and indicators
        self.type_patterns = {
            QueryType.FACTUAL: {
                'patterns': [
                    r'\bwhat\s+(is|are|was|were)\b',
                    r'\bwho\s+(is|are|was|were)\b',
                    r'\bwhen\s+(is|are|was|were|did|does)\b',
                    r'\bwhere\s+(is|are|was|were)\b',
                    r'\bdefine\b', r'\bdefinition\b'
                ],
                'keywords': ['what', 'who', 'when', 'where', 'define', 'fact', 'information']
            },
            QueryType.ANALYTICAL: {
                'patterns': [
                    r'\banalyze\b', r'\banalysis\b', r'\bexamine\b',
                    r'\bevaluate\b', r'\bassess\b', r'\breview\b'
                ],
                'keywords': ['analyze', 'analysis', 'examine', 'evaluate', 'assess', 'insights', 'trends']
            },
            QueryType.COMPARATIVE: {
                'patterns': [
                    r'\bcompare\b', r'\bcomparison\b', r'\bversus\b', r'\bvs\b',
                    r'\bdifference\b', r'\bsimilar\b', r'\bbetter\b', r'\bworse\b'
                ],
                'keywords': ['compare', 'comparison', 'versus', 'difference', 'similar', 'better', 'worse']
            },
            QueryType.PROCEDURAL: {
                'patterns': [
                    r'\bhow\s+to\b', r'\bsteps\b', r'\bprocess\b',
                    r'\bprocedure\b', r'\bmethod\b', r'\bway\s+to\b'
                ],
                'keywords': ['how', 'steps', 'process', 'procedure', 'method', 'guide', 'tutorial']
            },
            QueryType.TEMPORAL: {
                'patterns': [
                    r'\b(last|past|recent|current|future|next)\b',
                    r'\b(today|yesterday|tomorrow)\b',
                    r'\b(week|month|year|quarter)\b',
                    r'\b(since|until|before|after|during)\b'
                ],
                'keywords': ['last', 'recent', 'current', 'future', 'trend', 'history', 'timeline']
            },
            QueryType.CAUSAL: {
                'patterns': [
                    r'\bwhy\b', r'\bcause\b', r'\breason\b',
                    r'\bimpact\b', r'\beffect\b', r'\bresult\b'
                ],
                'keywords': ['why', 'cause', 'reason', 'impact', 'effect', 'result', 'because']
            },
            QueryType.EXPLORATORY: {
                'patterns': [
                    r'\bexplore\b', r'\binvestigate\b', r'\bdiscover\b',
                    r'\bfind\s+out\b', r'\blearn\s+about\b'
                ],
                'keywords': ['explore', 'investigate', 'discover', 'research', 'learn', 'understand']
            },
            QueryType.DEFINITIONAL: {
                'patterns': [
                    r'\bdefine\b', r'\bdefinition\b', r'\bmeaning\b',
                    r'\bwhat\s+does\s+.+\s+mean\b', r'\bexplain\b'
                ],
                'keywords': ['define', 'definition', 'meaning', 'explain', 'concept', 'term']
            }
        }
        
        self.intent_patterns = {
            QueryIntent.INFORMATION_SEEKING: {
                'keywords': ['information', 'data', 'facts', 'details', 'learn', 'know'],
                'patterns': [r'\b(what|who|when|where|which)\b']
            },
            QueryIntent.PROBLEM_SOLVING: {
                'keywords': ['solve', 'fix', 'resolve', 'issue', 'problem', 'error'],
                'patterns': [r'\b(how\s+to\s+fix|solve|resolve)\b']
            },
            QueryIntent.DECISION_MAKING: {
                'keywords': ['decide', 'choose', 'select', 'option', 'recommendation', 'should'],
                'patterns': [r'\b(should\s+i|which\s+is\s+better|recommend)\b']
            },
            QueryIntent.MONITORING: {
                'keywords': ['status', 'progress', 'update', 'current', 'latest', 'monitor'],
                'patterns': [r'\b(current\s+status|latest|progress)\b']
            },
            QueryIntent.OPTIMIZATION: {
                'keywords': ['optimize', 'improve', 'enhance', 'increase', 'reduce', 'efficiency'],
                'patterns': [r'\b(how\s+to\s+improve|optimize|enhance)\b']
            },
            QueryIntent.PLANNING: {
                'keywords': ['plan', 'strategy', 'roadmap', 'schedule', 'future', 'next'],
                'patterns': [r'\b(plan\s+for|strategy|roadmap)\b']
            },
            QueryIntent.ANALYSIS: {
                'keywords': ['analyze', 'analysis', 'examine', 'evaluate', 'insights', 'trends'],
                'patterns': [r'\b(analyze|analysis|examine)\b']
            },
            QueryIntent.RESEARCH: {
                'keywords': ['research', 'investigate', 'study', 'explore', 'survey'],
                'patterns': [r'\b(research|investigate|study)\b']
            }
        }
    
    def extract_entities(self, query: str) -> List[str]:
        """Extract potential entities from query using simple heuristics"""
        # Extract capitalized words (potential proper nouns)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        
        # Extract quoted terms
        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)
        
        # Extract technical terms (words with numbers, hyphens, or special chars)
        technical = re.findall(r'\b\w*(?:\d+\w*|\w*-\w*)\w*\b', query)
        entities.extend([term for term in technical if len(term) > 2])
        
        return list(set(entities))
    
    def extract_keywords(self, query: str, min_length: int = 3) -> List[str]:
        """Extract meaningful keywords from query"""
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
            'could', 'can', 'may', 'might', 'must', 'shall', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words 
                   if len(word) >= min_length and word not in stop_words]
        
        return keywords
    
    def extract_temporal_indicators(self, query: str) -> List[str]:
        """Extract temporal indicators from query"""
        temporal_patterns = [
            r'\b(yesterday|today|tomorrow)\b',
            r'\b(last|this|next)\s+(week|month|year|quarter)\b',
            r'\b(since|until|before|after|during)\s+\w+\b',
            r'\b\d{4}\b',  # Years
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b(recent|current|latest|upcoming|future|past)\b'
        ]
        
        indicators = []
        for pattern in temporal_patterns:
            matches = re.findall(pattern, query.lower())
            indicators.extend(matches if isinstance(matches[0], str) if matches else [] 
                            for matches in [re.findall(pattern, query.lower())])
        
        return list(set(indicators))
    
    def classify_query(self, query: str) -> QueryAnalysis:
        """Comprehensive query classification"""
        query_lower = query.lower()
        
        # Classify query type
        type_scores = {}
        for query_type, indicators in self.type_patterns.items():
            score = 0.0
            
            # Pattern matching
            for pattern in indicators['patterns']:
                if re.search(pattern, query_lower):
                    score += 0.4
            
            # Keyword matching
            for keyword in indicators['keywords']:
                if keyword in query_lower:
                    score += 0.2
            
            type_scores[query_type] = score
        
        best_type = max(type_scores.keys(), key=lambda k: type_scores[k])
        type_confidence = min(type_scores[best_type], 1.0)
        
        # Classify intent
        intent_scores = {}
        for intent, indicators in self.intent_patterns.items():
            score = 0.0
            
            # Pattern matching
            for pattern in indicators['patterns']:
                if re.search(pattern, query_lower):
                    score += 0.5
            
            # Keyword matching
            for keyword in indicators['keywords']:
                if keyword in query_lower:
                    score += 0.3
            
            intent_scores[intent] = score
        
        best_intent = max(intent_scores.keys(), key=lambda k: intent_scores[k])
        intent_confidence = min(intent_scores[best_intent], 1.0)
        
        # Overall confidence is average of type and intent confidence
        overall_confidence = (type_confidence + intent_confidence) / 2
        
        # Calculate complexity score
        complexity_indicators = {
            'multiple_clauses': len(re.findall(r'[,;]', query)) * 0.1,
            'question_words': len(re.findall(r'\b(what|how|why|when|where|who|which)\b', query_lower)) * 0.15,
            'conditional_words': len(re.findall(r'\b(if|unless|when|while|although)\b', query_lower)) * 0.2,
            'complex_verbs': len(re.findall(r'\b(analyze|evaluate|compare|optimize|synthesize)\b', query_lower)) * 0.25,
            'length_factor': min(len(query.split()) / 20, 0.3)  # Longer queries tend to be more complex
        }
        complexity_score = min(sum(complexity_indicators.values()), 1.0)
        
        return QueryAnalysis(
            original_query=query,
            query_type=best_type,
            intent=best_intent,
            confidence=overall_confidence,
            entities=self.extract_entities(query),
            keywords=self.extract_keywords(query),
            temporal_indicators=self.extract_temporal_indicators(query),
            complexity_score=complexity_score,
            domain_indicators=self._extract_domain_indicators(query),
            question_words=re.findall(r'\b(what|how|why|when|where|who|which)\b', query_lower),
            metadata={
                'type_scores': {t.value: s for t, s in type_scores.items()},
                'intent_scores': {i.value: s for i, s in intent_scores.items()},
                'complexity_breakdown': complexity_indicators
            }
        )
    
    def _extract_domain_indicators(self, query: str) -> List[str]:
        """Extract domain-specific indicators"""
        domain_keywords = {
            'business': ['revenue', 'profit', 'sales', 'marketing', 'customer', 'roi', 'kpi'],
            'technical': ['api', 'database', 'server', 'code', 'algorithm', 'system'],
            'financial': ['budget', 'cost', 'expense', 'investment', 'finance', 'accounting'],
            'operations': ['process', 'workflow', 'efficiency', 'performance', 'productivity'],
            'hr': ['employee', 'staff', 'hiring', 'training', 'performance review'],
            'legal': ['compliance', 'regulation', 'policy', 'contract', 'agreement'],
            'healthcare': ['patient', 'medical', 'treatment', 'diagnosis', 'clinical'],
            'education': ['student', 'course', 'curriculum', 'learning', 'assessment']
        }
        
        query_lower = query.lower()
        detected_domains = []
        
        for domain, keywords in domain_keywords.items():
            domain_score = sum(1 for keyword in keywords if keyword in query_lower)
            if domain_score >= 1:  # At least one keyword match
                detected_domains.append(f"{domain}:{domain_score}")
        
        return detected_domains


class QueryExpander:
    """Advanced query expansion with multiple strategies"""
    
    def __init__(self):
        # Synonym mappings for common business terms
        self.synonym_mappings = {
            'revenue': ['income', 'sales', 'earnings', 'turnover'],
            'profit': ['earnings', 'gain', 'margin', 'return'],
            'customer': ['client', 'user', 'buyer', 'consumer'],
            'employee': ['staff', 'worker', 'personnel', 'team member'],
            'analyze': ['examine', 'review', 'assess', 'evaluate'],
            'improve': ['enhance', 'optimize', 'upgrade', 'better'],
            'issue': ['problem', 'challenge', 'difficulty', 'concern'],
            'data': ['information', 'statistics', 'metrics', 'figures'],
            'strategy': ['plan', 'approach', 'method', 'framework'],
            'performance': ['results', 'outcomes', 'effectiveness', 'efficiency']
        }
        
        # Domain-specific term mappings
        self.domain_expansions = {
            'business': {
                'kpi': ['key performance indicator', 'metric', 'measurement'],
                'roi': ['return on investment', 'profitability', 'value'],
                'crm': ['customer relationship management', 'customer system'],
                'b2b': ['business to business', 'enterprise'],
                'b2c': ['business to consumer', 'retail']
            },
            'technical': {
                'api': ['application programming interface', 'service', 'endpoint'],
                'ui': ['user interface', 'frontend', 'interface'],
                'ux': ['user experience', 'usability', 'design'],
                'db': ['database', 'data storage', 'repository'],
                'ml': ['machine learning', 'artificial intelligence', 'ai']
            },
            'operations': {
                'sla': ['service level agreement', 'performance standard'],
                'kpi': ['key performance indicator', 'success metric'],
                'bottleneck': ['constraint', 'limitation', 'blocking issue'],
                'workflow': ['process', 'procedure', 'system']
            }
        }
        
        # Hierarchical relationships (parent -> children)
        self.hierarchical_terms = {
            'marketing': ['advertising', 'promotion', 'branding', 'campaigns'],
            'sales': ['leads', 'prospects', 'conversion', 'deals'],
            'finance': ['accounting', 'budgeting', 'forecasting', 'reporting'],
            'operations': ['logistics', 'supply chain', 'procurement', 'inventory'],
            'technology': ['software', 'hardware', 'infrastructure', 'systems'],
            'human resources': ['recruiting', 'training', 'performance', 'benefits']
        }
    
    def expand_with_synonyms(self, query: str, analysis: QueryAnalysis) -> List[str]:
        """Expand query with synonyms for key terms"""
        expansions = []
        query_words = set(word.lower() for word in query.split())
        
        for original_term, synonyms in self.synonym_mappings.items():
            if original_term in query_words:
                for synonym in synonyms:
                    expanded_query = query.lower().replace(original_term, synonym)
                    if expanded_query != query.lower():
                        expansions.append(expanded_query)
        
        return expansions[:5]  # Limit to top 5 synonym expansions
    
    def expand_with_domain_terms(self, query: str, analysis: QueryAnalysis) -> List[str]:
        """Expand query with domain-specific terms"""
        expansions = []
        query_lower = query.lower()
        
        # Detect relevant domains from analysis
        relevant_domains = []
        for domain_info in analysis.domain_indicators:
            domain_name = domain_info.split(':')[0]
            relevant_domains.append(domain_name)
        
        # If no specific domain detected, use general business terms
        if not relevant_domains:
            relevant_domains = ['business', 'technical']
        
        for domain in relevant_domains:
            if domain in self.domain_expansions:
                domain_terms = self.domain_expansions[domain]
                for short_term, expanded_terms in domain_terms.items():
                    if short_term in query_lower:
                        for expanded_term in expanded_terms:
                            expanded_query = query_lower.replace(short_term, expanded_term)
                            if expanded_query != query_lower:
                                expansions.append(expanded_query)
        
        return expansions[:5]  # Limit expansions
    
    def expand_with_hierarchical_terms(self, query: str, analysis: QueryAnalysis) -> List[str]:
        """Expand query with hierarchically related terms"""
        expansions = []
        query_lower = query.lower()
        
        for parent_term, child_terms in self.hierarchical_terms.items():
            if parent_term in query_lower:
                # Add child terms
                for child_term in child_terms:
                    expanded_query = f"{query} {child_term}"
                    expansions.append(expanded_query)
            
            # Also check if any child terms are present to add parent
            for child_term in child_terms:
                if child_term in query_lower and parent_term not in query_lower:
                    expanded_query = f"{query} {parent_term}"
                    expansions.append(expanded_query)
                    break
        
        return expansions[:3]  # Limit hierarchical expansions
    
    def expand_with_temporal_context(self, query: str, analysis: QueryAnalysis) -> List[str]:
        """Add temporal context to queries when appropriate"""
        if not analysis.temporal_indicators:
            return []
        
        expansions = []
        current_year = datetime.now().year
        
        # Add current timeframe context
        temporal_variants = [
            f"{query} current year",
            f"{query} {current_year}",
            f"{query} recent",
            f"{query} latest"
        ]
        
        # Add specific temporal expansions based on detected indicators
        for indicator in analysis.temporal_indicators:
            if 'recent' in indicator or 'current' in indicator:
                expansions.extend([
                    f"{query} past 3 months",
                    f"{query} year to date"
                ])
                break
        
        return expansions[:3]
    
    def generate_semantic_variations(self, query: str, analysis: QueryAnalysis) -> List[str]:
        """Generate semantic variations based on query type and intent"""
        variations = []
        
        # Question reformulation based on type
        if analysis.query_type == QueryType.FACTUAL:
            if query.lower().startswith('what is'):
                variations.append(query.replace('what is', 'define'))
                variations.append(query.replace('what is', 'explain'))
        
        elif analysis.query_type == QueryType.ANALYTICAL:
            if 'analyze' in query.lower():
                variations.append(query.replace('analyze', 'examine'))
                variations.append(query.replace('analyze', 'evaluate'))
        
        elif analysis.query_type == QueryType.PROCEDURAL:
            if 'how to' in query.lower():
                variations.append(query.replace('how to', 'steps to'))
                variations.append(query.replace('how to', 'process for'))
        
        # Intent-based variations
        if analysis.intent == QueryIntent.PROBLEM_SOLVING:
            if 'problem' in query.lower():
                variations.append(query.replace('problem', 'issue'))
                variations.append(query.replace('problem', 'challenge'))
        
        elif analysis.intent == QueryIntent.OPTIMIZATION:
            if 'improve' in query.lower():
                variations.append(query.replace('improve', 'optimize'))
                variations.append(query.replace('improve', 'enhance'))
        
        return variations[:4]
    
    def expand_query(self, query: str, analysis: QueryAnalysis, 
                    strategies: List[ExpansionStrategy] = None) -> ExpandedQuery:
        """Generate comprehensive query expansion"""
        
        if strategies is None:
            strategies = [
                ExpansionStrategy.SYNONYM_BASED,
                ExpansionStrategy.DOMAIN_SPECIFIC,
                ExpansionStrategy.SEMANTIC_SIMILARITY,
                ExpansionStrategy.HIERARCHICAL
            ]
        
        all_expansions = []
        expansion_details = {}
        
        # Apply different expansion strategies
        if ExpansionStrategy.SYNONYM_BASED in strategies:
            synonym_expansions = self.expand_with_synonyms(query, analysis)
            all_expansions.extend(synonym_expansions)
            expansion_details['synonym_expansions'] = synonym_expansions
        
        if ExpansionStrategy.DOMAIN_SPECIFIC in strategies:
            domain_expansions = self.expand_with_domain_terms(query, analysis)
            all_expansions.extend(domain_expansions)
            expansion_details['domain_expansions'] = domain_expansions
        
        if ExpansionStrategy.HIERARCHICAL in strategies:
            hierarchical_expansions = self.expand_with_hierarchical_terms(query, analysis)
            all_expansions.extend(hierarchical_expansions)
            expansion_details['hierarchical_expansions'] = hierarchical_expansions
        
        if ExpansionStrategy.SEMANTIC_SIMILARITY in strategies:
            semantic_variations = self.generate_semantic_variations(query, analysis)
            all_expansions.extend(semantic_variations)
            expansion_details['semantic_variations'] = semantic_variations
        
        if ExpansionStrategy.TEMPORAL in strategies:
            temporal_expansions = self.expand_with_temporal_context(query, analysis)
            all_expansions.extend(temporal_expansions)
            expansion_details['temporal_expansions'] = temporal_expansions
        
        # Extract unique expanded terms
        original_words = set(query.lower().split())
        expanded_terms = set()
        
        for expansion in all_expansions:
            expansion_words = set(expansion.lower().split())
            new_terms = expansion_words - original_words
            expanded_terms.update(new_terms)
        
        # Create combined query with most relevant expansions
        key_expansions = []
        if expansion_details.get('synonym_expansions'):
            key_expansions.extend(expansion_details['synonym_expansions'][:2])
        if expansion_details.get('domain_expansions'):
            key_expansions.extend(expansion_details['domain_expansions'][:2])
        
        # Combine original query with key expanded terms
        combined_terms = list(expanded_terms)[:8]  # Limit to prevent query bloat
        combined_query = f"{query} {' '.join(combined_terms)}" if combined_terms else query
        
        return ExpandedQuery(
            original_query=query,
            expanded_terms=list(expanded_terms),
            synonym_expansions=expansion_details.get('synonym_expansions', []),
            domain_expansions=expansion_details.get('domain_expansions', []),
            semantic_variations=expansion_details.get('semantic_variations', []),
            combined_query=combined_query,
            expansion_metadata={
                'strategies_used': [s.value for s in strategies],
                'total_expansions': len(all_expansions),
                'unique_terms_added': len(expanded_terms),
                'expansion_details': expansion_details
            }
        )


class AgentQueryRouter:
    """Routes and optimizes queries for specific AI agents"""
    
    def __init__(self):
        # Agent specialization mappings
        self.agent_specializations = {
            'research': {
                'keywords': ['research', 'study', 'investigate', 'analyze', 'data', 'findings'],
                'query_types': [QueryType.ANALYTICAL, QueryType.EXPLORATORY],
                'intents': [QueryIntent.RESEARCH, QueryIntent.ANALYSIS],
                'optimization_hints': ['detailed analysis', 'comprehensive data', 'research findings']
            },
            'ceo': {
                'keywords': ['strategy', 'leadership', 'vision', 'decision', 'business', 'executive'],
                'query_types': [QueryType.ANALYTICAL, QueryType.COMPARATIVE],
                'intents': [QueryIntent.DECISION_MAKING, QueryIntent.PLANNING],
                'optimization_hints': ['strategic perspective', 'executive summary', 'business impact']
            },
            'performance': {
                'keywords': ['performance', 'metrics', 'kpi', 'results', 'efficiency', 'productivity'],
                'query_types': [QueryType.FACTUAL, QueryType.ANALYTICAL],
                'intents': [QueryIntent.MONITORING, QueryIntent.OPTIMIZATION],
                'optimization_hints': ['performance metrics', 'quantitative data', 'benchmarks']
            },
            'coaching': {
                'keywords': ['coaching', 'development', 'training', 'improvement', 'skills', 'growth'],
                'query_types': [QueryType.PROCEDURAL, QueryType.ANALYTICAL],
                'intents': [QueryIntent.PROBLEM_SOLVING, QueryIntent.OPTIMIZATION],
                'optimization_hints': ['development guidance', 'best practices', 'skill improvement']
            },
            'business_intelligence': {
                'keywords': ['intelligence', 'insights', 'trends', 'forecasting', 'analytics', 'reporting'],
                'query_types': [QueryType.ANALYTICAL, QueryType.TEMPORAL],
                'intents': [QueryIntent.ANALYSIS, QueryIntent.MONITORING],
                'optimization_hints': ['business intelligence', 'trend analysis', 'data insights']
            },
            'contact_center': {
                'keywords': ['customer', 'service', 'support', 'contact', 'calls', 'satisfaction'],
                'query_types': [QueryType.PROCEDURAL, QueryType.ANALYTICAL],
                'intents': [QueryIntent.PROBLEM_SOLVING, QueryIntent.OPTIMIZATION],
                'optimization_hints': ['customer service', 'support processes', 'service quality']
            }
        }
    
    def calculate_agent_affinity(self, analysis: QueryAnalysis, agent_name: str) -> float:
        """Calculate how well a query matches an agent's specialization"""
        if agent_name not in self.agent_specializations:
            return 0.0
        
        spec = self.agent_specializations[agent_name]
        affinity_score = 0.0
        
        # Keyword matching
        query_words = set(analysis.original_query.lower().split())
        keyword_matches = len(query_words.intersection(set(spec['keywords'])))
        affinity_score += keyword_matches * 0.3
        
        # Query type matching
        if analysis.query_type in spec['query_types']:
            affinity_score += 0.4
        
        # Intent matching
        if analysis.intent in spec['intents']:
            affinity_score += 0.3
        
        return min(affinity_score, 1.0)
    
    def route_query(self, analysis: QueryAnalysis, 
                   available_agents: List[str] = None) -> List[QueryRoute]:
        """Route query to most appropriate agents"""
        if available_agents is None:
            available_agents = list(self.agent_specializations.keys())
        
        routes = []
        
        for agent_name in available_agents:
            affinity = self.calculate_agent_affinity(analysis, agent_name)
            
            if affinity > 0.3:  # Minimum threshold for routing
                # Optimize query for this agent
                spec = self.agent_specializations[agent_name]
                optimized_query = self._optimize_query_for_agent(analysis.original_query, spec)
                
                route = QueryRoute(
                    target_agent=agent_name,
                    optimized_query=optimized_query,
                    context_hints=spec['optimization_hints'],
                    retrieval_strategy='specialized',
                    confidence=affinity
                )
                routes.append(route)
        
        # Sort by confidence
        routes.sort(key=lambda r: r.confidence, reverse=True)
        
        return routes[:3]  # Return top 3 routes
    
    def _optimize_query_for_agent(self, query: str, agent_spec: Dict[str, Any]) -> str:
        """Optimize query for specific agent specialization"""
        # Add context hints to query
        context_hint = agent_spec['optimization_hints'][0]  # Use primary hint
        
        # Simple optimization: prepend context hint if not already present
        if context_hint.split()[0].lower() not in query.lower():
            optimized_query = f"{context_hint}: {query}"
        else:
            optimized_query = query
        
        return optimized_query


class QueryIntelligenceManager:
    """Main manager for query intelligence and expansion"""
    
    def __init__(self):
        self.classifier = QueryClassifier()
        self.expander = QueryExpander()
        self.router = AgentQueryRouter()
        self.query_history = []
        
    def process_query(self, query: str, 
                     target_agents: List[str] = None,
                     expansion_strategies: List[ExpansionStrategy] = None) -> Dict[str, Any]:
        """Comprehensive query processing pipeline"""
        
        # Step 1: Analyze query
        analysis = self.classifier.classify_query(query)
        
        # Step 2: Expand query
        expanded_query = self.expander.expand_query(query, analysis, expansion_strategies)
        
        # Step 3: Route to appropriate agents
        routes = self.router.route_query(analysis, target_agents)
        
        # Step 4: Log for learning
        self.query_history.append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'analysis': analysis,
            'expansion': expanded_query,
            'routes': routes
        })
        
        # Keep history manageable
        if len(self.query_history) > 1000:
            self.query_history = self.query_history[-800:]
        
        return {
            'original_query': query,
            'analysis': analysis,
            'expanded_query': expanded_query,
            'agent_routes': routes,
            'processing_metadata': {
                'timestamp': datetime.now().isoformat(),
                'confidence': analysis.confidence,
                'complexity': analysis.complexity_score,
                'expansion_count': len(expanded_query.expanded_terms)
            }
        }
    
    def get_query_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """Generate query suggestions based on history and patterns"""
        if len(partial_query) < 3:
            return []
        
        suggestions = []
        partial_lower = partial_query.lower()
        
        # Look for similar queries in history
        for entry in self.query_history[-100:]:  # Check recent queries
            if partial_lower in entry['query'].lower():
                suggestions.append(entry['query'])
        
        # Add template-based suggestions based on detected pattern
        if 'how to' in partial_lower:
            suggestions.extend([
                f"{partial_query} improve efficiency",
                f"{partial_query} optimize process",
                f"{partial_query} best practices"
            ])
        elif 'what is' in partial_lower:
            suggestions.extend([
                f"{partial_query} definition",
                f"{partial_query} impact",
                f"{partial_query} benefits"
            ])
        
        # Remove duplicates and limit
        unique_suggestions = list(dict.fromkeys(suggestions))
        return unique_suggestions[:limit]
    
    def get_expansion_preview(self, query: str) -> Dict[str, List[str]]:
        """Get a preview of how the query would be expanded"""
        analysis = self.classifier.classify_query(query)
        expanded = self.expander.expand_query(query, analysis)
        
        return {
            'synonym_expansions': expanded.synonym_expansions[:3],
            'domain_expansions': expanded.domain_expansions[:3],
            'semantic_variations': expanded.semantic_variations[:3],
            'key_terms_added': expanded.expanded_terms[:5]
        }