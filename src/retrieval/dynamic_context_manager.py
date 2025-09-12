"""
Dynamic Context Management System for RAG Applications

This module provides intelligent context window management based on:
- Query complexity analysis
- Available token budget
- Document relevance scoring
- Hierarchical context compression
"""

import logging
import tiktoken
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
from collections import Counter
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """Query complexity classification"""
    SIMPLE = "simple"           # Single concept, direct facts
    MODERATE = "moderate"       # Multiple concepts, some reasoning
    COMPLEX = "complex"         # Multi-step reasoning, analysis required
    ANALYTICAL = "analytical"   # Deep analysis, synthesis required


class ContextCompressionLevel(Enum):
    """Context compression strategies"""
    NONE = "none"               # Full content
    LIGHT = "light"             # Remove redundancy, keep details
    MODERATE = "moderate"       # Summarize sections, keep key points
    AGGRESSIVE = "aggressive"   # Extract key facts only


@dataclass
class ContextBudget:
    """Token budget allocation for different context components"""
    total_tokens: int = 8192
    system_prompt_tokens: int = 500
    query_tokens: int = 100
    response_reserve_tokens: int = 1000
    available_context_tokens: int = 0
    
    def __post_init__(self):
        self.available_context_tokens = (
            self.total_tokens - 
            self.system_prompt_tokens - 
            self.query_tokens - 
            self.response_reserve_tokens
        )


@dataclass
class ContextDocument:
    """Enhanced document representation with context metadata"""
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    token_count: int
    summary: Optional[str] = None
    key_points: Optional[List[str]] = None
    compression_ratio: float = 1.0
    importance_score: float = 0.0


@dataclass
class ContextWindow:
    """Optimized context window with hierarchical information"""
    documents: List[ContextDocument]
    total_tokens: int
    compression_level: ContextCompressionLevel
    coverage_score: float
    diversity_score: float
    metadata: Dict[str, Any]


class QueryComplexityAnalyzer:
    """Analyzes query complexity to determine optimal context strategy"""
    
    def __init__(self):
        self.complexity_indicators = {
            QueryComplexity.SIMPLE: {
                'keywords': ['what is', 'who is', 'when', 'where', 'define', 'meaning'],
                'patterns': [r'\bwhat\s+is\b', r'\bwho\s+is\b', r'\bdefine\b'],
                'max_concepts': 1,
                'question_words': 1
            },
            QueryComplexity.MODERATE: {
                'keywords': ['how', 'why', 'compare', 'difference', 'relationship', 'impact'],
                'patterns': [r'\bhow\s+(does|do|can)\b', r'\bwhy\b', r'\bcompare\b'],
                'max_concepts': 3,
                'question_words': 2
            },
            QueryComplexity.COMPLEX: {
                'keywords': ['analyze', 'evaluate', 'assess', 'strategy', 'optimize', 'implications'],
                'patterns': [r'\banalyze\b', r'\bevaluate\b', r'\bstrategy\b'],
                'max_concepts': 5,
                'question_words': 3
            },
            QueryComplexity.ANALYTICAL: {
                'keywords': ['synthesize', 'correlation', 'trends', 'patterns', 'insights', 'recommendations'],
                'patterns': [r'\bsynthesize\b', r'\btrends?\b', r'\binsights?\b'],
                'max_concepts': 10,
                'question_words': 5
            }
        }
        
    def analyze_complexity(self, query: str) -> Tuple[QueryComplexity, float]:
        """Analyze query complexity and return classification with confidence"""
        query_lower = query.lower()
        complexity_scores = {}
        
        for complexity, indicators in self.complexity_indicators.items():
            score = 0.0
            
            # Keyword matching
            keyword_matches = sum(1 for kw in indicators['keywords'] if kw in query_lower)
            score += keyword_matches * 0.3
            
            # Pattern matching
            pattern_matches = sum(1 for pattern in indicators['patterns'] 
                                if re.search(pattern, query_lower))
            score += pattern_matches * 0.4
            
            # Concept density (unique nouns/total words)
            words = query_lower.split()
            concept_density = len(set(words)) / len(words) if words else 0
            if concept_density >= indicators['max_concepts'] / 10:
                score += 0.2
            
            # Question complexity (multiple question words)
            question_words = len([w for w in words if w in ['what', 'how', 'why', 'when', 'where', 'who']])
            if question_words >= indicators['question_words']:
                score += 0.1
            
            complexity_scores[complexity] = score
        
        # Get the complexity with highest score
        best_complexity = max(complexity_scores.keys(), key=lambda k: complexity_scores[k])
        confidence = complexity_scores[best_complexity]
        
        return best_complexity, min(confidence, 1.0)


class ContextCompressor:
    """Compresses context using various strategies"""
    
    def __init__(self, tokenizer_name: str = "cl100k_base"):
        try:
            self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer {tokenizer_name}: {e}")
            self.tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: rough estimation (4 chars per token)
            return len(text) // 4
    
    def extract_key_points(self, content: str, max_points: int = 5) -> List[str]:
        """Extract key points from content using simple heuristics"""
        sentences = re.split(r'[.!?]+', content)
        
        # Score sentences by various factors
        scored_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Too short
                continue
                
            score = 0.0
            
            # Length penalty for very long sentences
            if len(sentence) > 200:
                score -= 0.2
            
            # Boost for numbers and statistics
            if re.search(r'\d+(?:\.\d+)?[%$]?', sentence):
                score += 0.3
            
            # Boost for key indicator words
            key_indicators = ['important', 'significant', 'key', 'main', 'primary', 'critical']
            score += sum(0.1 for indicator in key_indicators if indicator in sentence.lower())
            
            # Boost for sentences with strong verbs
            strong_verbs = ['shows', 'demonstrates', 'indicates', 'reveals', 'proves']
            score += sum(0.1 for verb in strong_verbs if verb in sentence.lower())
            
            scored_sentences.append((sentence, score))
        
        # Sort by score and return top points
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        return [sentence for sentence, _ in scored_sentences[:max_points]]
    
    def create_summary(self, content: str, target_ratio: float = 0.3) -> str:
        """Create a summary of the content with target compression ratio"""
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return content
        
        target_sentences = max(1, int(len(sentences) * target_ratio))
        
        # Use key points extraction for summary
        key_points = self.extract_key_points(content, target_sentences)
        
        return '. '.join(key_points) + '.'
    
    def compress_document(self, 
                         document: ContextDocument, 
                         compression_level: ContextCompressionLevel,
                         target_tokens: Optional[int] = None) -> ContextDocument:
        """Compress a document based on specified compression level"""
        
        if compression_level == ContextCompressionLevel.NONE:
            return document
        
        original_tokens = document.token_count
        compressed_content = document.content
        compression_ratio = 1.0
        
        if compression_level == ContextCompressionLevel.LIGHT:
            # Remove redundant whitespace and minor cleanup
            compressed_content = re.sub(r'\s+', ' ', compressed_content.strip())
            compression_ratio = 0.9
            
        elif compression_level == ContextCompressionLevel.MODERATE:
            # Create summary while preserving key information
            compressed_content = self.create_summary(document.content, target_ratio=0.5)
            compression_ratio = 0.5
            
        elif compression_level == ContextCompressionLevel.AGGRESSIVE:
            # Extract only key facts and figures
            key_points = self.extract_key_points(document.content, max_points=3)
            compressed_content = '. '.join(key_points)
            compression_ratio = 0.2
        
        # Apply target token limit if specified
        if target_tokens:
            current_tokens = self.count_tokens(compressed_content)
            if current_tokens > target_tokens:
                # Further compress by truncating
                char_ratio = target_tokens / current_tokens
                target_chars = int(len(compressed_content) * char_ratio * 0.9)  # Safety margin
                compressed_content = compressed_content[:target_chars] + "..."
                compression_ratio *= char_ratio
        
        # Create compressed document
        compressed_doc = ContextDocument(
            content=compressed_content,
            metadata=document.metadata.copy(),
            relevance_score=document.relevance_score,
            token_count=self.count_tokens(compressed_content),
            summary=document.summary,
            key_points=self.extract_key_points(compressed_content, 3),
            compression_ratio=compression_ratio,
            importance_score=document.importance_score
        )
        
        return compressed_doc


class DynamicContextManager:
    """Main context management system with dynamic optimization"""
    
    def __init__(self, 
                 default_budget: Optional[ContextBudget] = None,
                 enable_caching: bool = True):
        self.complexity_analyzer = QueryComplexityAnalyzer()
        self.compressor = ContextCompressor()
        self.default_budget = default_budget or ContextBudget()
        self.enable_caching = enable_caching
        self.context_cache = {} if enable_caching else None
        
        # Context strategy mappings
        self.complexity_strategies = {
            QueryComplexity.SIMPLE: {
                'max_documents': 3,
                'preferred_compression': ContextCompressionLevel.LIGHT,
                'diversity_weight': 0.2,
                'relevance_weight': 0.8
            },
            QueryComplexity.MODERATE: {
                'max_documents': 5,
                'preferred_compression': ContextCompressionLevel.LIGHT,
                'diversity_weight': 0.3,
                'relevance_weight': 0.7
            },
            QueryComplexity.COMPLEX: {
                'max_documents': 8,
                'preferred_compression': ContextCompressionLevel.MODERATE,
                'diversity_weight': 0.4,
                'relevance_weight': 0.6
            },
            QueryComplexity.ANALYTICAL: {
                'max_documents': 12,
                'preferred_compression': ContextCompressionLevel.MODERATE,
                'diversity_weight': 0.5,
                'relevance_weight': 0.5
            }
        }
    
    def create_context_documents(self, 
                               search_results: List[Dict[str, Any]]) -> List[ContextDocument]:
        """Convert search results to ContextDocument objects"""
        context_docs = []
        
        for result in search_results:
            content = result.get('content', '')
            metadata = result.get('metadata', {})
            relevance_score = result.get('relevance_score', 0.0)
            
            # Calculate importance score based on multiple factors
            importance_score = relevance_score
            
            # Boost for recent documents
            if 'timestamp' in metadata:
                try:
                    doc_date = datetime.fromisoformat(metadata['timestamp'].replace('Z', '+00:00'))
                    days_old = (datetime.now() - doc_date.replace(tzinfo=None)).days
                    recency_boost = max(0, 1.0 - (days_old / 365))  # Decay over a year
                    importance_score += recency_boost * 0.1
                except Exception:
                    pass
            
            # Boost for documents with rich metadata
            if len(metadata) > 3:
                importance_score += 0.05
            
            doc = ContextDocument(
                content=content,
                metadata=metadata,
                relevance_score=relevance_score,
                token_count=self.compressor.count_tokens(content),
                importance_score=importance_score
            )
            
            context_docs.append(doc)
        
        return context_docs
    
    def select_documents_with_diversity(self, 
                                      documents: List[ContextDocument],
                                      max_documents: int,
                                      diversity_weight: float) -> List[ContextDocument]:
        """Select documents balancing relevance and diversity using MMR-like approach"""
        if len(documents) <= max_documents:
            return documents
        
        selected = []
        remaining = documents.copy()
        
        # Always select the most relevant document first
        best_doc = max(remaining, key=lambda d: d.importance_score)
        selected.append(best_doc)
        remaining.remove(best_doc)
        
        # Select remaining documents using MMR-like scoring
        while len(selected) < max_documents and remaining:
            best_score = -1
            best_doc = None
            
            for candidate in remaining:
                # Relevance component
                relevance_score = candidate.importance_score
                
                # Diversity component (simple content overlap check)
                diversity_score = 1.0
                for selected_doc in selected:
                    # Simple content overlap estimation
                    candidate_words = set(candidate.content.lower().split())
                    selected_words = set(selected_doc.content.lower().split())
                    
                    if candidate_words and selected_words:
                        overlap = len(candidate_words & selected_words) / len(candidate_words | selected_words)
                        diversity_score = min(diversity_score, 1.0 - overlap)
                
                # Combined score
                combined_score = ((1 - diversity_weight) * relevance_score + 
                                diversity_weight * diversity_score)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_doc = candidate
            
            if best_doc:
                selected.append(best_doc)
                remaining.remove(best_doc)
            else:
                break
        
        return selected
    
    def optimize_context_window(self, 
                              query: str,
                              search_results: List[Dict[str, Any]],
                              budget: Optional[ContextBudget] = None) -> ContextWindow:
        """Create optimized context window for the given query and results"""
        
        # Use provided budget or default
        context_budget = budget or self.default_budget
        
        # Analyze query complexity
        complexity, confidence = self.complexity_analyzer.analyze_complexity(query)
        strategy = self.complexity_strategies[complexity]
        
        logger.info(f"Query complexity: {complexity.value} (confidence: {confidence:.2f})")
        
        # Create context documents
        context_docs = self.create_context_documents(search_results)
        
        if not context_docs:
            return ContextWindow(
                documents=[],
                total_tokens=0,
                compression_level=ContextCompressionLevel.NONE,
                coverage_score=0.0,
                diversity_score=0.0,
                metadata={'complexity': complexity.value, 'confidence': confidence}
            )
        
        # Select documents with diversity
        selected_docs = self.select_documents_with_diversity(
            context_docs,
            strategy['max_documents'],
            strategy['diversity_weight']
        )
        
        # Calculate total tokens needed
        total_tokens_needed = sum(doc.token_count for doc in selected_docs)
        
        # Determine compression level
        compression_level = strategy['preferred_compression']
        if total_tokens_needed > context_budget.available_context_tokens:
            # Need more aggressive compression
            if total_tokens_needed > context_budget.available_context_tokens * 2:
                compression_level = ContextCompressionLevel.AGGRESSIVE
            else:
                compression_level = ContextCompressionLevel.MODERATE
        
        # Apply compression
        compressed_docs = []
        remaining_budget = context_budget.available_context_tokens
        
        for doc in selected_docs:
            if remaining_budget <= 0:
                break
                
            # Allocate tokens proportionally based on importance
            total_importance = sum(d.importance_score for d in selected_docs)
            doc_proportion = doc.importance_score / total_importance if total_importance > 0 else 1.0 / len(selected_docs)
            target_tokens = int(context_budget.available_context_tokens * doc_proportion)
            target_tokens = min(target_tokens, remaining_budget)
            
            compressed_doc = self.compressor.compress_document(doc, compression_level, target_tokens)
            compressed_docs.append(compressed_doc)
            remaining_budget -= compressed_doc.token_count
        
        # Calculate quality metrics
        total_tokens = sum(doc.token_count for doc in compressed_docs)
        coverage_score = len(compressed_docs) / len(context_docs) if context_docs else 0.0
        
        # Calculate diversity score (simplified)
        diversity_score = min(1.0, len(compressed_docs) / strategy['max_documents'])
        
        context_window = ContextWindow(
            documents=compressed_docs,
            total_tokens=total_tokens,
            compression_level=compression_level,
            coverage_score=coverage_score,
            diversity_score=diversity_score,
            metadata={
                'complexity': complexity.value,
                'confidence': confidence,
                'original_docs': len(context_docs),
                'selected_docs': len(compressed_docs),
                'budget_used': total_tokens,
                'budget_available': context_budget.available_context_tokens,
                'compression_ratio': np.mean([doc.compression_ratio for doc in compressed_docs]) if compressed_docs else 1.0
            }
        )
        
        return context_window
    
    def format_context_for_llm(self, context_window: ContextWindow) -> str:
        """Format the context window for LLM consumption"""
        if not context_window.documents:
            return "No relevant context found."
        
        formatted_parts = []
        
        # Add summary if heavily compressed
        if context_window.compression_level in [ContextCompressionLevel.MODERATE, ContextCompressionLevel.AGGRESSIVE]:
            summary_info = (f"Context Summary: {len(context_window.documents)} documents, "
                          f"compression level: {context_window.compression_level.value}, "
                          f"coverage: {context_window.coverage_score:.1%}")
            formatted_parts.append(f"[{summary_info}]\n")
        
        # Format each document
        for i, doc in enumerate(context_window.documents, 1):
            doc_header = f"Document {i}"
            
            # Add metadata if available
            if doc.metadata:
                title = doc.metadata.get('title', '')
                source = doc.metadata.get('source', '')
                if title:
                    doc_header += f": {title}"
                if source:
                    doc_header += f" (Source: {source})"
            
            doc_header += f" [Relevance: {doc.relevance_score:.2f}]"
            
            formatted_parts.append(f"{doc_header}\n{doc.content}\n")
        
        return "\n".join(formatted_parts)
    
    def get_context_stats(self, context_window: ContextWindow) -> Dict[str, Any]:
        """Get detailed statistics about the context window"""
        if not context_window.documents:
            return {'total_documents': 0, 'total_tokens': 0}
        
        return {
            'total_documents': len(context_window.documents),
            'total_tokens': context_window.total_tokens,
            'compression_level': context_window.compression_level.value,
            'coverage_score': context_window.coverage_score,
            'diversity_score': context_window.diversity_score,
            'average_relevance': np.mean([doc.relevance_score for doc in context_window.documents]),
            'average_compression_ratio': np.mean([doc.compression_ratio for doc in context_window.documents]),
            'metadata': context_window.metadata
        }