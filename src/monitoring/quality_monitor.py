"""
Real-Time Quality Monitoring System

This module provides continuous monitoring of RAG system quality:
- Confidence scoring and tracking
- Hallucination detection
- User feedback integration
- Real-time alerts and notifications
- Quality trend analysis
- Automated quality assurance
"""

import logging
import json
import time
import threading
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
import statistics
from concurrent.futures import ThreadPoolExecutor
import uuid
import hashlib

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class QualityMetric(Enum):
    """Quality metrics to monitor"""
    CONFIDENCE_SCORE = "confidence_score"
    HALLUCINATION_RISK = "hallucination_risk"
    RESPONSE_RELEVANCE = "response_relevance"
    FACTUAL_CONSISTENCY = "factual_consistency"
    COMPLETENESS = "completeness"
    COHERENCE = "coherence"
    USER_SATISFACTION = "user_satisfaction"


class FeedbackType(Enum):
    """Types of user feedback"""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"
    EXPLICIT_FEEDBACK = "explicit_feedback"
    CORRECTION = "correction"


@dataclass
class QualityAlert:
    """Quality alert representation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    level: AlertLevel = AlertLevel.INFO
    metric: QualityMetric = QualityMetric.CONFIDENCE_SCORE
    message: str = ""
    value: float = 0.0
    threshold: float = 0.0
    query: str = ""
    response: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False


@dataclass
class UserFeedback:
    """User feedback on system responses"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    response: str = ""
    feedback_type: FeedbackType = FeedbackType.RATING
    rating: Optional[int] = None  # 1-5 scale
    feedback_text: Optional[str] = None
    correction: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityMetrics:
    """Real-time quality metrics"""
    timestamp: datetime
    confidence_score: float = 0.0
    hallucination_risk: float = 0.0
    response_relevance: float = 0.0
    factual_consistency: float = 0.0
    completeness: float = 0.0
    coherence: float = 0.0
    user_satisfaction: float = 0.0
    overall_quality: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringConfiguration:
    """Configuration for quality monitoring"""
    confidence_threshold: float = 0.7
    hallucination_threshold: float = 0.3
    relevance_threshold: float = 0.6
    consistency_threshold: float = 0.8
    user_satisfaction_threshold: float = 3.0
    
    # Alert configuration
    enable_alerts: bool = True
    alert_cooldown_minutes: int = 5
    max_alerts_per_hour: int = 10
    
    # Monitoring intervals
    quality_check_interval_seconds: int = 30
    trend_analysis_window_hours: int = 24
    
    # Feedback configuration
    feedback_timeout_hours: int = 72
    min_feedback_samples: int = 10


class ConfidenceScorer:
    """Calculates confidence scores for responses"""
    
    def __init__(self):
        self.semantic_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logger.warning(f"Failed to load semantic model: {e}")
    
    def calculate_confidence(self, 
                           query: str,
                           response: str,
                           retrieved_documents: List[Dict[str, Any]],
                           metadata: Dict[str, Any] = None) -> float:
        """Calculate overall confidence score for a response"""
        
        confidence_factors = []
        
        # 1. Semantic consistency between query and response
        if self.semantic_model:
            query_response_similarity = self._calculate_semantic_similarity(query, response)
            confidence_factors.append(('query_response_similarity', query_response_similarity, 0.25))
        
        # 2. Grounding in retrieved documents
        document_grounding = self._calculate_document_grounding(response, retrieved_documents)
        confidence_factors.append(('document_grounding', document_grounding, 0.35))
        
        # 3. Response completeness and specificity
        completeness_score = self._calculate_completeness(response, query)
        confidence_factors.append(('completeness', completeness_score, 0.15))
        
        # 4. Factual consistency indicators
        factual_score = self._calculate_factual_indicators(response, retrieved_documents)
        confidence_factors.append(('factual_consistency', factual_score, 0.15))
        
        # 5. Language quality and coherence
        coherence_score = self._calculate_coherence(response)
        confidence_factors.append(('coherence', coherence_score, 0.10))
        
        # Calculate weighted confidence
        total_confidence = sum(score * weight for _, score, weight in confidence_factors)
        
        return min(1.0, max(0.0, total_confidence))
    
    def _calculate_semantic_similarity(self, query: str, response: str) -> float:
        """Calculate semantic similarity between query and response"""
        if not self.semantic_model:
            # Fallback: simple word overlap
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())
            
            if not query_words:
                return 0.0
            
            overlap = len(query_words & response_words)
            return overlap / len(query_words)
        
        try:
            query_embedding = self.semantic_model.encode([query])
            response_embedding = self.semantic_model.encode([response])
            
            similarity = util.cos_sim(query_embedding, response_embedding)[0][0].item()
            return max(0.0, similarity)
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {e}")
            return 0.5  # Neutral score
    
    def _calculate_document_grounding(self, response: str, documents: List[Dict[str, Any]]) -> float:
        """Calculate how well response is grounded in documents"""
        if not documents:
            return 0.0
        
        # Combine document content
        combined_content = " ".join([
            doc.get('content', '') for doc in documents[:5]  # Top 5 documents
        ])
        
        if not combined_content.strip():
            return 0.0
        
        # Calculate overlap between response and documents
        response_words = set(response.lower().split())
        content_words = set(combined_content.lower().split())
        
        if not response_words:
            return 0.0
        
        overlap = len(response_words & content_words)
        grounding_score = overlap / len(response_words)
        
        # Bonus for citing specific facts/numbers
        import re
        response_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', response))
        content_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', combined_content))
        
        if response_numbers and content_numbers:
            number_overlap = len(response_numbers & content_numbers)
            number_bonus = min(0.2, number_overlap * 0.05)
            grounding_score += number_bonus
        
        return min(1.0, grounding_score)
    
    def _calculate_completeness(self, response: str, query: str) -> float:
        """Calculate response completeness relative to query"""
        
        # Response length appropriateness
        response_length = len(response.split())
        
        if response_length < 5:
            length_score = 0.2  # Too short
        elif response_length > 300:
            length_score = 0.8  # Might be too verbose
        else:
            # Optimal range: 10-200 words
            if 10 <= response_length <= 200:
                length_score = 1.0
            else:
                length_score = 0.7
        
        # Question coverage
        question_words = {'what', 'how', 'why', 'when', 'where', 'who', 'which'}
        query_questions = [w for w in query.lower().split() if w in question_words]
        
        coverage_score = 1.0
        if query_questions:
            # Check if response addresses the question type
            response_lower = response.lower()
            
            for q_word in query_questions:
                if q_word == 'what' and not any(word in response_lower for word in ['is', 'are', 'means', 'refers to']):
                    coverage_score *= 0.8
                elif q_word == 'how' and not any(word in response_lower for word in ['by', 'through', 'process', 'method']):
                    coverage_score *= 0.8
                elif q_word == 'why' and not any(word in response_lower for word in ['because', 'due to', 'reason']):
                    coverage_score *= 0.8
        
        return (length_score + coverage_score) / 2
    
    def _calculate_factual_indicators(self, response: str, documents: List[Dict[str, Any]]) -> float:
        """Calculate indicators of factual accuracy"""
        
        factual_score = 0.5  # Start neutral
        
        # Check for uncertainty expressions (good sign)
        uncertainty_phrases = [
            'according to', 'based on', 'suggests that', 'indicates that',
            'appears to', 'seems to', 'likely', 'possibly', 'may be'
        ]
        
        uncertainty_count = sum(1 for phrase in uncertainty_phrases if phrase in response.lower())
        if uncertainty_count > 0:
            factual_score += min(0.2, uncertainty_count * 0.05)
        
        # Check for absolute statements without grounding (bad sign)
        absolute_phrases = [
            'definitely', 'certainly', 'absolutely', 'without doubt',
            'always', 'never', 'all', 'none', 'every', 'completely'
        ]
        
        if not documents:  # No documents to ground absolute statements
            absolute_count = sum(1 for phrase in absolute_phrases if phrase in response.lower())
            if absolute_count > 0:
                factual_score -= min(0.3, absolute_count * 0.1)
        
        # Check for specific citations or references
        citation_indicators = [
            'according to the document', 'the text states', 'as mentioned',
            'the source indicates', 'from the provided information'
        ]
        
        citation_count = sum(1 for phrase in citation_indicators if phrase in response.lower())
        if citation_count > 0:
            factual_score += min(0.3, citation_count * 0.1)
        
        return min(1.0, max(0.0, factual_score))
    
    def _calculate_coherence(self, response: str) -> float:
        """Calculate response coherence and language quality"""
        
        if not response.strip():
            return 0.0
        
        # Sentence structure
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        if not sentences:
            return 0.1
        
        coherence_factors = []
        
        # 1. Sentence length variation
        if len(sentences) > 1:
            lengths = [len(s.split()) for s in sentences]
            length_variance = statistics.variance(lengths) if len(lengths) > 1 else 0
            # Moderate variance is good
            variance_score = 1.0 - min(1.0, length_variance / 100)
            coherence_factors.append(variance_score)
        else:
            coherence_factors.append(0.8)  # Single sentence is okay but not optimal
        
        # 2. Transition and connector words
        connectors = {
            'however', 'therefore', 'furthermore', 'moreover', 'additionally',
            'consequently', 'meanwhile', 'nevertheless', 'thus', 'hence',
            'also', 'furthermore', 'in addition', 'for example', 'specifically'
        }
        
        response_words = set(response.lower().split())
        connector_count = len(response_words & connectors)
        connector_score = min(1.0, connector_count / max(1, len(sentences) - 1))
        coherence_factors.append(connector_score)
        
        # 3. Repetition check (penalize excessive repetition)
        words = response.lower().split()
        unique_words = len(set(words))
        total_words = len(words)
        
        if total_words > 0:
            diversity_score = unique_words / total_words
            coherence_factors.append(diversity_score)
        else:
            coherence_factors.append(0.0)
        
        return np.mean(coherence_factors)


class HallucinationDetector:
    """Detects potential hallucinations in responses"""
    
    def __init__(self):
        self.semantic_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logger.warning(f"Failed to load semantic model: {e}")
    
    def detect_hallucination_risk(self, 
                                response: str,
                                retrieved_documents: List[Dict[str, Any]],
                                query: str = "") -> float:
        """Detect hallucination risk in response"""
        
        if not retrieved_documents:
            # No grounding documents - high hallucination risk
            return 0.8
        
        risk_factors = []
        
        # 1. Factual claims not supported by documents
        unsupported_claims_risk = self._check_unsupported_claims(response, retrieved_documents)
        risk_factors.append(('unsupported_claims', unsupported_claims_risk, 0.4))
        
        # 2. Semantic drift from source material
        semantic_drift_risk = self._check_semantic_drift(response, retrieved_documents)
        risk_factors.append(('semantic_drift', semantic_drift_risk, 0.3))
        
        # 3. Specificity without grounding
        specificity_risk = self._check_ungrounded_specificity(response, retrieved_documents)
        risk_factors.append(('ungrounded_specificity', specificity_risk, 0.2))
        
        # 4. Contradictions with source material
        contradiction_risk = self._check_contradictions(response, retrieved_documents)
        risk_factors.append(('contradictions', contradiction_risk, 0.1))
        
        # Calculate weighted risk
        total_risk = sum(risk * weight for _, risk, weight in risk_factors)
        
        return min(1.0, max(0.0, total_risk))
    
    def _check_unsupported_claims(self, response: str, documents: List[Dict[str, Any]]) -> float:
        """Check for factual claims not supported by documents"""
        
        # Combine document content
        combined_content = " ".join([
            doc.get('content', '') for doc in documents[:5]
        ]).lower()
        
        if not combined_content.strip():
            return 0.8  # No content to support claims
        
        # Look for specific factual patterns
        import re
        
        # Numbers and statistics
        response_numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', response)
        content_numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', combined_content)
        
        if response_numbers:
            unsupported_numbers = [num for num in response_numbers if num not in content_numbers]
            number_risk = len(unsupported_numbers) / len(response_numbers)
        else:
            number_risk = 0.0
        
        # Proper nouns and specific entities
        response_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', response)
        content_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', combined_content.title())
        
        if response_entities:
            unsupported_entities = [entity for entity in response_entities 
                                  if entity.lower() not in combined_content]
            entity_risk = len(unsupported_entities) / len(response_entities)
        else:
            entity_risk = 0.0
        
        return (number_risk + entity_risk) / 2
    
    def _check_semantic_drift(self, response: str, documents: List[Dict[str, Any]]) -> float:
        """Check for semantic drift from source material"""
        
        if not self.semantic_model or not documents:
            return 0.5  # Neutral risk
        
        try:
            # Combine document content
            combined_content = " ".join([
                doc.get('content', '') for doc in documents[:3]  # Top 3 documents
            ])
            
            if not combined_content.strip():
                return 0.7
            
            # Calculate semantic similarity
            response_embedding = self.semantic_model.encode([response])
            content_embedding = self.semantic_model.encode([combined_content])
            
            similarity = util.cos_sim(response_embedding, content_embedding)[0][0].item()
            
            # Convert similarity to risk (inverse relationship)
            drift_risk = 1.0 - max(0.0, similarity)
            
            return drift_risk
            
        except Exception as e:
            logger.warning(f"Semantic drift calculation failed: {e}")
            return 0.5
    
    def _check_ungrounded_specificity(self, response: str, documents: List[Dict[str, Any]]) -> float:
        """Check for specific claims without grounding"""
        
        # Combine document content
        combined_content = " ".join([
            doc.get('content', '') for doc in documents[:5]
        ])
        
        specificity_indicators = [
            'exactly', 'precisely', 'specifically', 'particularly',
            'in fact', 'actually', 'definitely', 'certainly'
        ]
        
        response_lower = response.lower()
        specificity_count = sum(1 for indicator in specificity_indicators 
                              if indicator in response_lower)
        
        if specificity_count == 0:
            return 0.0  # No specific claims
        
        # Check if specific claims are grounded
        response_words = set(response_lower.split())
        content_words = set(combined_content.lower().split())
        
        grounding_ratio = len(response_words & content_words) / len(response_words) if response_words else 0
        
        # Higher specificity with lower grounding = higher risk
        specificity_risk = (specificity_count * 0.2) * (1.0 - grounding_ratio)
        
        return min(1.0, specificity_risk)
    
    def _check_contradictions(self, response: str, documents: List[Dict[str, Any]]) -> float:
        """Check for contradictions with source material"""
        
        # This is a simplified implementation
        # A more sophisticated approach would use NLI models
        
        contradiction_patterns = [
            ('not', 'is'), ('never', 'always'), ('no', 'yes'),
            ('false', 'true'), ('incorrect', 'correct')
        ]
        
        combined_content = " ".join([
            doc.get('content', '') for doc in documents[:3]
        ]).lower()
        
        response_lower = response.lower()
        
        contradiction_risk = 0.0
        
        for neg_word, pos_word in contradiction_patterns:
            if neg_word in response_lower and pos_word in combined_content:
                # Simple contradiction check - this is very basic
                contradiction_risk += 0.1
            elif pos_word in response_lower and neg_word in combined_content:
                contradiction_risk += 0.1
        
        return min(1.0, contradiction_risk)


class QualityMonitor:
    """Main quality monitoring system"""
    
    def __init__(self, config: MonitoringConfiguration = None):
        self.config = config or MonitoringConfiguration()
        self.confidence_scorer = ConfidenceScorer()
        self.hallucination_detector = HallucinationDetector()
        
        # Monitoring state
        self.quality_history = deque(maxlen=10000)  # Keep last 10k measurements
        self.alerts = deque(maxlen=1000)  # Keep last 1k alerts
        self.feedback_history = deque(maxlen=5000)  # Keep last 5k feedback items
        self.alert_counts = defaultdict(int)  # Alert count by hour
        
        # Threading
        self.monitoring_active = False
        self.monitoring_thread = None
        self.lock = threading.RLock()
        
        # Callbacks
        self.alert_callbacks = []
        self.quality_callbacks = []
    
    def start_monitoring(self):
        """Start background quality monitoring"""
        with self.lock:
            if self.monitoring_active:
                return
            
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            logger.info("Quality monitoring started")
    
    def stop_monitoring(self):
        """Stop background quality monitoring"""
        with self.lock:
            self.monitoring_active = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5)
            
            logger.info("Quality monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._perform_quality_checks()
                self._cleanup_old_data()
                
                time.sleep(self.config.quality_check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)  # Wait before retrying
    
    def record_interaction(self, 
                         query: str,
                         response: str,
                         retrieved_documents: List[Dict[str, Any]],
                         metadata: Dict[str, Any] = None) -> QualityMetrics:
        """Record and analyze a user interaction"""
        
        start_time = time.time()
        
        # Calculate quality metrics
        confidence_score = self.confidence_scorer.calculate_confidence(
            query, response, retrieved_documents, metadata
        )
        
        hallucination_risk = self.hallucination_detector.detect_hallucination_risk(
            response, retrieved_documents, query
        )
        
        # Calculate other metrics
        relevance_score = self._calculate_response_relevance(query, response)
        consistency_score = self._calculate_factual_consistency(response, retrieved_documents)
        completeness_score = self._calculate_completeness(response, query)
        coherence_score = self._calculate_coherence(response)
        
        # Overall quality score
        overall_quality = np.mean([
            confidence_score,
            1.0 - hallucination_risk,  # Invert risk to get quality
            relevance_score,
            consistency_score,
            completeness_score,
            coherence_score
        ])
        
        # Create quality metrics
        quality_metrics = QualityMetrics(
            timestamp=datetime.now(),
            confidence_score=confidence_score,
            hallucination_risk=hallucination_risk,
            response_relevance=relevance_score,
            factual_consistency=consistency_score,
            completeness=completeness_score,
            coherence=coherence_score,
            overall_quality=overall_quality,
            metadata={
                'query': query,
                'response_length': len(response),
                'document_count': len(retrieved_documents),
                'processing_time_ms': (time.time() - start_time) * 1000,
                **(metadata or {})
            }
        )
        
        # Store in history
        with self.lock:
            self.quality_history.append(quality_metrics)
        
        # Check for alerts
        self._check_quality_alerts(quality_metrics, query, response)
        
        # Notify callbacks
        for callback in self.quality_callbacks:
            try:
                callback(quality_metrics)
            except Exception as e:
                logger.error(f"Quality callback error: {e}")
        
        return quality_metrics
    
    def add_user_feedback(self, feedback: UserFeedback):
        """Add user feedback to monitoring system"""
        with self.lock:
            self.feedback_history.append(feedback)
        
        # Update user satisfaction metrics
        self._update_satisfaction_metrics()
        
        logger.info(f"User feedback recorded: {feedback.feedback_type.value}")
    
    def _calculate_response_relevance(self, query: str, response: str) -> float:
        """Calculate relevance score using the confidence scorer"""
        if hasattr(self.confidence_scorer, '_calculate_semantic_similarity'):
            return self.confidence_scorer._calculate_semantic_similarity(query, response)
        return 0.5
    
    def _calculate_factual_consistency(self, response: str, documents: List[Dict[str, Any]]) -> float:
        """Calculate factual consistency score"""
        if hasattr(self.confidence_scorer, '_calculate_factual_indicators'):
            return self.confidence_scorer._calculate_factual_indicators(response, documents)
        return 0.5
    
    def _calculate_completeness(self, response: str, query: str) -> float:
        """Calculate completeness score"""
        if hasattr(self.confidence_scorer, '_calculate_completeness'):
            return self.confidence_scorer._calculate_completeness(response, query)
        return 0.5
    
    def _calculate_coherence(self, response: str) -> float:
        """Calculate coherence score"""
        if hasattr(self.confidence_scorer, '_calculate_coherence'):
            return self.confidence_scorer._calculate_coherence(response)
        return 0.5
    
    def _check_quality_alerts(self, metrics: QualityMetrics, query: str, response: str):
        """Check if quality metrics trigger alerts"""
        
        if not self.config.enable_alerts:
            return
        
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        hour_alert_count = self.alert_counts[current_hour]
        
        if hour_alert_count >= self.config.max_alerts_per_hour:
            return  # Rate limiting
        
        alerts_to_create = []
        
        # Check each metric against thresholds
        if metrics.confidence_score < self.config.confidence_threshold:
            alerts_to_create.append((
                AlertLevel.WARNING,
                QualityMetric.CONFIDENCE_SCORE,
                f"Low confidence score: {metrics.confidence_score:.2f}",
                metrics.confidence_score,
                self.config.confidence_threshold
            ))
        
        if metrics.hallucination_risk > self.config.hallucination_threshold:
            alerts_to_create.append((
                AlertLevel.ERROR,
                QualityMetric.HALLUCINATION_RISK,
                f"High hallucination risk: {metrics.hallucination_risk:.2f}",
                metrics.hallucination_risk,
                self.config.hallucination_threshold
            ))
        
        if metrics.response_relevance < self.config.relevance_threshold:
            alerts_to_create.append((
                AlertLevel.WARNING,
                QualityMetric.RESPONSE_RELEVANCE,
                f"Low response relevance: {metrics.response_relevance:.2f}",
                metrics.response_relevance,
                self.config.relevance_threshold
            ))
        
        if metrics.factual_consistency < self.config.consistency_threshold:
            alerts_to_create.append((
                AlertLevel.ERROR,
                QualityMetric.FACTUAL_CONSISTENCY,
                f"Low factual consistency: {metrics.factual_consistency:.2f}",
                metrics.factual_consistency,
                self.config.consistency_threshold
            ))
        
        # Create alerts
        for level, metric, message, value, threshold in alerts_to_create:
            alert = QualityAlert(
                level=level,
                metric=metric,
                message=message,
                value=value,
                threshold=threshold,
                query=query[:200],  # Truncate for storage
                response=response[:500],
                metadata={'overall_quality': metrics.overall_quality}
            )
            
            with self.lock:
                self.alerts.append(alert)
                self.alert_counts[current_hour] += 1
            
            # Notify callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")
            
            logger.warning(f"Quality alert: {message}")
    
    def _perform_quality_checks(self):
        """Perform periodic quality checks"""
        
        with self.lock:
            if not self.quality_history:
                return
            
            # Check recent quality trends
            recent_window = datetime.now() - timedelta(minutes=30)
            recent_metrics = [
                m for m in self.quality_history 
                if m.timestamp >= recent_window
            ]
            
            if len(recent_metrics) < 5:
                return  # Not enough data
            
            # Calculate trend
            recent_quality_scores = [m.overall_quality for m in recent_metrics]
            avg_recent_quality = np.mean(recent_quality_scores)
            
            # Compare with historical average
            if len(self.quality_history) > 100:
                historical_scores = [m.overall_quality for m in list(self.quality_history)[-100:]]
                avg_historical_quality = np.mean(historical_scores)
                
                # Check for significant degradation
                if avg_recent_quality < avg_historical_quality - 0.15:  # 15% degradation
                    alert = QualityAlert(
                        level=AlertLevel.CRITICAL,
                        metric=QualityMetric.CONFIDENCE_SCORE,  # General quality
                        message=f"Quality degradation detected: {avg_recent_quality:.2f} vs {avg_historical_quality:.2f} historical average",
                        value=avg_recent_quality,
                        threshold=avg_historical_quality,
                        metadata={'trend_analysis': True}
                    )
                    
                    self.alerts.append(alert)
                    
                    for callback in self.alert_callbacks:
                        try:
                            callback(alert)
                        except Exception as e:
                            logger.error(f"Alert callback error: {e}")
    
    def _update_satisfaction_metrics(self):
        """Update user satisfaction metrics based on feedback"""
        
        with self.lock:
            if not self.feedback_history:
                return
            
            recent_feedback = [
                f for f in self.feedback_history 
                if f.timestamp >= datetime.now() - timedelta(hours=self.config.trend_analysis_window_hours)
            ]
            
            if len(recent_feedback) < self.config.min_feedback_samples:
                return
            
            # Calculate satisfaction score
            satisfaction_scores = []
            
            for feedback in recent_feedback:
                if feedback.feedback_type == FeedbackType.RATING and feedback.rating:
                    satisfaction_scores.append(feedback.rating / 5.0)  # Normalize to 0-1
                elif feedback.feedback_type == FeedbackType.THUMBS_UP:
                    satisfaction_scores.append(1.0)
                elif feedback.feedback_type == FeedbackType.THUMBS_DOWN:
                    satisfaction_scores.append(0.0)
            
            if satisfaction_scores:
                avg_satisfaction = np.mean(satisfaction_scores)
                
                # Check against threshold
                if avg_satisfaction < self.config.user_satisfaction_threshold / 5.0:  # Normalize threshold
                    alert = QualityAlert(
                        level=AlertLevel.WARNING,
                        metric=QualityMetric.USER_SATISFACTION,
                        message=f"Low user satisfaction: {avg_satisfaction:.2f}",
                        value=avg_satisfaction,
                        threshold=self.config.user_satisfaction_threshold / 5.0,
                        metadata={'feedback_count': len(satisfaction_scores)}
                    )
                    
                    self.alerts.append(alert)
    
    def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        
        cutoff_time = datetime.now() - timedelta(hours=self.config.trend_analysis_window_hours * 2)
        
        with self.lock:
            # Clean up alert counts
            old_hours = [hour for hour in self.alert_counts if hour < cutoff_time]
            for hour in old_hours:
                del self.alert_counts[hour]
    
    def get_quality_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get quality summary for specified time window"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        with self.lock:
            recent_metrics = [
                m for m in self.quality_history 
                if m.timestamp >= cutoff_time
            ]
            
            recent_alerts = [
                a for a in self.alerts 
                if a.timestamp >= cutoff_time
            ]
            
            recent_feedback = [
                f for f in self.feedback_history 
                if f.timestamp >= cutoff_time
            ]
        
        if not recent_metrics:
            return {'error': 'No data available for specified time window'}
        
        # Calculate summary statistics
        quality_scores = [m.overall_quality for m in recent_metrics]
        confidence_scores = [m.confidence_score for m in recent_metrics]
        hallucination_risks = [m.hallucination_risk for m in recent_metrics]
        
        # Alert summary
        alert_counts_by_level = defaultdict(int)
        for alert in recent_alerts:
            alert_counts_by_level[alert.level.value] += 1
        
        # Feedback summary
        satisfaction_scores = []
        for feedback in recent_feedback:
            if feedback.feedback_type == FeedbackType.RATING and feedback.rating:
                satisfaction_scores.append(feedback.rating)
        
        summary = {
            'time_window_hours': hours_back,
            'total_interactions': len(recent_metrics),
            'quality_metrics': {
                'average_overall_quality': np.mean(quality_scores),
                'quality_std_dev': np.std(quality_scores),
                'min_quality': np.min(quality_scores),
                'max_quality': np.max(quality_scores),
                'average_confidence': np.mean(confidence_scores),
                'average_hallucination_risk': np.mean(hallucination_risks)
            },
            'alerts': {
                'total_alerts': len(recent_alerts),
                'by_level': dict(alert_counts_by_level),
                'alert_rate_per_hour': len(recent_alerts) / hours_back if hours_back > 0 else 0
            },
            'user_feedback': {
                'total_feedback': len(recent_feedback),
                'average_satisfaction': np.mean(satisfaction_scores) if satisfaction_scores else None,
                'feedback_rate': len(recent_feedback) / len(recent_metrics) if recent_metrics else 0
            }
        }
        
        return summary
    
    def register_alert_callback(self, callback: Callable[[QualityAlert], None]):
        """Register callback for quality alerts"""
        self.alert_callbacks.append(callback)
    
    def register_quality_callback(self, callback: Callable[[QualityMetrics], None]):
        """Register callback for quality metrics"""
        self.quality_callbacks.append(callback)
    
    def get_active_alerts(self) -> List[QualityAlert]:
        """Get all active (unresolved) alerts"""
        with self.lock:
            return [alert for alert in self.alerts if not alert.resolved]
    
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved"""
        with self.lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.resolved = True
                    logger.info(f"Alert resolved: {alert_id}")
                    break
    
    def export_monitoring_data(self, filepath: str, hours_back: int = 24):
        """Export monitoring data to file"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        with self.lock:
            recent_metrics = [
                asdict(m) for m in self.quality_history 
                if m.timestamp >= cutoff_time
            ]
            
            recent_alerts = [
                asdict(a) for a in self.alerts 
                if a.timestamp >= cutoff_time
            ]
            
            recent_feedback = [
                asdict(f) for f in self.feedback_history 
                if f.timestamp >= cutoff_time
            ]
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'time_window_hours': hours_back,
            'quality_metrics': recent_metrics,
            'alerts': recent_alerts,
            'feedback': recent_feedback,
            'summary': self.get_quality_summary(hours_back)
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Monitoring data exported to {filepath}")


# Example usage and integration helpers

def create_simple_alert_handler():
    """Create a simple alert handler that logs alerts"""
    
    def alert_handler(alert: QualityAlert):
        level_colors = {
            AlertLevel.INFO: '\033[94m',      # Blue
            AlertLevel.WARNING: '\033[93m',    # Yellow
            AlertLevel.ERROR: '\033[91m',      # Red
            AlertLevel.CRITICAL: '\033[95m'    # Magenta
        }
        
        color = level_colors.get(alert.level, '')
        reset_color = '\033[0m'
        
        print(f"{color}[{alert.level.value.upper()}] {alert.message}{reset_color}")
    
    return alert_handler


def create_webhook_alert_handler(webhook_url: str):
    """Create alert handler that sends alerts to webhook"""
    
    def webhook_handler(alert: QualityAlert):
        try:
            import requests
            
            payload = {
                'alert_id': alert.id,
                'level': alert.level.value,
                'metric': alert.metric.value,
                'message': alert.message,
                'value': alert.value,
                'threshold': alert.threshold,
                'timestamp': alert.timestamp.isoformat()
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Webhook alert handler failed: {e}")
    
    return webhook_handler