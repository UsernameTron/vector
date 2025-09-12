"""
Comprehensive RAG Evaluation Framework

This module provides systematic evaluation capabilities for RAG systems:
- Retrieval quality metrics (Recall@K, MRR, NDCG)
- Generation quality assessment (BLEU, ROUGE, BERTScore)
- End-to-end RAG evaluation (RAGAS framework)
- Groundedness and faithfulness evaluation
- A/B testing framework
- Performance benchmarking
"""

import logging
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import time
import statistics
from collections import defaultdict, Counter
import hashlib
import concurrent.futures
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    NLTK_AVAILABLE = True
except ImportError:
    logger.warning("NLTK and rouge-score not available. Some metrics will be disabled.")
    NLTK_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("SentenceTransformers not available. Some metrics will be disabled.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class EvaluationMetric(Enum):
    """Types of evaluation metrics"""
    # Retrieval metrics
    RECALL_AT_K = "recall_at_k"
    PRECISION_AT_K = "precision_at_k"
    MRR = "mean_reciprocal_rank"
    NDCG = "normalized_discounted_cumulative_gain"
    
    # Generation metrics
    BLEU = "bleu"
    ROUGE_1 = "rouge_1"
    ROUGE_2 = "rouge_2"
    ROUGE_L = "rouge_l"
    BERT_SCORE = "bert_score"
    
    # RAG-specific metrics
    GROUNDEDNESS = "groundedness"
    FAITHFULNESS = "faithfulness"
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    INFORMATIVENESS = "informativeness"
    
    # System metrics
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"


@dataclass
class EvaluationExample:
    """Single evaluation example with query, ground truth, and system output"""
    query: str
    ground_truth_answer: str
    system_answer: str
    retrieved_documents: List[Dict[str, Any]]
    ground_truth_documents: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RetrievalEvaluationResult:
    """Results from retrieval evaluation"""
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    average_precision: float = 0.0


@dataclass
class GenerationEvaluationResult:
    """Results from generation evaluation"""
    bleu_1: float = 0.0
    bleu_2: float = 0.0
    bleu_3: float = 0.0
    bleu_4: float = 0.0
    rouge_1_f: float = 0.0
    rouge_1_p: float = 0.0
    rouge_1_r: float = 0.0
    rouge_2_f: float = 0.0
    rouge_2_p: float = 0.0
    rouge_2_r: float = 0.0
    rouge_l_f: float = 0.0
    rouge_l_p: float = 0.0
    rouge_l_r: float = 0.0
    bert_score_f1: float = 0.0
    bert_score_precision: float = 0.0
    bert_score_recall: float = 0.0


@dataclass
class RAGEvaluationResult:
    """Comprehensive RAG evaluation results"""
    groundedness_score: float = 0.0
    faithfulness_score: float = 0.0
    relevance_score: float = 0.0
    coherence_score: float = 0.0
    informativeness_score: float = 0.0
    overall_rag_score: float = 0.0


@dataclass
class SystemPerformanceResult:
    """System performance metrics"""
    average_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    throughput_queries_per_second: float = 0.0
    peak_memory_usage_mb: float = 0.0
    average_memory_usage_mb: float = 0.0
    error_rate: float = 0.0


@dataclass
class ComprehensiveEvaluationResult:
    """Complete evaluation results"""
    retrieval_metrics: RetrievalEvaluationResult
    generation_metrics: GenerationEvaluationResult
    rag_metrics: RAGEvaluationResult
    performance_metrics: SystemPerformanceResult
    overall_score: float = 0.0
    evaluation_timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RetrievalEvaluator:
    """Evaluates retrieval quality using standard IR metrics"""
    
    def __init__(self):
        self.evaluation_cache = {}
    
    def evaluate_retrieval(self, 
                         examples: List[EvaluationExample],
                         k_values: List[int] = None) -> RetrievalEvaluationResult:
        """Evaluate retrieval performance across multiple examples"""
        
        if k_values is None:
            k_values = [1, 3, 5, 10]
        
        all_recall_scores = defaultdict(list)
        all_precision_scores = defaultdict(list)
        all_mrr_scores = []
        all_ndcg_scores = defaultdict(list)
        all_ap_scores = []
        
        for example in examples:
            if not example.ground_truth_documents:
                continue  # Skip examples without ground truth
            
            retrieved_doc_ids = [
                doc.get('id', doc.get('document_id', str(hash(doc.get('content', '')))))
                for doc in example.retrieved_documents
            ]
            
            # Calculate metrics for this example
            recall_scores = self._calculate_recall_at_k(
                retrieved_doc_ids, example.ground_truth_documents, k_values
            )
            precision_scores = self._calculate_precision_at_k(
                retrieved_doc_ids, example.ground_truth_documents, k_values
            )
            mrr_score = self._calculate_mrr(retrieved_doc_ids, example.ground_truth_documents)
            ndcg_scores = self._calculate_ndcg_at_k(
                retrieved_doc_ids, example.ground_truth_documents, [5, 10]
            )
            ap_score = self._calculate_average_precision(
                retrieved_doc_ids, example.ground_truth_documents
            )
            
            # Accumulate scores
            for k in k_values:
                all_recall_scores[k].append(recall_scores.get(k, 0.0))
                all_precision_scores[k].append(precision_scores.get(k, 0.0))
            
            all_mrr_scores.append(mrr_score)
            all_ap_scores.append(ap_score)
            
            for k in [5, 10]:
                all_ndcg_scores[k].append(ndcg_scores.get(k, 0.0))
        
        # Compute averages
        result = RetrievalEvaluationResult()
        
        if 1 in all_recall_scores:
            result.recall_at_1 = np.mean(all_recall_scores[1])
            result.precision_at_1 = np.mean(all_precision_scores[1])
        if 3 in all_recall_scores:
            result.recall_at_3 = np.mean(all_recall_scores[3])
            result.precision_at_3 = np.mean(all_precision_scores[3])
        if 5 in all_recall_scores:
            result.recall_at_5 = np.mean(all_recall_scores[5])
            result.precision_at_5 = np.mean(all_precision_scores[5])
        if 10 in all_recall_scores:
            result.recall_at_10 = np.mean(all_recall_scores[10])
            result.precision_at_10 = np.mean(all_precision_scores[10])
        
        if all_mrr_scores:
            result.mrr = np.mean(all_mrr_scores)
        if 5 in all_ndcg_scores:
            result.ndcg_at_5 = np.mean(all_ndcg_scores[5])
        if 10 in all_ndcg_scores:
            result.ndcg_at_10 = np.mean(all_ndcg_scores[10])
        if all_ap_scores:
            result.average_precision = np.mean(all_ap_scores)
        
        return result
    
    def _calculate_recall_at_k(self, 
                             retrieved_docs: List[str], 
                             relevant_docs: List[str], 
                             k_values: List[int]) -> Dict[int, float]:
        """Calculate Recall@K for given k values"""
        relevant_set = set(relevant_docs)
        results = {}
        
        for k in k_values:
            retrieved_at_k = set(retrieved_docs[:k])
            if relevant_set:
                recall = len(retrieved_at_k & relevant_set) / len(relevant_set)
            else:
                recall = 0.0
            results[k] = recall
        
        return results
    
    def _calculate_precision_at_k(self, 
                                retrieved_docs: List[str], 
                                relevant_docs: List[str], 
                                k_values: List[int]) -> Dict[int, float]:
        """Calculate Precision@K for given k values"""
        relevant_set = set(relevant_docs)
        results = {}
        
        for k in k_values:
            retrieved_at_k = retrieved_docs[:k]
            if retrieved_at_k:
                precision = len([doc for doc in retrieved_at_k if doc in relevant_set]) / len(retrieved_at_k)
            else:
                precision = 0.0
            results[k] = precision
        
        return results
    
    def _calculate_mrr(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """Calculate Mean Reciprocal Rank"""
        relevant_set = set(relevant_docs)
        
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_set:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def _calculate_ndcg_at_k(self, 
                           retrieved_docs: List[str], 
                           relevant_docs: List[str], 
                           k_values: List[int]) -> Dict[int, float]:
        """Calculate NDCG@K"""
        relevant_set = set(relevant_docs)
        results = {}
        
        for k in k_values:
            # Create relevance scores (1 for relevant, 0 for not relevant)
            relevance_scores = [1 if doc in relevant_set else 0 for doc in retrieved_docs[:k]]
            
            # Calculate DCG
            dcg = sum(
                (2 ** rel - 1) / np.log2(i + 2)
                for i, rel in enumerate(relevance_scores)
            )
            
            # Calculate IDCG (ideal DCG)
            ideal_relevance = sorted([1] * min(k, len(relevant_docs)) + [0] * max(0, k - len(relevant_docs)), reverse=True)
            idcg = sum(
                (2 ** rel - 1) / np.log2(i + 2)
                for i, rel in enumerate(ideal_relevance)
            )
            
            # Calculate NDCG
            if idcg > 0:
                ndcg = dcg / idcg
            else:
                ndcg = 0.0
            
            results[k] = ndcg
        
        return results
    
    def _calculate_average_precision(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """Calculate Average Precision"""
        relevant_set = set(relevant_docs)
        precision_sum = 0.0
        relevant_retrieved = 0
        
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_set:
                relevant_retrieved += 1
                precision_at_i = relevant_retrieved / (i + 1)
                precision_sum += precision_at_i
        
        if relevant_retrieved > 0:
            return precision_sum / len(relevant_set)
        else:
            return 0.0


class GenerationEvaluator:
    """Evaluates generation quality using NLG metrics"""
    
    def __init__(self):
        self.rouge_scorer = None
        self.bert_model = None
        
        if NLTK_AVAILABLE:
            try:
                self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                logger.info("ROUGE scorer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize ROUGE scorer: {e}")
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("BERT model for scoring initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize BERT model: {e}")
    
    def evaluate_generation(self, examples: List[EvaluationExample]) -> GenerationEvaluationResult:
        """Evaluate generation quality across multiple examples"""
        
        all_bleu_scores = defaultdict(list)
        all_rouge_scores = defaultdict(list)
        all_bert_scores = defaultdict(list)
        
        for example in examples:
            # BLEU scores
            if NLTK_AVAILABLE:
                bleu_scores = self._calculate_bleu_scores(
                    example.system_answer, example.ground_truth_answer
                )
                for n in [1, 2, 3, 4]:
                    all_bleu_scores[n].append(bleu_scores.get(n, 0.0))
            
            # ROUGE scores
            if self.rouge_scorer:
                rouge_scores = self._calculate_rouge_scores(
                    example.system_answer, example.ground_truth_answer
                )
                for metric in ['rouge1', 'rouge2', 'rougeL']:
                    for stat in ['f', 'p', 'r']:
                        key = f"{metric}_{stat}"
                        all_rouge_scores[key].append(rouge_scores.get(key, 0.0))
            
            # BERTScore
            if self.bert_model:
                bert_scores = self._calculate_bert_score(
                    example.system_answer, example.ground_truth_answer
                )
                for stat in ['f1', 'precision', 'recall']:
                    all_bert_scores[stat].append(bert_scores.get(stat, 0.0))
        
        # Compute averages
        result = GenerationEvaluationResult()
        
        # BLEU scores
        for n in [1, 2, 3, 4]:
            if n in all_bleu_scores:
                setattr(result, f'bleu_{n}', np.mean(all_bleu_scores[n]))
        
        # ROUGE scores
        rouge_mapping = {
            'rouge1_f': 'rouge_1_f', 'rouge1_p': 'rouge_1_p', 'rouge1_r': 'rouge_1_r',
            'rouge2_f': 'rouge_2_f', 'rouge2_p': 'rouge_2_p', 'rouge2_r': 'rouge_2_r',
            'rougeL_f': 'rouge_l_f', 'rougeL_p': 'rouge_l_p', 'rougeL_r': 'rouge_l_r'
        }
        
        for rouge_key, result_attr in rouge_mapping.items():
            if rouge_key in all_rouge_scores:
                setattr(result, result_attr, np.mean(all_rouge_scores[rouge_key]))
        
        # BERTScore
        if 'f1' in all_bert_scores:
            result.bert_score_f1 = np.mean(all_bert_scores['f1'])
        if 'precision' in all_bert_scores:
            result.bert_score_precision = np.mean(all_bert_scores['precision'])
        if 'recall' in all_bert_scores:
            result.bert_score_recall = np.mean(all_bert_scores['recall'])
        
        return result
    
    def _calculate_bleu_scores(self, candidate: str, reference: str) -> Dict[int, float]:
        """Calculate BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores"""
        if not NLTK_AVAILABLE:
            return {}
        
        # Tokenize
        candidate_tokens = candidate.lower().split()
        reference_tokens = [reference.lower().split()]
        
        smoothing_function = SmoothingFunction().method4
        scores = {}
        
        for n in [1, 2, 3, 4]:
            weights = [1.0/n] * n + [0.0] * (4-n)
            try:
                score = sentence_bleu(
                    reference_tokens, candidate_tokens, 
                    weights=weights, smoothing_function=smoothing_function
                )
                scores[n] = score
            except Exception:
                scores[n] = 0.0
        
        return scores
    
    def _calculate_rouge_scores(self, candidate: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        if not self.rouge_scorer:
            return {}
        
        try:
            scores = self.rouge_scorer.score(reference, candidate)
            
            result = {}
            for metric in ['rouge1', 'rouge2', 'rougeL']:
                score_obj = scores[metric]
                result[f'{metric}_f'] = score_obj.fmeasure
                result[f'{metric}_p'] = score_obj.precision
                result[f'{metric}_r'] = score_obj.recall
            
            return result
        except Exception as e:
            logger.warning(f"ROUGE calculation failed: {e}")
            return {}
    
    def _calculate_bert_score(self, candidate: str, reference: str) -> Dict[str, float]:
        """Calculate BERTScore using sentence similarity"""
        if not self.bert_model:
            return {}
        
        try:
            # Encode sentences
            candidate_embedding = self.bert_model.encode([candidate])
            reference_embedding = self.bert_model.encode([reference])
            
            # Calculate cosine similarity
            similarity = util.cos_sim(candidate_embedding, reference_embedding)[0][0].item()
            
            # For simplicity, use similarity as all three metrics
            # In a more sophisticated implementation, you'd calculate token-level alignments
            return {
                'f1': similarity,
                'precision': similarity,
                'recall': similarity
            }
        except Exception as e:
            logger.warning(f"BERTScore calculation failed: {e}")
            return {}


class RAGSpecificEvaluator:
    """Evaluates RAG-specific quality dimensions"""
    
    def __init__(self):
        self.semantic_model = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logger.warning(f"Failed to initialize semantic model: {e}")
    
    def evaluate_rag_quality(self, examples: List[EvaluationExample]) -> RAGEvaluationResult:
        """Evaluate RAG-specific quality metrics"""
        
        groundedness_scores = []
        faithfulness_scores = []
        relevance_scores = []
        coherence_scores = []
        informativeness_scores = []
        
        for example in examples:
            # Groundedness: How well is the answer grounded in retrieved documents
            groundedness = self._calculate_groundedness(
                example.system_answer, example.retrieved_documents
            )
            groundedness_scores.append(groundedness)
            
            # Faithfulness: How faithful is the answer to the retrieved content
            faithfulness = self._calculate_faithfulness(
                example.system_answer, example.retrieved_documents
            )
            faithfulness_scores.append(faithfulness)
            
            # Relevance: How relevant is the answer to the query
            relevance = self._calculate_relevance(
                example.query, example.system_answer
            )
            relevance_scores.append(relevance)
            
            # Coherence: How coherent and well-structured is the answer
            coherence = self._calculate_coherence(example.system_answer)
            coherence_scores.append(coherence)
            
            # Informativeness: How informative is the answer
            informativeness = self._calculate_informativeness(
                example.system_answer, example.ground_truth_answer
            )
            informativeness_scores.append(informativeness)
        
        # Calculate overall RAG score
        all_scores = [
            np.mean(groundedness_scores) if groundedness_scores else 0,
            np.mean(faithfulness_scores) if faithfulness_scores else 0,
            np.mean(relevance_scores) if relevance_scores else 0,
            np.mean(coherence_scores) if coherence_scores else 0,
            np.mean(informativeness_scores) if informativeness_scores else 0
        ]
        overall_score = np.mean(all_scores)
        
        return RAGEvaluationResult(
            groundedness_score=np.mean(groundedness_scores) if groundedness_scores else 0,
            faithfulness_score=np.mean(faithfulness_scores) if faithfulness_scores else 0,
            relevance_score=np.mean(relevance_scores) if relevance_scores else 0,
            coherence_score=np.mean(coherence_scores) if coherence_scores else 0,
            informativeness_score=np.mean(informativeness_scores) if informativeness_scores else 0,
            overall_rag_score=overall_score
        )
    
    def _calculate_groundedness(self, answer: str, retrieved_docs: List[Dict[str, Any]]) -> float:
        """Calculate how well the answer is grounded in retrieved documents"""
        
        if not retrieved_docs or not self.semantic_model:
            return 0.0
        
        try:
            # Combine retrieved document content
            combined_context = " ".join([
                doc.get('content', '') for doc in retrieved_docs[:5]  # Top 5 docs
            ])
            
            if not combined_context.strip():
                return 0.0
            
            # Calculate semantic similarity between answer and context
            answer_embedding = self.semantic_model.encode([answer])
            context_embedding = self.semantic_model.encode([combined_context])
            
            similarity = util.cos_sim(answer_embedding, context_embedding)[0][0].item()
            
            # Normalize to [0, 1] range
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.warning(f"Groundedness calculation failed: {e}")
            return 0.0
    
    def _calculate_faithfulness(self, answer: str, retrieved_docs: List[Dict[str, Any]]) -> float:
        """Calculate faithfulness using overlap of key information"""
        
        if not retrieved_docs:
            return 0.0
        
        # Simple approach: check for factual consistency
        answer_words = set(answer.lower().split())
        context_words = set()
        
        for doc in retrieved_docs[:3]:  # Top 3 docs
            content = doc.get('content', '')
            context_words.update(content.lower().split())
        
        if not context_words:
            return 0.0
        
        # Calculate word overlap as proxy for faithfulness
        overlap = len(answer_words & context_words)
        union = len(answer_words | context_words)
        
        if union == 0:
            return 0.0
        
        # Jaccard similarity as faithfulness proxy
        faithfulness = overlap / len(answer_words) if answer_words else 0.0
        
        return min(1.0, faithfulness)
    
    def _calculate_relevance(self, query: str, answer: str) -> float:
        """Calculate relevance of answer to query"""
        
        if not self.semantic_model:
            # Fallback: simple word overlap
            query_words = set(query.lower().split())
            answer_words = set(answer.lower().split())
            
            overlap = len(query_words & answer_words)
            return overlap / len(query_words) if query_words else 0.0
        
        try:
            query_embedding = self.semantic_model.encode([query])
            answer_embedding = self.semantic_model.encode([answer])
            
            similarity = util.cos_sim(query_embedding, answer_embedding)[0][0].item()
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.warning(f"Relevance calculation failed: {e}")
            return 0.0
    
    def _calculate_coherence(self, answer: str) -> float:
        """Calculate coherence based on text structure"""
        
        if not answer.strip():
            return 0.0
        
        # Simple coherence metrics
        sentences = [s.strip() for s in answer.split('.') if s.strip()]
        
        if len(sentences) == 0:
            return 0.0
        
        coherence_score = 0.0
        
        # Sentence length consistency
        if len(sentences) > 1:
            lengths = [len(s.split()) for s in sentences]
            length_variance = statistics.variance(lengths) if len(lengths) > 1 else 0
            length_consistency = 1.0 / (1.0 + length_variance / 100)  # Normalize
            coherence_score += length_consistency * 0.3
        else:
            coherence_score += 0.3
        
        # Transition words presence
        transition_words = {
            'however', 'therefore', 'furthermore', 'moreover', 'additionally', 
            'consequently', 'meanwhile', 'nevertheless', 'thus', 'hence'
        }
        
        answer_words = set(answer.lower().split())
        transition_count = len(answer_words & transition_words)
        transition_score = min(1.0, transition_count / max(1, len(sentences) - 1))
        coherence_score += transition_score * 0.2
        
        # Answer completeness (not too short, not too long)
        word_count = len(answer.split())
        if 10 <= word_count <= 200:
            completeness_score = 1.0
        elif word_count < 10:
            completeness_score = word_count / 10.0
        else:
            completeness_score = max(0.5, 200.0 / word_count)
        
        coherence_score += completeness_score * 0.5
        
        return min(1.0, coherence_score)
    
    def _calculate_informativeness(self, answer: str, reference: str) -> float:
        """Calculate how informative the answer is"""
        
        if not answer.strip():
            return 0.0
        
        # Information density metrics
        answer_words = answer.split()
        unique_words = len(set(w.lower() for w in answer_words))
        total_words = len(answer_words)
        
        if total_words == 0:
            return 0.0
        
        # Lexical diversity
        diversity = unique_words / total_words
        
        # Content richness (presence of numbers, specific terms)
        numbers = len([w for w in answer_words if any(c.isdigit() for c in w)])
        proper_nouns = len([w for w in answer_words if w[0].isupper() and len(w) > 1])
        
        content_richness = (numbers + proper_nouns) / total_words
        
        # Compare with reference if available
        if reference and self.semantic_model:
            try:
                answer_embedding = self.semantic_model.encode([answer])
                reference_embedding = self.semantic_model.encode([reference])
                
                coverage = util.cos_sim(answer_embedding, reference_embedding)[0][0].item()
            except Exception:
                coverage = 0.5
        else:
            coverage = 0.5
        
        # Combine metrics
        informativeness = (diversity * 0.4 + content_richness * 0.3 + coverage * 0.3)
        
        return min(1.0, informativeness)


class PerformanceEvaluator:
    """Evaluates system performance metrics"""
    
    def __init__(self):
        self.response_times = []
        self.memory_usage = []
        self.error_count = 0
        self.total_requests = 0
        self.start_time = time.time()
    
    def record_request(self, response_time_ms: float, memory_usage_mb: float = 0.0, error: bool = False):
        """Record a single request's performance metrics"""
        self.response_times.append(response_time_ms)
        if memory_usage_mb > 0:
            self.memory_usage.append(memory_usage_mb)
        
        if error:
            self.error_count += 1
        
        self.total_requests += 1
    
    def get_performance_metrics(self) -> SystemPerformanceResult:
        """Calculate performance metrics"""
        
        result = SystemPerformanceResult()
        
        if self.response_times:
            result.average_response_time_ms = np.mean(self.response_times)
            result.p95_response_time_ms = np.percentile(self.response_times, 95)
            result.p99_response_time_ms = np.percentile(self.response_times, 99)
        
        if self.memory_usage:
            result.peak_memory_usage_mb = max(self.memory_usage)
            result.average_memory_usage_mb = np.mean(self.memory_usage)
        
        # Calculate throughput
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            result.throughput_queries_per_second = self.total_requests / elapsed_time
        
        # Calculate error rate
        if self.total_requests > 0:
            result.error_rate = self.error_count / self.total_requests
        
        return result
    
    def reset_metrics(self):
        """Reset all performance metrics"""
        self.response_times.clear()
        self.memory_usage.clear()
        self.error_count = 0
        self.total_requests = 0
        self.start_time = time.time()


class ABTestingFramework:
    """A/B testing framework for RAG configurations"""
    
    def __init__(self):
        self.experiments = {}
        self.results = defaultdict(list)
    
    def create_experiment(self, 
                         experiment_name: str,
                         configuration_a: Dict[str, Any],
                         configuration_b: Dict[str, Any],
                         traffic_split: float = 0.5) -> str:
        """Create a new A/B test experiment"""
        
        experiment_id = hashlib.md5(f"{experiment_name}_{time.time()}".encode()).hexdigest()[:8]
        
        self.experiments[experiment_id] = {
            'name': experiment_name,
            'config_a': configuration_a,
            'config_b': configuration_b,
            'traffic_split': traffic_split,
            'created_at': datetime.now(),
            'status': 'active'
        }
        
        return experiment_id
    
    def assign_configuration(self, experiment_id: str, user_id: str) -> str:
        """Assign user to configuration A or B"""
        
        if experiment_id not in self.experiments:
            return 'default'
        
        # Deterministic assignment based on user_id hash
        user_hash = int(hashlib.md5(user_id.encode()).hexdigest()[:8], 16)
        traffic_split = self.experiments[experiment_id]['traffic_split']
        
        if (user_hash % 100) / 100.0 < traffic_split:
            return 'A'
        else:
            return 'B'
    
    def record_result(self, 
                     experiment_id: str,
                     configuration: str,
                     evaluation_result: ComprehensiveEvaluationResult):
        """Record evaluation result for an experiment"""
        
        if experiment_id not in self.experiments:
            return
        
        self.results[experiment_id].append({
            'configuration': configuration,
            'result': evaluation_result,
            'timestamp': datetime.now()
        })
    
    def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Analyze A/B test results"""
        
        if experiment_id not in self.experiments:
            return {'error': 'Experiment not found'}
        
        results = self.results[experiment_id]
        
        if not results:
            return {'error': 'No results recorded'}
        
        # Separate results by configuration
        config_a_results = [r for r in results if r['configuration'] == 'A']
        config_b_results = [r for r in results if r['configuration'] == 'B']
        
        if not config_a_results or not config_b_results:
            return {'error': 'Insufficient data for both configurations'}
        
        # Calculate metrics for each configuration
        def calculate_avg_metrics(results_list):
            if not results_list:
                return {}
            
            metrics = {
                'overall_score': np.mean([r['result'].overall_score for r in results_list]),
                'retrieval_recall_at_5': np.mean([r['result'].retrieval_metrics.recall_at_5 for r in results_list]),
                'generation_rouge_l': np.mean([r['result'].generation_metrics.rouge_l_f for r in results_list]),
                'rag_groundedness': np.mean([r['result'].rag_metrics.groundedness_score for r in results_list]),
                'response_time': np.mean([r['result'].performance_metrics.average_response_time_ms for r in results_list]),
                'sample_size': len(results_list)
            }
            
            return metrics
        
        config_a_metrics = calculate_avg_metrics(config_a_results)
        config_b_metrics = calculate_avg_metrics(config_b_results)
        
        # Simple statistical significance test (t-test approximation)
        def calculate_significance(metric_name):
            a_scores = [r['result'].overall_score for r in config_a_results]
            b_scores = [r['result'].overall_score for r in config_b_results]
            
            if len(a_scores) < 2 or len(b_scores) < 2:
                return False, 0.0
            
            try:
                from scipy import stats
                t_stat, p_value = stats.ttest_ind(a_scores, b_scores)
                return p_value < 0.05, p_value
            except ImportError:
                # Simple approximation without scipy
                mean_diff = abs(np.mean(a_scores) - np.mean(b_scores))
                pooled_std = np.sqrt((np.var(a_scores) + np.var(b_scores)) / 2)
                return mean_diff > 2 * pooled_std, 0.0  # Rough significance
        
        significant, p_value = calculate_significance('overall_score')
        
        return {
            'experiment': self.experiments[experiment_id],
            'configuration_a': config_a_metrics,
            'configuration_b': config_b_metrics,
            'winner': 'A' if config_a_metrics['overall_score'] > config_b_metrics['overall_score'] else 'B',
            'improvement': abs(config_a_metrics['overall_score'] - config_b_metrics['overall_score']),
            'statistically_significant': significant,
            'p_value': p_value,
            'recommendation': self._generate_recommendation(config_a_metrics, config_b_metrics, significant)
        }
    
    def _generate_recommendation(self, 
                               config_a_metrics: Dict[str, Any],
                               config_b_metrics: Dict[str, Any],
                               significant: bool) -> str:
        """Generate recommendation based on A/B test results"""
        
        if not significant:
            return "No statistically significant difference found. More data may be needed."
        
        a_score = config_a_metrics['overall_score']
        b_score = config_b_metrics['overall_score']
        
        if a_score > b_score:
            improvement = ((a_score - b_score) / b_score) * 100
            return f"Configuration A is significantly better by {improvement:.1f}%. Recommend deploying A."
        else:
            improvement = ((b_score - a_score) / a_score) * 100
            return f"Configuration B is significantly better by {improvement:.1f}%. Recommend deploying B."


class ComprehensiveRAGEvaluator:
    """Main evaluation framework orchestrating all evaluation components"""
    
    def __init__(self):
        self.retrieval_evaluator = RetrievalEvaluator()
        self.generation_evaluator = GenerationEvaluator()
        self.rag_evaluator = RAGSpecificEvaluator()
        self.performance_evaluator = PerformanceEvaluator()
        self.ab_testing = ABTestingFramework()
        
        self.evaluation_history = []
        self.baseline_results = None
    
    def evaluate_rag_system(self, 
                           examples: List[EvaluationExample],
                           include_performance: bool = True) -> ComprehensiveEvaluationResult:
        """Perform comprehensive RAG system evaluation"""
        
        start_time = time.time()
        
        # Parallel evaluation of different components
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit evaluation tasks
            retrieval_future = executor.submit(
                self.retrieval_evaluator.evaluate_retrieval, examples
            )
            generation_future = executor.submit(
                self.generation_evaluator.evaluate_generation, examples
            )
            rag_future = executor.submit(
                self.rag_evaluator.evaluate_rag_quality, examples
            )
            
            # Wait for results
            retrieval_metrics = retrieval_future.result()
            generation_metrics = generation_future.result()
            rag_metrics = rag_future.result()
        
        # Performance metrics
        if include_performance:
            performance_metrics = self.performance_evaluator.get_performance_metrics()
        else:
            performance_metrics = SystemPerformanceResult()
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(
            retrieval_metrics, generation_metrics, rag_metrics, performance_metrics
        )
        
        # Create comprehensive result
        result = ComprehensiveEvaluationResult(
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            rag_metrics=rag_metrics,
            performance_metrics=performance_metrics,
            overall_score=overall_score,
            metadata={
                'evaluation_time_ms': (time.time() - start_time) * 1000,
                'examples_count': len(examples),
                'evaluation_timestamp': datetime.now().isoformat()
            }
        )
        
        # Store in history
        self.evaluation_history.append(result)
        
        return result
    
    def _calculate_overall_score(self, 
                               retrieval_metrics: RetrievalEvaluationResult,
                               generation_metrics: GenerationEvaluationResult,
                               rag_metrics: RAGEvaluationResult,
                               performance_metrics: SystemPerformanceResult) -> float:
        """Calculate weighted overall score"""
        
        # Weights for different components
        weights = {
            'retrieval': 0.3,
            'generation': 0.25,
            'rag': 0.35,
            'performance': 0.1
        }
        
        # Retrieval score (average of key metrics)
        retrieval_score = np.mean([
            retrieval_metrics.recall_at_5,
            retrieval_metrics.ndcg_at_5,
            retrieval_metrics.mrr
        ])
        
        # Generation score (average of key metrics)
        generation_score = np.mean([
            generation_metrics.rouge_l_f,
            generation_metrics.bleu_4,
            generation_metrics.bert_score_f1
        ])
        
        # RAG score
        rag_score = rag_metrics.overall_rag_score
        
        # Performance score (normalized, higher is better)
        performance_score = 1.0
        if performance_metrics.average_response_time_ms > 0:
            # Normalize response time (target: 1000ms)
            time_score = min(1.0, 1000.0 / performance_metrics.average_response_time_ms)
            error_score = 1.0 - performance_metrics.error_rate
            performance_score = (time_score + error_score) / 2
        
        # Calculate weighted average
        overall_score = (
            retrieval_score * weights['retrieval'] +
            generation_score * weights['generation'] +
            rag_score * weights['rag'] +
            performance_score * weights['performance']
        )
        
        return min(1.0, max(0.0, overall_score))
    
    def set_baseline(self, baseline_result: ComprehensiveEvaluationResult):
        """Set baseline results for comparison"""
        self.baseline_results = baseline_result
    
    def compare_with_baseline(self, current_result: ComprehensiveEvaluationResult) -> Dict[str, Any]:
        """Compare current results with baseline"""
        
        if not self.baseline_results:
            return {'error': 'No baseline set'}
        
        comparison = {
            'overall_improvement': current_result.overall_score - self.baseline_results.overall_score,
            'retrieval_improvements': {
                'recall_at_5': current_result.retrieval_metrics.recall_at_5 - self.baseline_results.retrieval_metrics.recall_at_5,
                'ndcg_at_5': current_result.retrieval_metrics.ndcg_at_5 - self.baseline_results.retrieval_metrics.ndcg_at_5,
                'mrr': current_result.retrieval_metrics.mrr - self.baseline_results.retrieval_metrics.mrr
            },
            'generation_improvements': {
                'rouge_l_f': current_result.generation_metrics.rouge_l_f - self.baseline_results.generation_metrics.rouge_l_f,
                'bleu_4': current_result.generation_metrics.bleu_4 - self.baseline_results.generation_metrics.bleu_4,
                'bert_score_f1': current_result.generation_metrics.bert_score_f1 - self.baseline_results.generation_metrics.bert_score_f1
            },
            'rag_improvements': {
                'groundedness': current_result.rag_metrics.groundedness_score - self.baseline_results.rag_metrics.groundedness_score,
                'faithfulness': current_result.rag_metrics.faithfulness_score - self.baseline_results.rag_metrics.faithfulness_score,
                'overall_rag': current_result.rag_metrics.overall_rag_score - self.baseline_results.rag_metrics.overall_rag_score
            },
            'performance_improvements': {
                'response_time': self.baseline_results.performance_metrics.average_response_time_ms - current_result.performance_metrics.average_response_time_ms,
                'error_rate': self.baseline_results.performance_metrics.error_rate - current_result.performance_metrics.error_rate
            }
        }
        
        return comparison
    
    def generate_evaluation_report(self, 
                                 result: ComprehensiveEvaluationResult,
                                 include_recommendations: bool = True) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        
        report = {
            'summary': {
                'overall_score': result.overall_score,
                'evaluation_timestamp': result.evaluation_timestamp.isoformat(),
                'examples_evaluated': result.metadata.get('examples_count', 0)
            },
            'detailed_metrics': {
                'retrieval': asdict(result.retrieval_metrics),
                'generation': asdict(result.generation_metrics),
                'rag_specific': asdict(result.rag_metrics),
                'performance': asdict(result.performance_metrics)
            }
        }
        
        # Add trend analysis if history available
        if len(self.evaluation_history) > 1:
            recent_scores = [r.overall_score for r in self.evaluation_history[-5:]]  # Last 5 evaluations
            report['trends'] = {
                'score_trend': 'improving' if recent_scores[-1] > recent_scores[0] else 'declining',
                'average_recent_score': np.mean(recent_scores),
                'score_volatility': np.std(recent_scores)
            }
        
        # Add recommendations
        if include_recommendations:
            report['recommendations'] = self._generate_recommendations(result)
        
        # Comparison with baseline if available
        if self.baseline_results:
            report['baseline_comparison'] = self.compare_with_baseline(result)
        
        return report
    
    def _generate_recommendations(self, result: ComprehensiveEvaluationResult) -> List[str]:
        """Generate improvement recommendations based on evaluation results"""
        
        recommendations = []
        
        # Retrieval recommendations
        if result.retrieval_metrics.recall_at_5 < 0.7:
            recommendations.append("Low retrieval recall detected. Consider improving embedding quality or expanding document corpus.")
        
        if result.retrieval_metrics.ndcg_at_5 < 0.6:
            recommendations.append("Poor ranking quality. Consider implementing reranking or improving relevance scoring.")
        
        # Generation recommendations
        if result.generation_metrics.rouge_l_f < 0.3:
            recommendations.append("Low generation quality scores. Consider fine-tuning the generation model or improving prompts.")
        
        # RAG-specific recommendations
        if result.rag_metrics.groundedness_score < 0.6:
            recommendations.append("Low groundedness score. Improve retrieval quality or add citation mechanisms.")
        
        if result.rag_metrics.faithfulness_score < 0.7:
            recommendations.append("Faithfulness concerns detected. Consider adding fact-checking or improving context utilization.")
        
        # Performance recommendations
        if result.performance_metrics.average_response_time_ms > 2000:
            recommendations.append("High response times detected. Consider caching, model optimization, or infrastructure scaling.")
        
        if result.performance_metrics.error_rate > 0.05:
            recommendations.append("High error rate detected. Investigate system stability and error handling.")
        
        return recommendations
    
    def export_results(self, filepath: str, results: List[ComprehensiveEvaluationResult] = None):
        """Export evaluation results to file"""
        
        if results is None:
            results = self.evaluation_history
        
        export_data = []
        for result in results:
            export_data.append({
                'timestamp': result.evaluation_timestamp.isoformat(),
                'overall_score': result.overall_score,
                'retrieval_metrics': asdict(result.retrieval_metrics),
                'generation_metrics': asdict(result.generation_metrics),
                'rag_metrics': asdict(result.rag_metrics),
                'performance_metrics': asdict(result.performance_metrics),
                'metadata': result.metadata
            })
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        if filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        elif filepath.endswith('.csv'):
            df = pd.json_normalize(export_data)
            df.to_csv(filepath, index=False)
        
        logger.info(f"Exported {len(export_data)} evaluation results to {filepath}")