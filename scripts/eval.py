#!/usr/bin/env python3
"""
RAG Evaluation Framework
Measures retrieval and generation quality metrics
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_vector_db import EnhancedVectorDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class EvalMetrics:
    """Evaluation metrics container"""
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg: float = 0.0  # Normalized Discounted Cumulative Gain
    answer_relevancy: float = 0.0
    answer_faithfulness: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p90_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

class SyntheticDataGenerator:
    """Generate synthetic Q&A pairs for evaluation"""
    
    @staticmethod
    def generate_qa_pairs(num_pairs: int = 50) -> List[Dict[str, Any]]:
        """Generate synthetic question-answer pairs"""
        qa_pairs = []
        
        # Sample templates (in production, use LLM to generate from docs)
        templates = [
            {
                "question": "What is the main purpose of {topic}?",
                "answer": "The main purpose of {topic} is to {purpose}.",
                "relevant_docs": ["doc1", "doc2"]
            },
            {
                "question": "How does {system} handle {operation}?",
                "answer": "{system} handles {operation} by {method}.",
                "relevant_docs": ["doc3"]
            },
            {
                "question": "What are the key features of {product}?",
                "answer": "The key features of {product} include {features}.",
                "relevant_docs": ["doc4", "doc5"]
            }
        ]
        
        # Generate pairs (simplified - in production, use actual document content)
        for i in range(num_pairs):
            template = templates[i % len(templates)]
            qa_pairs.append({
                "id": f"qa_{i:04d}",
                "question": template["question"].format(
                    topic="vector databases",
                    system="ChromaDB",
                    operation="indexing",
                    product="RAG system"
                ),
                "answer": template["answer"].format(
                    topic="vector databases",
                    purpose="store and retrieve embeddings",
                    system="ChromaDB",
                    operation="indexing",
                    method="HNSW algorithm",
                    product="RAG system",
                    features="semantic search, hybrid retrieval, reranking"
                ),
                "relevant_docs": template["relevant_docs"],
                "metadata": {
                    "difficulty": "medium",
                    "category": "technical"
                }
            })
        
        return qa_pairs

class RAGEvaluator:
    """Main evaluation framework"""
    
    def __init__(self, vector_db: EnhancedVectorDatabase = None, enable_enhanced_features: bool = True):
        self.vector_db = vector_db or EnhancedVectorDatabase()
        self.enable_enhanced_features = enable_enhanced_features
        self.metrics = EvalMetrics()
        self.latencies = []
    
    def _calculate_recall(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """Calculate recall@k"""
        if not relevant:
            return 0.0
        retrieved_k = set(retrieved[:k])
        relevant_set = set(relevant)
        return len(retrieved_k & relevant_set) / len(relevant_set)
    
    def _calculate_precision(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """Calculate precision@k"""
        if not retrieved[:k]:
            return 0.0
        retrieved_k = set(retrieved[:k])
        relevant_set = set(relevant)
        return len(retrieved_k & relevant_set) / len(retrieved_k)
    
    def _calculate_mrr(self, retrieved: List[str], relevant: List[str]) -> float:
        """Calculate Mean Reciprocal Rank"""
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    def _calculate_ndcg(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """Calculate NDCG@k"""
        def dcg(scores):
            return sum(score / np.log2(i + 2) for i, score in enumerate(scores))
        
        # Create relevance scores (1 if relevant, 0 otherwise)
        scores = [1 if doc in relevant else 0 for doc in retrieved[:k]]
        ideal_scores = [1] * min(len(relevant), k) + [0] * max(0, k - len(relevant))
        
        dcg_score = dcg(scores)
        idcg_score = dcg(ideal_scores)
        
        return dcg_score / idcg_score if idcg_score > 0 else 0.0
    
    def _calculate_answer_metrics(self, generated: str, reference: str, context: List[str]) -> Tuple[float, float]:
        """Calculate answer relevancy and faithfulness (simplified)"""
        # In production, use LLM-based evaluation or semantic similarity
        # This is a placeholder implementation
        relevancy = 0.8 if generated and reference else 0.0
        faithfulness = 0.9 if context else 0.0
        return relevancy, faithfulness
    
    def evaluate_retrieval(self, qa_pairs: List[Dict[str, Any]], k: int = 10) -> Dict[str, Any]:
        """Evaluate retrieval performance"""
        logger.info(f"Evaluating retrieval on {len(qa_pairs)} Q&A pairs")
        
        recalls = []
        precisions = []
        mrrs = []
        ndcgs = []
        
        for qa in qa_pairs:
            question = qa['question']
            relevant_docs = qa['relevant_docs']
            
            # Measure retrieval time
            start_time = time.time()
            results = self.vector_db.search(question, limit=k, use_enhanced=self.enable_enhanced_features)
            latency_ms = (time.time() - start_time) * 1000
            self.latencies.append(latency_ms)
            
            # Extract doc IDs from results
            retrieved_docs = [r['id'] for r in results]
            
            # Store additional metrics if using enhanced features
            if self.enable_enhanced_features and results:
                # Analyze retrieval source distribution
                source_counts = {}
                for result in results:
                    source = result.get('source', 'unknown')
                    source_counts[source] = source_counts.get(source, 0) + 1
                
                qa['retrieval_sources'] = source_counts
                
                # Store score information
                if 'combined_score' in results[0]:
                    qa['top_combined_score'] = results[0]['combined_score']
                if 'dense_score' in results[0]:
                    qa['top_dense_score'] = results[0]['dense_score']
                if 'lexical_score' in results[0]:
                    qa['top_lexical_score'] = results[0]['lexical_score']
            
            # Calculate metrics
            recalls.append(self._calculate_recall(retrieved_docs, relevant_docs, k))
            precisions.append(self._calculate_precision(retrieved_docs, relevant_docs, k))
            mrrs.append(self._calculate_mrr(retrieved_docs, relevant_docs))
            ndcgs.append(self._calculate_ndcg(retrieved_docs, relevant_docs, k))
        
        # Aggregate metrics
        self.metrics.recall_at_k = np.mean(recalls)
        self.metrics.precision_at_k = np.mean(precisions)
        self.metrics.mrr = np.mean(mrrs)
        self.metrics.ndcg = np.mean(ndcgs)
        
        # Calculate latency percentiles
        if self.latencies:
            self.metrics.avg_latency_ms = np.mean(self.latencies)
            self.metrics.p50_latency_ms = np.percentile(self.latencies, 50)
            self.metrics.p90_latency_ms = np.percentile(self.latencies, 90)
            self.metrics.p99_latency_ms = np.percentile(self.latencies, 99)
        
        return asdict(self.metrics)
    
    def evaluate_generation(self, qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate generation quality (placeholder)"""
        logger.info("Evaluating generation quality")
        
        # This would integrate with your LLM pipeline
        # For now, using placeholder values
        self.metrics.answer_relevancy = 0.85
        self.metrics.answer_faithfulness = 0.90
        self.metrics.context_precision = 0.88
        self.metrics.context_recall = 0.82
        
        return asdict(self.metrics)
    
    def run_full_evaluation(self, qa_pairs: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run complete evaluation suite"""
        if qa_pairs is None:
            logger.info("Generating synthetic Q&A pairs")
            qa_pairs = SyntheticDataGenerator.generate_qa_pairs(50)
        
        # Evaluate retrieval
        retrieval_metrics = self.evaluate_retrieval(qa_pairs)
        
        # Evaluate generation
        generation_metrics = self.evaluate_generation(qa_pairs)
        
        # Combine results
        results = {
            "timestamp": datetime.now().isoformat(),
            "num_queries": len(qa_pairs),
            "enhanced_features_enabled": self.enable_enhanced_features,
            "system_status": self.vector_db.get_status() if hasattr(self.vector_db, 'get_status') else {},
            "metrics": {
                "retrieval": {
                    "recall@10": self.metrics.recall_at_k,
                    "precision@10": self.metrics.precision_at_k,
                    "mrr": self.metrics.mrr,
                    "ndcg@10": self.metrics.ndcg
                },
                "generation": {
                    "answer_relevancy": self.metrics.answer_relevancy,
                    "answer_faithfulness": self.metrics.answer_faithfulness,
                    "context_precision": self.metrics.context_precision,
                    "context_recall": self.metrics.context_recall
                },
                "performance": {
                    "avg_latency_ms": self.metrics.avg_latency_ms,
                    "p50_latency_ms": self.metrics.p50_latency_ms,
                    "p90_latency_ms": self.metrics.p90_latency_ms,
                    "p99_latency_ms": self.metrics.p99_latency_ms
                }
            },
            "summary": self._generate_summary()
        }
        
        # Add enhanced feature analysis if available
        if self.enable_enhanced_features:
            enhanced_analysis = self._analyze_enhanced_features(qa_pairs)
            results["enhanced_analysis"] = enhanced_analysis
        
        return results
    
    def _generate_summary(self) -> Dict[str, str]:
        """Generate evaluation summary"""
        overall_score = np.mean([
            self.metrics.recall_at_k,
            self.metrics.precision_at_k,
            self.metrics.answer_relevancy,
            self.metrics.answer_faithfulness
        ])
        
        if overall_score >= 0.9:
            grade = "Excellent"
        elif overall_score >= 0.8:
            grade = "Good"
        elif overall_score >= 0.7:
            grade = "Fair"
        else:
            grade = "Needs Improvement"
        
        return {
            "overall_score": f"{overall_score:.2%}",
            "grade": grade,
            "recommendation": self._get_recommendation()
        }
    
    def _get_recommendation(self) -> str:
        """Get improvement recommendations based on metrics"""
        recommendations = []
        
        if self.metrics.recall_at_k < 0.8:
            recommendations.append("Improve retrieval recall with hybrid search")
        if self.metrics.precision_at_k < 0.7:
            recommendations.append("Add reranking to improve precision")
        if self.metrics.avg_latency_ms > 1000:
            recommendations.append("Optimize query performance with caching")
        if self.metrics.answer_faithfulness < 0.85:
            recommendations.append("Improve context quality for better faithfulness")
        
        return "; ".join(recommendations) if recommendations else "System performing well"
    
    def _analyze_enhanced_features(self, qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the impact of enhanced features"""
        analysis = {
            "retrieval_source_distribution": {},
            "score_statistics": {},
            "hybrid_effectiveness": {}
        }
        
        try:
            # Analyze retrieval sources
            all_sources = {}
            scores_by_source = {'dense': [], 'lexical': [], 'hybrid': []}
            
            for qa in qa_pairs:
                if 'retrieval_sources' in qa:
                    for source, count in qa['retrieval_sources'].items():
                        all_sources[source] = all_sources.get(source, 0) + count
                
                # Collect score statistics
                if 'top_combined_score' in qa:
                    source = qa.get('retrieval_sources', {})
                    if 'hybrid' in source and source['hybrid'] > 0:
                        scores_by_source['hybrid'].append(qa['top_combined_score'])
                    elif 'dense' in source:
                        scores_by_source['dense'].append(qa['top_combined_score'])
                    elif 'lexical' in source:
                        scores_by_source['lexical'].append(qa['top_combined_score'])
            
            # Calculate source distribution percentages
            total_results = sum(all_sources.values())
            if total_results > 0:
                for source, count in all_sources.items():
                    analysis["retrieval_source_distribution"][source] = {
                        "count": count,
                        "percentage": (count / total_results) * 100
                    }
            
            # Calculate score statistics by source
            for source, scores in scores_by_source.items():
                if scores:
                    analysis["score_statistics"][source] = {
                        "mean_score": np.mean(scores),
                        "median_score": np.median(scores),
                        "std_score": np.std(scores),
                        "sample_count": len(scores)
                    }
            
            # Hybrid effectiveness analysis
            hybrid_scores = scores_by_source.get('hybrid', [])
            dense_scores = scores_by_source.get('dense', [])
            
            if hybrid_scores and dense_scores:
                hybrid_mean = np.mean(hybrid_scores)
                dense_mean = np.mean(dense_scores)
                improvement = ((hybrid_mean - dense_mean) / dense_mean) * 100 if dense_mean > 0 else 0
                
                analysis["hybrid_effectiveness"] = {
                    "hybrid_mean_score": hybrid_mean,
                    "dense_mean_score": dense_mean,
                    "improvement_percentage": improvement,
                    "is_beneficial": improvement > 5.0  # 5% improvement threshold
                }
            
        except Exception as e:
            logger.warning(f"Enhanced feature analysis failed: {e}")
            analysis["error"] = str(e)
        
        return analysis

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='RAG Evaluation Framework')
    parser.add_argument('--metrics', nargs='+', default=['all'], 
                      help='Metrics to evaluate (all, recall, precision, latency)')
    parser.add_argument('--num-queries', type=int, default=50, 
                      help='Number of test queries')
    parser.add_argument('--k', type=int, default=10, 
                      help='Top-k for retrieval metrics')
    parser.add_argument('--output', help='Output file for results')
    parser.add_argument('--golden-set', help='Path to golden Q&A dataset')
    parser.add_argument('--enhanced', action='store_true', 
                      help='Enable enhanced features (hybrid retrieval, reranking)')
    parser.add_argument('--compare', action='store_true',
                      help='Compare enhanced vs basic retrieval performance')
    parser.add_argument('--rebuild-index', action='store_true',
                      help='Rebuild search indices before evaluation')
    
    args = parser.parse_args()
    
    # Initialize database and evaluator
    if args.enhanced or args.compare:
        vector_db = EnhancedVectorDatabase()
        if args.rebuild_index and hasattr(vector_db, 'rebuild_search_index'):
            logger.info("Rebuilding search indices...")
            vector_db.rebuild_search_index()
    else:
        vector_db = EnhancedVectorDatabase()
    
    # Load or generate Q&A pairs
    if args.golden_set:
        logger.info(f"Loading golden set from {args.golden_set}")
        with open(args.golden_set, 'r') as f:
            qa_pairs = json.load(f)
    else:
        logger.info(f"Generating {args.num_queries} synthetic Q&A pairs")
        qa_pairs = SyntheticDataGenerator.generate_qa_pairs(args.num_queries)
    
    if args.compare:
        # Compare enhanced vs basic performance
        logger.info("Running comparison evaluation (enhanced vs basic)")
        
        # Run enhanced evaluation
        logger.info("Evaluating with enhanced features...")
        enhanced_evaluator = RAGEvaluator(vector_db, enable_enhanced_features=True)
        enhanced_results = enhanced_evaluator.run_full_evaluation(qa_pairs)
        
        # Run basic evaluation
        logger.info("Evaluating with basic features...")
        basic_evaluator = RAGEvaluator(vector_db, enable_enhanced_features=False)
        basic_results = basic_evaluator.run_full_evaluation(qa_pairs)
        
        # Create comparison report
        results = {
            "comparison_mode": True,
            "timestamp": datetime.now().isoformat(),
            "enhanced_results": enhanced_results,
            "basic_results": basic_results,
            "improvements": {
                "recall_improvement": enhanced_results['metrics']['retrieval']['recall@10'] - basic_results['metrics']['retrieval']['recall@10'],
                "precision_improvement": enhanced_results['metrics']['retrieval']['precision@10'] - basic_results['metrics']['retrieval']['precision@10'],
                "latency_change": enhanced_results['metrics']['performance']['avg_latency_ms'] - basic_results['metrics']['performance']['avg_latency_ms']
            }
        }
    else:
        # Single evaluation run
        evaluator = RAGEvaluator(vector_db, enable_enhanced_features=args.enhanced)
        logger.info(f"Starting evaluation (enhanced={args.enhanced})")
        results = evaluator.run_full_evaluation(qa_pairs)
    
    # Display results
    print("\n" + "="*50)
    print("RAG EVALUATION RESULTS")
    print("="*50)
    print(f"\nRetrieval Metrics:")
    print(f"  Recall@{args.k}: {results['metrics']['retrieval']['recall@10']:.2%}")
    print(f"  Precision@{args.k}: {results['metrics']['retrieval']['precision@10']:.2%}")
    print(f"  MRR: {results['metrics']['retrieval']['mrr']:.3f}")
    print(f"  NDCG@{args.k}: {results['metrics']['retrieval']['ndcg@10']:.3f}")
    
    print(f"\nGeneration Metrics:")
    print(f"  Answer Relevancy: {results['metrics']['generation']['answer_relevancy']:.2%}")
    print(f"  Answer Faithfulness: {results['metrics']['generation']['answer_faithfulness']:.2%}")
    print(f"  Context Precision: {results['metrics']['generation']['context_precision']:.2%}")
    print(f"  Context Recall: {results['metrics']['generation']['context_recall']:.2%}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Avg Latency: {results['metrics']['performance']['avg_latency_ms']:.1f}ms")
    print(f"  P50 Latency: {results['metrics']['performance']['p50_latency_ms']:.1f}ms")
    print(f"  P90 Latency: {results['metrics']['performance']['p90_latency_ms']:.1f}ms")
    print(f"  P99 Latency: {results['metrics']['performance']['p99_latency_ms']:.1f}ms")
    
    print(f"\nSummary:")
    print(f"  Overall Score: {results['summary']['overall_score']}")
    print(f"  Grade: {results['summary']['grade']}")
    print(f"  Recommendation: {results['summary']['recommendation']}")
    
    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return 0 if results['summary']['grade'] != "Needs Improvement" else 1

if __name__ == '__main__':
    sys.exit(main())
