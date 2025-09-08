#!/usr/bin/env python3
"""
Performance Benchmarking Script
Measures latency, throughput, and resource usage
"""

import os
import sys
import json
import time
import argparse
import logging
import concurrent.futures
from typing import List, Dict, Any
from datetime import datetime
import psutil
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_db import VectorDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BenchmarkRunner:
    """Performance benchmark runner"""
    
    def __init__(self):
        self.vector_db = VectorDatabase()
        self.results = {
            'latencies': [],
            'throughput': [],
            'resource_usage': [],
            'errors': []
        }
    
    def generate_test_queries(self, num_queries: int) -> List[str]:
        """Generate test queries"""
        queries = [
            "What is vector database?",
            "How does ChromaDB work?",
            "Explain retrieval augmented generation",
            "What are embeddings?",
            "How to optimize search performance?",
            "What is semantic search?",
            "Explain HNSW algorithm",
            "What is cosine similarity?",
            "How to implement RAG?",
            "What are the best practices for chunking?"
        ]
        
        # Repeat and extend queries as needed
        extended_queries = []
        for i in range(num_queries):
            query = queries[i % len(queries)]
            # Add variation to prevent caching effects
            extended_queries.append(f"{query} (test {i})")
        
        return extended_queries
    
    def benchmark_single_query(self, query: str) -> Dict[str, Any]:
        """Benchmark a single query"""
        # Record start metrics
        start_time = time.time()
        start_cpu = psutil.cpu_percent(interval=None)
        start_memory = psutil.virtual_memory().percent
        
        try:
            # Execute query
            results = self.vector_db.search(query, limit=10)
            
            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000
            end_cpu = psutil.cpu_percent(interval=None)
            end_memory = psutil.virtual_memory().percent
            
            return {
                'success': True,
                'latency_ms': latency_ms,
                'cpu_delta': end_cpu - start_cpu,
                'memory_delta': end_memory - start_memory,
                'num_results': len(results)
            }
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'latency_ms': (time.time() - start_time) * 1000
            }
    
    def run_sequential_benchmark(self, queries: List[str]) -> Dict[str, Any]:
        """Run queries sequentially"""
        logger.info(f"Running sequential benchmark with {len(queries)} queries")
        
        start_time = time.time()
        for i, query in enumerate(queries):
            result = self.benchmark_single_query(query)
            if result['success']:
                self.results['latencies'].append(result['latency_ms'])
            else:
                self.results['errors'].append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1}/{len(queries)} queries")
        
        total_time = time.time() - start_time
        throughput = len(queries) / total_time
        
        return {
            'total_time_s': total_time,
            'throughput_qps': throughput,
            'queries_processed': len(queries),
            'errors': len(self.results['errors'])
        }
    
    def run_concurrent_benchmark(self, queries: List[str], concurrency: int) -> Dict[str, Any]:
        """Run queries concurrently"""
        logger.info(f"Running concurrent benchmark with {len(queries)} queries, concurrency={concurrency}")
        
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(self.benchmark_single_query, query) for query in queries]
            
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                result = future.result()
                if result['success']:
                    self.results['latencies'].append(result['latency_ms'])
                else:
                    self.results['errors'].append(result)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{len(queries)} queries")
        
        total_time = time.time() - start_time
        throughput = len(queries) / total_time
        
        return {
            'total_time_s': total_time,
            'throughput_qps': throughput,
            'queries_processed': len(queries),
            'errors': len(self.results['errors']),
            'concurrency': concurrency
        }
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate benchmark statistics"""
        if not self.results['latencies']:
            return {'error': 'No successful queries'}
        
        latencies = np.array(self.results['latencies'])
        
        return {
            'latency': {
                'min_ms': float(np.min(latencies)),
                'max_ms': float(np.max(latencies)),
                'mean_ms': float(np.mean(latencies)),
                'median_ms': float(np.median(latencies)),
                'std_ms': float(np.std(latencies)),
                'p50_ms': float(np.percentile(latencies, 50)),
                'p90_ms': float(np.percentile(latencies, 90)),
                'p95_ms': float(np.percentile(latencies, 95)),
                'p99_ms': float(np.percentile(latencies, 99))
            },
            'success_rate': (len(self.results['latencies']) / 
                           (len(self.results['latencies']) + len(self.results['errors']))) * 100
        }
    
    def generate_report(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate benchmark report"""
        stats = self.calculate_statistics()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'configuration': config,
            'results': {
                'statistics': stats,
                'throughput_qps': config.get('throughput_qps', 0),
                'total_queries': len(self.results['latencies']) + len(self.results['errors']),
                'successful_queries': len(self.results['latencies']),
                'failed_queries': len(self.results['errors'])
            },
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'python_version': sys.version
            }
        }

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Performance Benchmark')
    parser.add_argument('--queries', type=int, default=100, help='Number of queries')
    parser.add_argument('--concurrency', type=int, default=1, help='Concurrent workers')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup queries')
    parser.add_argument('--output', help='Output file for results')
    
    args = parser.parse_args()
    
    # Initialize benchmark runner
    runner = BenchmarkRunner()
    
    # Generate test queries
    test_queries = runner.generate_test_queries(args.queries)
    warmup_queries = runner.generate_test_queries(args.warmup)
    
    # Run warmup
    if args.warmup > 0:
        logger.info(f"Running {args.warmup} warmup queries")
        for query in warmup_queries:
            runner.benchmark_single_query(query)
        runner.results = {'latencies': [], 'throughput': [], 'resource_usage': [], 'errors': []}
    
    # Run benchmark
    if args.concurrency > 1:
        config = runner.run_concurrent_benchmark(test_queries, args.concurrency)
    else:
        config = runner.run_sequential_benchmark(test_queries)
    
    # Generate report
    report = runner.generate_report(config)
    
    # Display results
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    print(f"\nConfiguration:")
    print(f"  Total Queries: {args.queries}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Warmup Queries: {args.warmup}")
    
    print(f"\nPerformance:")
    print(f"  Throughput: {config['throughput_qps']:.2f} QPS")
    print(f"  Total Time: {config['total_time_s']:.2f}s")
    print(f"  Success Rate: {report['results']['statistics']['success_rate']:.1f}%")
    
    print(f"\nLatency Statistics:")
    latency_stats = report['results']['statistics']['latency']
    print(f"  Min: {latency_stats['min_ms']:.1f}ms")
    print(f"  Mean: {latency_stats['mean_ms']:.1f}ms")
    print(f"  Median: {latency_stats['median_ms']:.1f}ms")
    print(f"  P90: {latency_stats['p90_ms']:.1f}ms")
    print(f"  P95: {latency_stats['p95_ms']:.1f}ms")
    print(f"  P99: {latency_stats['p99_ms']:.1f}ms")
    print(f"  Max: {latency_stats['max_ms']:.1f}ms")
    
    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # Return non-zero if performance is poor
    return 0 if latency_stats['p99_ms'] < 2000 else 1

if __name__ == '__main__':
    sys.exit(main())
