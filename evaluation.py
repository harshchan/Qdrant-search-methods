import pandas as pd
from typing import List, Dict, Any, Tuple
from search_engine import SearchEngine
import time
import json
from collections import defaultdict

class SearchEvaluator:
    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        self.engine = SearchEngine(qdrant_url)
    
    def evaluate_search_methods(self, test_queries: List[str], limit: int = 10) -> Dict[str, Dict[str, Any]]:
        """Evaluate all search methods on test queries"""
        evaluation_results = {}
        
        for query in test_queries:
            print(f"Evaluating query: '{query}'")
            
            # Time and collect results for each method
            method_results = {}
            for method in ['tfidf', 'bm25', 'splade', 'dense', 'hybrid']:
                start_time = time.time()
                try:
                    results = getattr(self.engine, f'search_{method}')(query, limit=limit)
                    end_time = time.time()
                    
                    method_results[method] = {
                        'results': results,
                        'time': end_time - start_time,
                        'success': True
                    }
                except Exception as e:
                    end_time = time.time()
                    method_results[method] = {
                        'error': str(e),
                        'time': end_time - start_time,
                        'success': False
                    }
            
            evaluation_results[query] = method_results
        
        return evaluation_results
    
    def analyze_results(self, evaluation_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze and compare search method performance"""
        analysis = {
            'performance_metrics': defaultdict(dict),
            'result_overlap': defaultdict(dict),
            'average_times': defaultdict(float),
            'success_rates': defaultdict(float)
        }
        
        queries = list(evaluation_results.keys())
        methods = ['tfidf', 'bm25', 'splade', 'dense', 'hybrid']
        
        # Calculate metrics for each method
        for method in methods:
            total_time = 0
            successful_queries = 0
            
            for query in queries:
                if method in evaluation_results[query]:
                    result = evaluation_results[query][method]
                    
                    if result['success']:
                        total_time += result['time']
                        successful_queries += 1
                        
                        # Analyze result quality (simplified)
                        results = result['results']
                        if isinstance(results, list):
                            analysis['performance_metrics'][method][query] = {
                                'num_results': len(results),
                                'avg_score': sum(r.get('score', 0) for r in results) / len(results) if results else 0,
                                'top_score': results[0].get('score', 0) if results else 0
                            }
            
            analysis['average_times'][method] = total_time / len(queries) if queries else 0
            analysis['success_rates'][method] = successful_queries / len(queries) if queries else 0
        
        # Calculate result overlap between methods
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                overlap_count = 0
                total_comparisons = 0
                
                for query in queries:
                    if (method1 in evaluation_results[query] and 
                        method2 in evaluation_results[query] and
                        evaluation_results[query][method1]['success'] and
                        evaluation_results[query][method2]['success']):
                        
                        results1 = evaluation_results[query][method1]['results']
                        results2 = evaluation_results[query][method2]['results']
                        
                        if isinstance(results1, list) and isinstance(results2, list):
                            ids1 = {r['id'] for r in results1}
                            ids2 = {r['id'] for r in results2}
                            
                            overlap = len(ids1.intersection(ids2))
                            overlap_count += overlap
                            total_comparisons += min(len(ids1), len(ids2))
                
                if total_comparisons > 0:
                    analysis['result_overlap'][f'{method1}_vs_{method2}'] = overlap_count / total_comparisons
        
        return dict(analysis)
    
    def generate_report(self, evaluation_results: Dict[str, Dict[str, Any]], 
                      analysis: Dict[str, Any]) -> str:
        """Generate a comprehensive evaluation report"""
        report = []
        report.append("SEARCH METHOD EVALUATION REPORT")
        report.append("=" * 50)
        
        # Performance summary
        report.append("\nPERFORMANCE SUMMARY:")
        report.append("-" * 30)
        for method in ['tfidf', 'bm25', 'splade', 'dense', 'hybrid']:
            avg_time = analysis['average_times'].get(method, 0)
            success_rate = analysis['success_rates'].get(method, 0)
            report.append(f"{method.upper()}:")
            report.append(f"  Average Time: {avg_time:.4f}s")
            report.append(f"  Success Rate: {success_rate:.2%}")
        
        # Result overlap analysis
        report.append("\nRESULT OVERLAP ANALYSIS:")
        report.append("-" * 30)
        for comparison, overlap in analysis['result_overlap'].items():
            report.append(f"{comparison}: {overlap:.2%} overlap")
        
        # Detailed query results
        report.append("\nDETAILED QUERY RESULTS:")
        report.append("-" * 30)
        for query, methods in evaluation_results.items():
            report.append(f"\nQuery: '{query}'")
            for method, result in methods.items():
                if result['success']:
                    results = result['results']
                    if isinstance(results, list):
                        report.append(f"  {method.upper()}: {len(results)} results in {result['time']:.4f}s")
                        if results:
                            report.append(f"    Top result: {results[0]['payload'].get('title', 'N/A')}")
                else:
                    report.append(f"  {method.upper()}: FAILED - {result.get('error', 'Unknown error')}")
        
        return "\n".join(report)

if __name__ == "__main__":
    # Test queries for evaluation
    test_queries = [
        "running shoes for men",
        "women's athletic sneakers",
        "basketball shoes high top",
        "comfortable walking shoes",
        "formal dress shoes",
        "kids school shoes",
        "hiking boots waterproof",
        "tennis court shoes"
    ]
    
    evaluator = SearchEvaluator()
    
    print("Running evaluation...")
    evaluation_results = evaluator.evaluate_search_methods(test_queries, limit=5)
    
    print("Analyzing results...")
    analysis = evaluator.analyze_results(evaluation_results)
    
    print("Generating report...")
    report = evaluator.generate_report(evaluation_results, analysis)
    
    # Save report to file
    with open('evaluation_report.txt', 'w') as f:
        f.write(report)
    
    print("Evaluation complete! Report saved to 'evaluation_report.txt'")
    print("\nSummary:")
    print("-" * 30)
    for method in ['tfidf', 'bm25', 'splade', 'dense', 'hybrid']:
        avg_time = analysis['average_times'].get(method, 0)
        success_rate = analysis['success_rates'].get(method, 0)
        print(f"{method.upper()}: {avg_time:.4f}s avg, {success_rate:.2%} success")