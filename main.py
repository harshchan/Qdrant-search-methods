#!/usr/bin/env python3
"""
Main execution script for the e-commerce search engine with Qdrant
"""

import os
import sys
import argparse
from data_preprocessing import load_and_preprocess_data
from qdrant_setup import QdrantManager
from data_indexing import DataIndexer
from evaluation import SearchEvaluator

def setup_environment():
    """Setup the environment and verify requirements"""
    print("Setting up environment...")
    
    # Check if Qdrant is running
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url="http://localhost:6333")
        client.get_collections()
        print("✓ Qdrant is running")
    except Exception as e:
        print(f"✗ Qdrant is not accessible: {e}")
        print("Please start Qdrant with: docker run -p 6333:6333 qdrant/qdrant")
        return False
    
    # Check if data file exists
    if not os.path.exists("data/shoe_products_dataset.csv"):
        print("✗ Data file not found: data/shoe_products_dataset.csv")
        return False
    
    print("✓ Data file found")
    return True

def run_full_pipeline():
    """Run the complete pipeline"""
    print("\n" + "="*60)
    print("RUNNING COMPLETE SEARCH ENGINE PIPELINE")
    print("="*60)
    
    # Step 1: Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    df = load_and_preprocess_data("data/shoe_products_dataset.csv")
    print(f"   Loaded {len(df)} products")
    
    # Use subset for testing (remove this for production)
    df = df.head(1000)
    print(f"   Using subset of {len(df)} products for testing")
    
    # Step 2: Setup Qdrant collections
    print("\n2. Setting up Qdrant collections...")
    qdrant_manager = QdrantManager()
    qdrant_manager.create_individual_collections(dense_dim=384)
    
    # Step 3: Index data
    print("\n3. Indexing data...")
    indexer = DataIndexer()
    indexer.index_all_collections(df)
    
    # Step 4: Run evaluation
    print("\n4. Running evaluation...")
    evaluator = SearchEvaluator()
    
    test_queries = [
        "running shoes for men",
        "women's athletic sneakers",
        "basketball shoes high top",
        "comfortable walking shoes"
    ]
    
    evaluation_results = evaluator.evaluate_search_methods(test_queries, limit=5)
    analysis = evaluator.analyze_results(evaluation_results)
    
    # Generate and save report
    report = evaluator.generate_report(evaluation_results, analysis)
    with open('evaluation_report.txt', 'w') as f:
        f.write(report)
    
    print("\n5. Pipeline completed successfully!")
    print(f"   Evaluation report saved to 'evaluation_report.txt'")
    
    # Print summary
    print("\n   Performance Summary:")
    for method in ['tfidf', 'bm25', 'splade', 'dense', 'hybrid']:
        avg_time = analysis['average_times'].get(method, 0)
        success_rate = analysis['success_rates'].get(method, 0)
        print(f"   {method.upper()}: {avg_time:.4f}s avg, {success_rate:.2%} success")

def run_interactive_search():
    """Run interactive search mode"""
    print("\n" + "="*60)
    print("INTERACTIVE SEARCH MODE")
    print("="*60)
    print("Type 'exit' to quit, 'help' for commands")
    
    from search_engine import SearchEngine
    engine = SearchEngine()
    
    while True:
        try:
            query = input("\nEnter search query: ").strip()
            
            if query.lower() == 'exit':
                break
            elif query.lower() == 'help':
                print("\nCommands:")
                print("  exit - Exit the program")
                print("  help - Show this help message")
                print("  <query> - Search using all methods")
                continue
            elif not query:
                continue
            
            print(f"\nSearching for: '{query}'")
            print("-" * 40)
            
            results = engine.compare_search_methods(query, limit=3)
            
            for method, method_results in results.items():
                print(f"\n{method.upper()}:")
                if isinstance(method_results, dict) and 'error' in method_results:
                    print(f"  Error: {method_results['error']}")
                else:
                    for i, result in enumerate(method_results, 1):
                        title = result['payload'].get('title', 'N/A')
                        score = result['score']
                        print(f"  {i}. {title} (score: {score:.4f})")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="E-commerce Search Engine with Qdrant")
    parser.add_argument('--mode', choices=['pipeline', 'search', 'streamlit'], 
                       default='pipeline', help='Execution mode')
    parser.add_argument('--setup', action='store_true', 
                       help='Only setup environment and collections')
    
    args = parser.parse_args()
    
    if not setup_environment():
        sys.exit(1)
    
    if args.mode == 'pipeline':
        if args.setup:
            # Only setup collections
            print("\nSetting up Qdrant collections...")
            qdrant_manager = QdrantManager()
            qdrant_manager.create_individual_collections(dense_dim=384)
            print("Collections setup complete!")
        else:
            run_full_pipeline()
    elif args.mode == 'search':
        run_interactive_search()
    elif args.mode == 'streamlit':
        print("Starting Streamlit app...")
        os.system("streamlit run app.py")

if __name__ == "__main__":
    main()