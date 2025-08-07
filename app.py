import streamlit as st
import pandas as pd
from search_engine import SearchEngine
from evaluation import SearchEvaluator
import json

# Initialize search engine
@st.cache_resource
def init_search_engine():
    return SearchEngine()

def main():
    st.title("E-commerce Search Engine Comparison")
    st.sidebar.header("Search Configuration")
    
    # Initialize search engine
    engine = init_search_engine()
    
    # Search interface
    query = st.text_input("Enter your search query:", value="running shoes")
    search_method = st.sidebar.selectbox(
        "Search Method:",
        ["tfidf", "bm25", "splade", "dense", "hybrid", "compare_all"]
    )
    limit = st.sidebar.slider("Number of results:", 1, 20, 10)
    
    if st.button("Search"):
        if query:
            with st.spinner(f"Searching with {search_method}..."):
                try:
                    if search_method == "compare_all":
                        results = engine.compare_search_methods(query, limit=limit)
                        
                        # Display results for each method
                        for method, method_results in results.items():
                            st.subheader(f"{method.upper()} Results")
                            
                            if isinstance(method_results, dict) and 'error' in method_results:
                                st.error(f"Error: {method_results['error']}")
                            else:
                                for i, result in enumerate(method_results, 1):
                                    with st.expander(f"Result {i} (Score: {result['score']:.4f})"):
                                        st.write(f"**Title:** {result['payload'].get('title', 'N/A')}")
                                        st.write(f"**Description:** {result['payload'].get('description', 'N/A')}")
                                        st.write(f"**Attributes:** {result['payload'].get('attributes', 'N/A')}")
                    else:
                        results = getattr(engine, f'search_{search_method}')(query, limit=limit)
                        
                        st.subheader(f"{search_method.upper()} Results")
                        for i, result in enumerate(results, 1):
                            with st.expander(f"Result {i} (Score: {result['score']:.4f})"):
                                st.write(f"**Title:** {result['payload'].get('title', 'N/A')}")
                                st.write(f"**Description:** {result['payload'].get('description', 'N/A')}")
                                st.write(f"**Attributes:** {result['payload'].get('attributes', 'N/A')}")
                
                except Exception as e:
                    st.error(f"Search failed: {str(e)}")
    
    # Evaluation section
    st.sidebar.header("Evaluation")
    if st.sidebar.button("Run Evaluation"):
        test_queries = [
            "running shoes for men",
            "women's athletic sneakers",
            "basketball shoes high top",
            "comfortable walking shoes"
        ]
        
        with st.spinner("Running evaluation..."):
            evaluator = SearchEvaluator()
            evaluation_results = evaluator.evaluate_search_methods(test_queries, limit=5)
            analysis = evaluator.analyze_results(evaluation_results)
            
            # Display evaluation results
            st.subheader("Evaluation Results")
            
            # Performance metrics
            st.write("### Performance Metrics")
            metrics_df = pd.DataFrame({
                'Method': ['tfidf', 'bm25', 'splade', 'dense', 'hybrid'],
                'Avg Time (s)': [analysis['average_times'].get(m, 0) for m in ['tfidf', 'bm25', 'splade', 'dense', 'hybrid']],
                'Success Rate': [analysis['success_rates'].get(m, 0) for m in ['tfidf', 'bm25', 'splade', 'dense', 'hybrid']]
            })
            st.dataframe(metrics_df)
            
            # Result overlap
            st.write("### Result Overlap")
            overlap_data = []
            for comparison, overlap in analysis['result_overlap'].items():
                overlap_data.append({
                    'Comparison': comparison,
                    'Overlap': f"{overlap:.2%}"
                })
            st.dataframe(pd.DataFrame(overlap_data))

if __name__ == "__main__":
    main()