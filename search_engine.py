from qdrant_client import QdrantClient, models
from typing import List, Dict, Any, Optional
import numpy as np
from vector_generation import VectorGenerator
import json

class SearchEngine:
    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        self.client = QdrantClient(url=qdrant_url)
        self.vector_gen = VectorGenerator()
        # Load saved models
        self.vector_gen.load_models()
    
    def search_tfidf(self, query: str, collection_name: str = "tfidf_search", 
                    limit: int = 10) -> List[Dict[str, Any]]:
        """Search using TF-IDF vectors"""
        # Generate TF-IDF vector for query
        query_vector = self.vector_gen.tfidf_vectorizer.transform([query])
        sparse_vector = self.vector_gen.tfidf_to_sparse_vector(
            query_vector[0], self.vector_gen.tfidf_vectorizer
        )
        
        search_result = self.client.search(
            collection_name=collection_name,
            query_vector=models.NamedSparseVector(
                name="tfidf",
                vector=sparse_vector
            ),
            limit=limit,
            with_payload=True
        )
        
        return self._format_search_results(search_result)
    
    def search_bm25(self, query: str, collection_name: str = "bm25_search", 
                   limit: int = 10) -> List[Dict[str, Any]]:
        """Search using BM25 vectors"""
        # Clean and tokenize query
        cleaned_query = self.vector_gen.tfidf_vectorizer.build_preprocessor()(query)
        query_tokens = cleaned_query.split()
        
        # Get all documents to compute BM25 scores
        # Note: This is a simplified approach. In production, you'd want to optimize this.
        all_points = self.client.scroll(
            collection_name=collection_name,
            limit=1000,  # Adjust based on your dataset size
            with_payload=True,
            with_vectors=False
        )[0]
        
        # Compute BM25 scores for each document
        bm25_scores = []
        for point in all_points:
            doc_tokens = point.payload.get('tokens', [])
            score = self.vector_gen.bm25_corpus.get_scores(query_tokens)[point.id]
            bm25_scores.append((point.id, score))
        
        # Sort by score and get top results
        bm25_scores.sort(key=lambda x: x[1], reverse=True)
        top_ids = [point_id for point_id, score in bm25_scores[:limit]]
        
        # Get full documents for top results
        results = []
        for point_id in top_ids:
            point = self.client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_payload=True
            )[0]
            
            results.append({
                'id': point.id,
                'score': bm25_scores[top_ids.index(point_id)][1],
                'payload': point.payload
            })
        
        return results
    
    def search_splade(self, query: str, collection_name: str = "splade_search", 
                     limit: int = 10) -> List[Dict[str, Any]]:
        """Search using SPLADE vectors"""
        # Generate SPLADE vector for query
        query_embeddings = list(self.vector_gen.sparse_model.embed([query]))
        query_sparse_vector = models.SparseVector(
            indices=query_embeddings[0].indices.tolist(),
            values=query_embeddings[0].values.tolist()
        )
        
        search_result = self.client.search(
            collection_name=collection_name,
            query_vector=models.NamedSparseVector(
                name="splade",
                vector=query_sparse_vector
            ),
            limit=limit,
            with_payload=True
        )
        
        return self._format_search_results(search_result)
    
    def search_dense(self, query: str, collection_name: str = "dense_search", 
                    limit: int = 10) -> List[Dict[str, Any]]:
        """Search using dense embeddings"""
        # Generate dense vector for query
        query_vector = self.vector_gen.dense_model.encode([query])[0]
        
        # Use named vector for search
        search_result = self.client.search(
            collection_name=collection_name,
            query_vector=models.NamedVector(
                name="dense",
                vector=query_vector.tolist()
            ),
            limit=limit,
            with_payload=True
        )
        
        return self._format_search_results(search_result)
    
    def search_hybrid(self, query: str, collection_name: str = "hybrid_search", 
                     limit: int = 10) -> List[Dict[str, Any]]:
        """Search using hybrid approach (dense + SPLADE with manual fusion)"""
        # Search with dense vectors
        dense_results = self.client.search(
            collection_name=collection_name,
            query_vector=models.NamedVector(
                name="dense",
                vector=self.vector_gen.dense_model.encode([query])[0].tolist()
            ),
            limit=limit * 2,
            with_payload=True
        )
        
        # Search with SPLADE vectors
        query_embeddings = list(self.vector_gen.sparse_model.embed([query]))
        query_sparse_vector = models.SparseVector(
            indices=query_embeddings[0].indices.tolist(),
            values=query_embeddings[0].values.tolist()
        )
        
        sparse_results = self.client.search(
            collection_name=collection_name,
            query_vector=models.NamedSparseVector(
                name="splade",
                vector=query_sparse_vector
            ),
            limit=limit * 2,
            with_payload=True
        )
        
        # Manual fusion using Reciprocal Rank Fusion (RRF)
        return self._reciprocal_rank_fusion(dense_results, sparse_results, limit)
    
    def _reciprocal_rank_fusion(self, dense_results, sparse_results, limit=10, k=60):
        """
        Implement Reciprocal Rank Fusion (RRF)
        k is a parameter, typically set to 60
        """
        # Create dictionaries to store scores
        fused_scores = {}
        
        # Process dense results
        for rank, point in enumerate(dense_results, 1):
            doc_id = point.id
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {
                    'score': 0,
                    'point': point
                }
            fused_scores[doc_id]['score'] += 1.0 / (k + rank)
        
        # Process sparse results
        for rank, point in enumerate(sparse_results, 1):
            doc_id = point.id
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {
                    'score': 0,
                    'point': point
                }
            fused_scores[doc_id]['score'] += 1.0 / (k + rank)
        
        # Sort by fused score and return top results
        sorted_results = sorted(fused_scores.values(), key=lambda x: x['score'], reverse=True)
        
        # Format results
        formatted_results = []
        for item in sorted_results[:limit]:
            formatted_results.append({
                'id': item['point'].id,
                'score': item['score'],
                'payload': item['point'].payload
            })
        
        return formatted_results
    
    def _format_search_results(self, search_result) -> List[Dict[str, Any]]:
        """Format search results consistently"""
        results = []
        for point in search_result:
            results.append({
                'id': point.id,
                'score': point.score,
                'payload': point.payload
            })
        return results
    
    def compare_search_methods(self, query: str, limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """Compare results from all search methods"""
        results = {}
        
        try:
            results['tfidf'] = self.search_tfidf(query, limit=limit)
        except Exception as e:
            results['tfidf'] = {'error': str(e)}
        
        try:
            results['bm25'] = self.search_bm25(query, limit=limit)
        except Exception as e:
            results['bm25'] = {'error': str(e)}
        
        try:
            results['splade'] = self.search_splade(query, limit=limit)
        except Exception as e:
            results['splade'] = {'error': str(e)}
        
        try:
            results['dense'] = self.search_dense(query, limit=limit)
        except Exception as e:
            results['dense'] = {'error': str(e)}
        
        try:
            results['hybrid'] = self.search_hybrid(query, limit=limit)
        except Exception as e:
            results['hybrid'] = {'error': str(e)}
        
        return results

if __name__ == "__main__":
    # Test the search engine
    engine = SearchEngine()
    
    # Test query
    query = "running shoes for men"
    
    print(f"Testing search with query: '{query}'")
    print("=" * 50)
    
    # Compare all methods
    results = engine.compare_search_methods(query, limit=5)
    
    for method, method_results in results.items():
        print(f"\n{method.upper()} Results:")
        print("-" * 30)
        if isinstance(method_results, dict) and 'error' in method_results:
            print(f"Error: {method_results['error']}")
        else:
            for i, result in enumerate(method_results, 1):
                print(f"{i}. Score: {result['score']:.4f}")
                print(f"   Title: {result['payload'].get('title', 'N/A')}")
                print(f"   Description: {result['payload'].get('description', 'N/A')[:100]}...")
                print()