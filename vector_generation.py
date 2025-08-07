import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer
import numpy as np
from qdrant_client import QdrantClient, models
import json
import pickle
import os
from data_preprocessing import load_and_preprocess_data

class VectorGenerator:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.bm25_corpus = None
        self.sparse_model = None
        self.dense_model = None
        self.models_dir = "saved_models"
        
        # Create directory for saved models if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
    def save_models(self):
        """Save trained models for later use"""
        if self.tfidf_vectorizer:
            with open(os.path.join(self.models_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
        
        if self.bm25_corpus:
            with open(os.path.join(self.models_dir, 'bm25_corpus.pkl'), 'wb') as f:
                pickle.dump(self.bm25_corpus, f)
    
    def load_models(self):
        """Load saved models"""
        # Load TF-IDF vectorizer
        tfidf_path = os.path.join(self.models_dir, 'tfidf_vectorizer.pkl')
        if os.path.exists(tfidf_path):
            with open(tfidf_path, 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
        
        # Load BM25 corpus
        bm25_path = os.path.join(self.models_dir, 'bm25_corpus.pkl')
        if os.path.exists(bm25_path):
            with open(bm25_path, 'rb') as f:
                self.bm25_corpus = pickle.load(f)
        
        # Initialize sparse model if not already initialized
        if not self.sparse_model:
            self.sparse_model = SparseTextEmbedding(
                model_name="prithivida/Splade_PP_en_v1"
            )
        
        # Initialize dense model if not already initialized
        if not self.dense_model:
            self.dense_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def generate_tfidf_vectors(self, texts: List[str]) -> Tuple[np.ndarray, TfidfVectorizer]:
        """Generate TF-IDF sparse vectors"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        return tfidf_matrix, self.tfidf_vectorizer
    
    def generate_bm25_vectors(self, texts: List[str]) -> Tuple[BM25Okapi, List[List[str]]]:
        """Generate BM25 sparse representation"""
        # Tokenize texts for BM25
        tokenized_texts = [text.split() for text in texts]
        self.bm25_corpus = BM25Okapi(tokenized_texts)
        return self.bm25_corpus, tokenized_texts
    
    def generate_splade_vectors(self, texts: List[str]) -> List[models.SparseVector]:
        """Generate SPLADE sparse vectors using FastEmbed"""
        if not self.sparse_model:
            self.sparse_model = SparseTextEmbedding(
                model_name="prithivida/Splade_PP_en_v1"
            )
        
        sparse_embeddings = list(self.sparse_model.embed(texts))
        
        # Convert to Qdrant SparseVector format
        qdrant_sparse_vectors = []
        for embedding in sparse_embeddings:
            qdrant_sparse_vectors.append(
                models.SparseVector(
                    indices=embedding.indices.tolist(),
                    values=embedding.values.tolist()
                )
            )
        
        return qdrant_sparse_vectors
    
    def generate_dense_vectors(self, texts: List[str]) -> np.ndarray:
        """Generate dense embeddings using Sentence Transformers"""
        if not self.dense_model:
            self.dense_model = SentenceTransformer('all-MiniLM-L6-v2')
        dense_embeddings = self.dense_model.encode(texts)
        return dense_embeddings
    
    def tfidf_to_sparse_vector(self, tfidf_vector, vectorizer) -> models.SparseVector:
        """Convert TF-IDF sparse matrix to Qdrant SparseVector"""
        # Get non-zero elements
        nonzero_indices = tfidf_vector.nonzero()[1]
        nonzero_values = tfidf_vector.data
        
        return models.SparseVector(
            indices=nonzero_indices.tolist(),
            values=nonzero_values.tolist()
        )
    
    def bm25_to_sparse_vector(self, query_tokens: List[str], bm25: BM25Okapi, 
                            vectorizer: TfidfVectorizer) -> models.SparseVector:
        """Convert BM25 scores to sparse vector format"""
        # Get BM25 scores for all documents
        doc_scores = bm25.get_scores(query_tokens)
        
        # Create a sparse vector representation
        # We'll use the TF-IDF vocabulary indices
        vocab_dict = vectorizer.vocabulary_
        indices = []
        values = []
        
        for token in query_tokens:
            if token in vocab_dict:
                indices.append(vocab_dict[token])
                # Use BM25 score as value
                values.append(1.0)  # Simplified - in practice you'd calculate actual BM25 weights
        
        return models.SparseVector(indices=indices, values=values)

if __name__ == "__main__":
    # Test vector generation
    df = load_and_preprocess_data("data/shoe_products_dataset.csv")
    texts = df['cleaned_text'].tolist()
    
    vector_gen = VectorGenerator()
    
    # Generate all vector types
    tfidf_matrix, tfidf_vectorizer = vector_gen.generate_tfidf_vectors(texts[:100])  # Test with subset
    bm25_corpus, tokenized_texts = vector_gen.generate_bm25_vectors(texts[:100])
    splade_vectors = vector_gen.generate_splade_vectors(texts[:10])  # Test with smaller subset
    dense_vectors = vector_gen.generate_dense_vectors(texts[:100])
    
    # Save models
    vector_gen.save_models()
    
    print("Vector generation test completed")
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"SPLADE vectors count: {len(splade_vectors)}")
    print(f"Dense vectors shape: {dense_vectors.shape}")