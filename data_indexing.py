import pandas as pd
from typing import List, Dict, Any
from qdrant_client import QdrantClient, models
import numpy as np
from tqdm import tqdm
from vector_generation import VectorGenerator
from qdrant_setup import QdrantManager

class DataIndexer:
    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        self.client = QdrantClient(url=qdrant_url)
        self.vector_gen = VectorGenerator()
    
    def index_tfidf_collection(self, df: pd.DataFrame, collection_name: str = "tfidf_search"):
        """Index data using TF-IDF vectors"""
        texts = df['cleaned_text'].tolist()
        
        print("Generating TF-IDF vectors...")
        tfidf_matrix, tfidf_vectorizer = self.vector_gen.generate_tfidf_vectors(texts)
        
        print("Indexing TF-IDF vectors...")
        points = []
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            sparse_vector = self.vector_gen.tfidf_to_sparse_vector(
                tfidf_matrix[idx], tfidf_vectorizer
            )
            
            point = models.PointStruct(
                id=idx,
                payload={
                    "id": row.get('id', idx),
                    "title": row.get('title', ''),
                    "description": row.get('description', ''),
                    "attributes": row.get('attributes', ''),
                    "original_text": row['combined_text']
                },
                vector={"tfidf": sparse_vector}
            )
            points.append(point)
        
        self.client.upsert(collection_name=collection_name, points=points)
        print(f"Indexed {len(points)} documents with TF-IDF")
    
    def index_bm25_collection(self, df: pd.DataFrame, collection_name: str = "bm25_search"):
        """Index data using BM25 vectors"""
        texts = df['cleaned_text'].tolist()
        
        print("Generating BM25 representation...")
        bm25_corpus, tokenized_texts = self.vector_gen.generate_bm25_vectors(texts)
        
        # We need TF-IDF vectorizer for vocabulary mapping
        _, tfidf_vectorizer = self.vector_gen.generate_tfidf_vectors(texts)
        
        print("Indexing BM25 vectors...")
        points = []
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # For BM25, we store the tokenized text and compute scores at query time
            # For indexing, we'll create a simple frequency-based sparse vector
            tokens = tokenized_texts[idx]
            vocab_dict = tfidf_vectorizer.vocabulary_
            
            indices = []
            values = []
            
            # Create frequency-based sparse vector
            token_counts = {}
            for token in tokens:
                if token in vocab_dict:
                    token_counts[token] = token_counts.get(token, 0) + 1
            
            for token, count in token_counts.items():
                indices.append(vocab_dict[token])
                values.append(float(count))
            
            sparse_vector = models.SparseVector(indices=indices, values=values)
            
            point = models.PointStruct(
                id=idx,
                payload={
                    "id": row.get('id', idx),
                    "title": row.get('title', ''),
                    "description": row.get('description', ''),
                    "attributes": row.get('attributes', ''),
                    "original_text": row['combined_text'],
                    "tokens": tokens  # Store tokens for BM25 query processing
                },
                vector={"bm25": sparse_vector}
            )
            points.append(point)
        
        self.client.upsert(collection_name=collection_name, points=points)
        print(f"Indexed {len(points)} documents with BM25")
    
    def index_splade_collection(self, df: pd.DataFrame, collection_name: str = "splade_search"):
        """Index data using SPLADE vectors"""
        texts = df['cleaned_text'].tolist()
        
        print("Generating SPLADE vectors...")
        splade_vectors = self.vector_gen.generate_splade_vectors(texts)
        
        print("Indexing SPLADE vectors...")
        points = []
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            point = models.PointStruct(
                id=idx,
                payload={
                    "id": row.get('id', idx),
                    "title": row.get('title', ''),
                    "description": row.get('description', ''),
                    "attributes": row.get('attributes', ''),
                    "original_text": row['combined_text']
                },
                vector={"splade": splade_vectors[idx]}
            )
            points.append(point)
        
        self.client.upsert(collection_name=collection_name, points=points)
        print(f"Indexed {len(points)} documents with SPLADE")
    
    def index_dense_collection(self, df: pd.DataFrame, collection_name: str = "dense_search"):
        """Index data using dense embeddings"""
        texts = df['cleaned_text'].tolist()
        
        print("Generating dense vectors...")
        dense_vectors = self.vector_gen.generate_dense_vectors(texts)
        
        print("Indexing dense vectors...")
        points = []
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            point = models.PointStruct(
                id=idx,
                payload={
                    "id": row.get('id', idx),
                    "title": row.get('title', ''),
                    "description": row.get('description', ''),
                    "attributes": row.get('attributes', ''),
                    "original_text": row['combined_text']
                },
                vector={"dense": dense_vectors[idx].tolist()}  # Use named vector
            )
            points.append(point)
        
        self.client.upsert(collection_name=collection_name, points=points)
        print(f"Indexed {len(points)} documents with dense vectors")
    
    def index_hybrid_collection(self, df: pd.DataFrame, collection_name: str = "hybrid_search"):
        """Index data using both dense and SPLADE vectors for hybrid search"""
        texts = df['cleaned_text'].tolist()
        
        print("Generating dense and SPLADE vectors...")
        dense_vectors = self.vector_gen.generate_dense_vectors(texts)
        splade_vectors = self.vector_gen.generate_splade_vectors(texts)
        
        print("Indexing hybrid vectors...")
        points = []
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            point = models.PointStruct(
                id=idx,
                payload={
                    "id": row.get('id', idx),
                    "title": row.get('title', ''),
                    "description": row.get('description', ''),
                    "attributes": row.get('attributes', ''),
                    "original_text": row['combined_text']
                },
                vector={
                    "dense": dense_vectors[idx].tolist(),
                    "splade": splade_vectors[idx]
                }
            )
            points.append(point)
        
        self.client.upsert(collection_name=collection_name, points=points)
        print(f"Indexed {len(points)} documents with hybrid vectors")
    
    def index_all_collections(self, df: pd.DataFrame):
        """Index all collections and save models"""
        self.index_tfidf_collection(df)
        self.index_bm25_collection(df)
        self.index_splade_collection(df)
        self.index_dense_collection(df)
        self.index_hybrid_collection(df)
        
        # Save models after indexing
        print("Saving models...")
        self.vector_gen.save_models()
        print("Models saved successfully")

if __name__ == "__main__":
    # Load data
    from data_preprocessing import load_and_preprocess_data
    df = load_and_preprocess_data("data/shoe_products_dataset.csv")
    
    # Use a subset for testing (remove this for full dataset)
    df = df.head(1000)
    
    # Initialize indexer
    indexer = DataIndexer()
    
    # Index all collections
    indexer.index_all_collections(df)