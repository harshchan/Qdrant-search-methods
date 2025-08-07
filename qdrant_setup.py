from qdrant_client import QdrantClient, models
from typing import List, Dict, Any
import numpy as np

class QdrantManager:
    def __init__(self, url: str = "http://localhost:6333"):
        self.client = QdrantClient(url=url)
    
    def create_hybrid_collection(self, collection_name: str, dense_dim: int):
        """Create a collection with both dense and sparse vector support"""
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=dense_dim,
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "tfidf": models.SparseVectorParams(),
                    "bm25": models.SparseVectorParams(),
                    "splade": models.SparseVectorParams()
                }
            )
            print(f"Collection '{collection_name}' created successfully")
        else:
            print(f"Collection '{collection_name}' already exists")
    
    def create_individual_collections(self, dense_dim: int):
        """Create separate collections for each search method"""
        collections_config = {
            "tfidf_search": {"has_dense": False, "has_sparse": True, "sparse_name": "tfidf"},
            "bm25_search": {"has_dense": False, "has_sparse": True, "sparse_name": "bm25"},
            "splade_search": {"has_dense": False, "has_sparse": True, "sparse_name": "splade"},
            "dense_search": {"has_dense": True, "has_sparse": False, "dense_dim": dense_dim},
            "hybrid_search": {"has_dense": True, "has_sparse": True, "sparse_name": "splade", "dense_dim": dense_dim}
        }
        
        for collection_name, config in collections_config.items():
            if not self.client.collection_exists(collection_name):
                vectors_config = None
                sparse_vectors_config = None
                
                if config["has_dense"]:
                    vectors_config = {
                        "dense": models.VectorParams(
                            size=config["dense_dim"],
                            distance=models.Distance.COSINE
                        )
                    }
                
                if config["has_sparse"]:
                    sparse_vectors_config = {
                        config["sparse_name"]: models.SparseVectorParams()
                    }
                
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=vectors_config,
                    sparse_vectors_config=sparse_vectors_config
                )
                print(f"Collection '{collection_name}' created successfully")
            else:
                print(f"Collection '{collection_name}' already exists")

if __name__ == "__main__":
    qdrant_manager = QdrantManager()
    
    # Create collections (dense dimension for all-MiniLM-L6-v2 is 384)
    qdrant_manager.create_individual_collections(dense_dim=384)