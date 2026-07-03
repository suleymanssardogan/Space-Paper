import os
import sys
import logging
from dotenv import load_dotenv

# Add parent or current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import SparseTextEmbedding
from save_to_qdrant import SpaceScienceVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("migrate_to_hybrid")

def migrate():
    collection_name = "space_science_collection"
    
    # 1. Initialize store
    logger.info("Initializing vector store...")
    store = SpaceScienceVectorStore()
    
    # 2. Scroll and backup all points
    if not store.client.collection_exists(collection_name):
        logger.error(f"Collection '{collection_name}' does not exist. Nothing to migrate.")
        return
        
    logger.info(f"Fetching all existing points from '{collection_name}'...")
    
    points = []
    offset = None
    while True:
        res, next_offset = store.client.scroll(
            collection_name=collection_name,
            limit=500,
            with_payload=True,
            with_vectors=True,
            offset=offset
        )
        for point in res:
            # Extract dense vector safely
            dense_vec = None
            if isinstance(point.vector, list):
                dense_vec = point.vector
            elif isinstance(point.vector, dict):
                dense_vec = point.vector.get("")
                
            if not dense_vec:
                logger.warning(f"Point {point.id} has no dense vector, skipping.")
                continue
                
            points.append({
                "id": point.id,
                "dense": dense_vec,
                "payload": point.payload
            })
            
        if not next_offset:
            break
        offset = next_offset
        
    logger.info(f"Successfully retrieved {len(points)} points for migration.")
    
    if not points:
        logger.info("No points found to migrate.")
        return

    # 3. Initialize sparse embedding model
    logger.info("Initializing sparse embedding model (Qdrant/bm25)...")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25", threads=1)
    
    # 4. Recreate collection
    logger.info(f"Deleting existing collection '{collection_name}'...")
    store.client.delete_collection(collection_name)
    
    logger.info(f"Recreating collection '{collection_name}' with sparse vector configuration...")
    # Using create_collection directly to set up both dense and sparse vectors
    store.client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        sparse_vectors_config={
            "sparse-text": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        },
        hnsw_config=models.HnswConfigDiff(
            m=16,
            ef_construct=100,
            full_scan_threshold=10000
        )
    )
    
    # Create payload indexes
    store.client.create_payload_index(
        collection_name=collection_name,
        field_name="text",
        field_schema="text"
    )
    store.client.create_payload_index(
        collection_name=collection_name,
        field_name="source",
        field_schema="keyword"
    )
    
    # 5. Generate sparse embeddings and upload in batches
    batch_size = 64
    total_points = len(points)
    logger.info(f"Starting upload of {total_points} points in batches of {batch_size}...")
    
    for i in range(0, total_points, batch_size):
        batch = points[i : i + batch_size]
        batch_texts = [p["payload"].get("text", "") for p in batch]
        
        # Embed sparse vectors
        sparse_embeddings = list(sparse_model.embed(batch_texts))
        
        upsert_points = []
        for idx, p in enumerate(batch):
            sparse_emb = sparse_embeddings[idx]
            upsert_points.append(
                models.PointStruct(
                    id=p["id"],
                    vector={
                        "": p["dense"],
                        "sparse-text": models.SparseVector(
                            indices=sparse_emb.indices.tolist(),
                            values=sparse_emb.values.tolist()
                        )
                    },
                    payload=p["payload"]
                )
            )
            
        store.client.upsert(collection_name=collection_name, points=upsert_points)
        logger.info(f"Migrated and uploaded batch {i // batch_size + 1}/{(total_points - 1) // batch_size + 1}")
        
    logger.info("Migration successfully completed!")

if __name__ == "__main__":
    migrate()
