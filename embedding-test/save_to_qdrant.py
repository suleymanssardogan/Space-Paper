import os
import time
import logging
from qdrant_client import QdrantClient

from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    SparseIndexParams,
    Prefetch,
    FusionQuery,
    Fusion
)
from fastembed import TextEmbedding, SparseTextEmbedding

logging.basicConfig(
level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SpaceScienceVectorStore:
    def __init__(self, host: str = None):
        # QdrantClient'ı başlat (Yerel veya Qdrant Cloud)
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if qdrant_url:
            logger.info(f"Qdrant Cloud bağlantısı kuruluyor: {qdrant_url}")
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60)
        else:
            if os.getenv("GITHUB_ACTIONS") == "true":
                raise ValueError(
                    "\n\n❌ HATA: QDRANT_URL çevre değişkeni GitHub Actions ortamında bulunamadı!\n"
                    "Lütfen GitHub deposunun (Repository) ayarlarından (Settings -> Secrets and variables -> Actions) "
                    "QDRANT_URL ve QDRANT_API_KEY sırlarını (Secrets) eklediğinizden emin olun.\n"
                )
            if host is None:
                host = os.getenv("QDRANT_HOST", "http://localhost:6333")
            logger.info(f"Yerel Qdrant bağlantısı kuruluyor: {host}")
            self.client = QdrantClient(host, timeout=60)

        # 384 boyutlu vektör üreten modelmizi yükleme (fastembed kullanarak bellek tüketimini düşürüyoruz)
        # threads=1 parametresi ile bellek ve CPU kullanımını optimize ediyoruz.
        self.model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", threads=1)
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25", threads=1)

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Fastembed kullanarak metinleri vektörleştirir."""
        return [emb.tolist() for emb in self.model.embed(texts)]

    def encode_sparse(self, texts: list[str]) -> list[dict]:
        """Fastembed kullanarak metinlerin sparse vektörlerini (indices ve values) üretir."""
        return [
            {"indices": emb.indices.tolist(), "values": emb.values.tolist()}
            for emb in self.sparse_model.embed(texts)
        ]

    def init_collection(self, collection_name: str):
        try: 
            if self.client.collection_exists(collection_name=collection_name):
                logger.info(f"Collection already exists: {collection_name}")
                # Mevcut koleksiyon için source endeksini oluştur (hata verirse yok say)
                try:
                    self.client.create_payload_index(
                        collection_name=collection_name,
                        field_name="source",
                        field_schema="keyword"
                    )
                except Exception as ex:
                    logger.debug(f"Source index checking: {ex}")
                return
            
            logger.info(f"Creating collection: {collection_name}")
            from qdrant_client.models import HnswConfigDiff
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                sparse_vectors_config={
                    "sparse-text": SparseVectorParams(
                        index=SparseIndexParams(
                            on_disk=False,
                        )
                    )
                },
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=10000
                )
            )
            
            # Tam metin araması (Full-text keyword matching) için indeks oluştur
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="text",
                field_schema="text"
            )
            
            # Kaynak filtreleme için indeks oluştur
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="source",
                field_schema="keyword"
            )
            logger.info(f"Collection created with text payload index: {collection_name}")

        except Exception as e:
            logger.error(f"Error creating collection: {e}")
    


    def upsert_documents(self,collection_name:str,documents:list[str]):
        try:
            start_time = time.time()
            logger.info(f"{len(documents)} adet doküman için embedding üretiliyor...")

            # 1. Cümlelerin vektörlerini ve sparse vektörlerini çıkartmak
            embeddings = self.encode(documents)
            sparse_embeddings = self.encode_sparse(documents)

            # 2. Qdrant Point'lerini (PointStruct) hazırlamak
            points =[]
            for i,doc in enumerate(documents):
                points.append(
                    PointStruct(
                        id =i,
                        vector={
                            "": embeddings[i],
                            "sparse-text": SparseVector(
                                indices=sparse_embeddings[i]["indices"],
                                values=sparse_embeddings[i]["values"]
                            )
                        },
                        payload={
                            "text":doc,
                            "source":"space_paper_archive",
                            "char_count":len(doc)
                        }
                    )
                )

            # 3. Qdranta yüklemek (Upsert)
            logger.info(f"Vektörler Qdranta yükleniyor...")
            self.client.upsert(collection_name=collection_name,points=points)

            # 4. Latency Ölçme
            latency = time.time()-start_time
            logger.info(f"Upsert işlemi {latency:.4f} saniye sürdü")

        except Exception as e:
            logger.error(f"Upsert işlemi sırasında hata oluştu: {e}")
            raise e
    #Hybrid search
    def search_documents(self, collection_name: str, query: str, limit: int = 3, score_threshold: float = None, source_filter: str = None):
        try:
            start_time=time.time()
            logger.info(f"Sorgu için hibrid arama yapılıyor: '{query}' (Kaynak Filtresi: {source_filter})")

            # 1. Sorgu cümlesini vektörleştirme
            query_vector = self.encode([query])[0]
            sparse_vec = self.encode_sparse([query])[0]
            sparse_q = SparseVector(indices=sparse_vec["indices"], values=sparse_vec["values"])

            # 2. Ön-filtreleme (Pre-filtering) oluşturma
            query_filter = None
            if source_filter:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=source_filter)
                        )
                    ]
                )

            # 3. Qdrant üzerinde hibrid arama yapma (Prefetch ve RRF)
            # score_threshold parametresini sadece dense prefetch adımına uyguluyoruz.
            results = self.client.query_points(
                collection_name=collection_name,
                prefetch=[
                    Prefetch(
                        query=query_vector,
                        using="",
                        limit=limit * 2,
                        score_threshold=score_threshold
                    ),
                    Prefetch(
                        query=sparse_q,
                        using="sparse-text",
                        limit=limit * 2
                    )
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
                query_filter=query_filter
            )

            # 4.Latency Ölçme
            latency = time.time() - start_time
            logger.info(f"Arama işlemi {latency:.2f} saniye sürdü")
            
            return results.points
        except Exception as e:
            logger.error(f"Arama sırasında hata oluştu: {e}")
            raise e
        

            
if __name__ == "__main__":
    # 1. Sınıfımızı ilklendirelim
    store = SpaceScienceVectorStore()
    
    # 2. Koleksiyon ismini belirleyelim
    collection = "space_science_collection"
    
    # 3. Koleksiyonu güvenli bir şekilde oluşturalım
    store.init_collection(collection)
    
    # 4. Test cümlelerimizi hazırlayalım (2 uzay, 1 alakasız pizza cümlesi)
    test_docs = [
        "The James Webb Space Telescope captured deep field images of early galaxies.",
        "Webb telescope observed ancient galaxies in the deep universe.",
        "A pizza recipe requires flour, water, yeast, and tomato sauce."
    ]
    
    # 5. Cümleleri vektörleştirip Qdrant'a yükleyelim
    store.upsert_documents(collection, test_docs)

    print("\n=== SEMANTİK ARAMA TESTLERİ ===")
    
    # Test 1: Uzay araması (Uzay cümleleri yüksek skor almalı, pizza elenmeli)
    query_1 = "space exploration and telescopes"
    print(f"\nSorgu: '{query_1}' (Eşik Değeri: 0.40)")
    results_1 = store.search_documents(collection, query_1, score_threshold=0.40)
    for res in results_1:
        print(f"- [Skor: {res.score:.4f}] Metin: {res.payload['text']}")

    # Test 2: Yemek araması (Pizza cümlesi yüksek skor almalı, uzay elenmeli)
    query_2 = "baking pizza and cooking"
    print(f"\nSorgu: '{query_2}' (Eşik Değeri: 0.40)")
    results_2 = store.search_documents(collection, query_2, score_threshold=0.40)
    for res in results_2:
        print(f"- [Skor: {res.score:.4f}] Metin: {res.payload['text']}")

    # Test 3: Herhangi bir filtre olmadan tüm sonuçları listeleme
    query_3 = "universe"
    print(f"\nSorgu: '{query_3}' (Filtre yok)")
    results_3 = store.search_documents(collection, query_3)
    for res in results_3:
        print(f"- [Skor: {res.score:.4f}] Metin: {res.payload['text']}")

    
        
