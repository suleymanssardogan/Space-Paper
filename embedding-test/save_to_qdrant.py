import time
import logging
from qdrant_client import QdrantClient

from qdrant_client.models import VectorParams,Distance,PointStruct
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SpaceScienceVectorStore:
    def __init__(self,host:str="http://localhost:6333"):
        # QdrantClient'ı başlat
        self.client = QdrantClient(host)

        # 384 boyutlu vektör üreten modelmizi yükleme
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def init_collection(self,collection_name:str):

        try: 
            if self.client.collection_exists(collection_name=collection_name):
                logger.info(f"Collection already exists:{collection_name}")
                return
            
            logger.info(f"Creating collection: {collection_name}")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384,distance=Distance.COSINE)

            )
            logger.info(f"Collection created: {collection_name}")

        except Exception as e:
            logger.error(f"Error creating collection: {e}")
    


    def upsert_documents(self,collection_name:str,documents:list[str]):
        try:
            start_time = time.time()
            logger.info(f"{len(documents)} ader doküman için embedding üretiliyor...")

            # 1. Cümlelerin vektörleri çıkartmak
            embeddings = self.model.encode(documents)

            # 2. Qdrant Point'lerini (PointStruct) hazırlamak
            points =[]
            for i,doc in enumerate(documents):
                points.append(
                    PointStruct(
                        id =i,
                        vector=embeddings[i].tolist(),
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
    
        
