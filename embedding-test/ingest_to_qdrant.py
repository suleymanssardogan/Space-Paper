import time
import logging
import uuid
from ingest_pdfs import DocumetPipeline
from save_to_qdrant import SpaceScienceVectorStore
from qdrant_client.models import PointStruct

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def bulk_upsert_chunks(store: SpaceScienceVectorStore, collection_name: str, chunks: list[dict], batch_size: int = 32):
    """Chunks listesini belirlenen batch_size boyutlarında gruplayarak Qdrant'a bulk upsert yapar."""
    total_chunks = len(chunks)
    logger.info(f"Toplam {total_chunks} adet chunk bulk upsert için hazırlanıyor. (Batch boyutu: {batch_size})")
    
    start_total_time = time.time()
    
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        batch_start_time = time.time()
        
        # 1. Mevcut batch içindeki metinleri topla ve topluca embedding üret
        batch_texts = [c["text"] for c in batch]
        embeddings = store.model.encode(batch_texts)
        
        # 2. Qdrant PointStruct listesini oluştur
        points = []
        for idx, chunk in enumerate(batch):
            # Idempotency sağlamak için metinden deterministik UUID üretiyoruz
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["text"]))
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embeddings[idx].tolist(),
                    payload={
                        "text": chunk["text"],
                        "source": chunk["source"],
                        "page_number": chunk["page_number"],
                        "chunk_index": chunk["chunk_index"],
                        "char_count": chunk["char_count"]
                    }
                )
            )
            
        # 3. Batch'i Qdrant'a yükle
        store.client.upsert(collection_name=collection_name, points=points)
        
        batch_latency = time.time() - batch_start_time
        throughput = len(batch) / batch_latency
        logger.info(
            f"Batch [{i // batch_size + 1}/{(total_chunks - 1) // batch_size + 1}] "
            f"tamamlandı. Süre: {batch_latency:.4f} sn | Hız: {throughput:.2f} vektör/sn"
        )
        
    total_latency = time.time() - start_total_time
    logger.info(f"Tüm bulk upsert işlemi {total_latency:.4f} saniye sürdü.")

if __name__ == "__main__":
    # 1. Adım: Veri toplama ve parçalama hattı (Pipeline)
    pdf_pipeline = DocumetPipeline(download_dir="data")
    
    papers = {
        "jwst_performance.pdf": "https://arxiv.org/pdf/2207.13067",
        "kepler_mission.pdf": "https://arxiv.org/pdf/1001.0352"
    }
    
    all_extracted_chunks = []
    
    # PDF'leri indir, oku ve parçala
    for filename, url in papers.items():
        logger.info(f"Dosya işleniyor: {filename}")
        pdf_path = pdf_pipeline.download_pdf(url, filename)
        pages_data = pdf_pipeline.extract_text_from_pdf(pdf_path)
        chunks_data = pdf_pipeline.chunk_text_data(pages_data)
        
        # Her chunk'a kaynak PDF bilgisini (filename) ekliyoruz
        for chunk in chunks_data:
            chunk["source"] = filename
            all_extracted_chunks.append(chunk)
            
    # 2. Adım: Vektör Veritabanı Bağlantısı
    store = SpaceScienceVectorStore()
    collection = "space_science_collection"
    
    # Koleksiyonu başlat (zaten varsa bir şey yapmayacak)
    store.init_collection(collection)
    
    # 3. Adım: Bulk Upsert işlemini çalıştır
    bulk_upsert_chunks(store, collection, all_extracted_chunks, batch_size=32)
