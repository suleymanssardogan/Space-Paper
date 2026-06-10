import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from save_to_qdrant import SpaceScienceVectorStore

# FastAPI uygulamasını başlat
app = FastAPI(
    title="Antispace RAG API Gateway",
    description="Space Science Academic Papers Search API",
    version="1.0.0"
)

# Vektör veritabanı sınıfımızı ilklendir
# (Zaten çalışan local Qdrant container'ımıza bağlanacak)
store = SpaceScienceVectorStore()
COLLECTION_NAME = "space_science_collection"

# --- 1. PYDANTIC ŞEMALARI (Data Validation) ---

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Aranacak semantik sorgu metni")
    limit: int = Field(default=3, ge=1, le=10, description="Dönecek maksimum sonuç sayısı")
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0, description="Benzerlik eşik değeri (Cosine similarity)")

class SearchResultItem(BaseModel):
    text: str = Field(..., description="Belge parçası (chunk)")
    source: str = Field(..., description="Kaynak dosya adı (PDF)")
    page_number: int = Field(..., description="Sayfa numarası")
    score: float = Field(..., description="Semantik benzerlik skoru")

class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]
    latency_seconds: float = Field(..., description="Arama işleminin sunucu tarafındaki gecikme süresi")

# --- 2. API ENDPOINTS ---

@app.post("/api/v1/search", response_model=SearchResponse)
def search_documents(request: SearchRequest):
    """
    Qdrant üzerindeki akademik uzay belgelerinde semantik arama gerçekleştirir.
    """
    try:
        start_time = time.time()
        
        # Qdrant üzerinde semantik aramayı çalıştır
        raw_results = store.search_documents(
            collection_name=COLLECTION_NAME,
            query=request.query,
            limit=request.limit,
            score_threshold=request.score_threshold
        )
        
        # Sonuçları Pydantic şemamıza uygun hale getir
        results = []
        for point in raw_results:
            results.append(
                SearchResultItem(
                    text=point.payload.get("text", ""),
                    source=point.payload.get("source", "unknown"),
                    page_number=point.payload.get("page_number", 0),
                    score=point.score
                )
            )
            
        latency = time.time() - start_time
        
        return SearchResponse(
            query=request.query,
            results=results,
            latency_seconds=round(latency, 4)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Arama işlemi sırasında bir sunucu hatası oluştu: {str(e)}")
