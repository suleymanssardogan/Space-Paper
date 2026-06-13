import os
import time
import requests
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from save_to_qdrant import SpaceScienceVectorStore
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fastapi")

# FastAPI uygulamasını başlat
app = FastAPI(
    title="Antispace RAG API Gateway",
    description="Space Science Academic Papers Search & Q&A API",
    version="1.2.0"
)

# Vektör veritabanı sınıfımızı ilklendir
store = SpaceScienceVectorStore()
COLLECTION_NAME = "space_science_collection"

# Çevre değişkeninden Hugging Face Token'ını oku
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Langfuse Observability entegrasyonu
from langfuse import Langfuse
langfuse = None
if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
    try:
        logger.info("Langfuse monitoring aktif edildi.")
        langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        )
    except Exception as le:
        logger.warning(f"Langfuse başlatılamadı: {le}")

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

# --- DAY 9: NEW SCHEMAS FOR RAG Q&A ---

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Uzay bilimleriyle ilgili sorunuz")
    limit: int = Field(default=3, ge=1, le=10, description="Bağlam olarak kullanılacak kaynak sayısı")
    score_threshold: float = Field(default=0.35, ge=0.0, le=1.0, description="Min benzerlik eşiği")

class CitationItem(BaseModel):
    source: str = Field(..., description="Kaynak dosya adı")
    page_number: int = Field(..., description="Sayfa numarası")
    score: float = Field(..., description="Benzerlik skoru")

class AskResponse(BaseModel):
    question: str
    answer: str = Field(..., description="Yapay zeka tarafından üretilen güvenilir cevap")
    citations: list[CitationItem] = Field(..., description="Cevap için kullanılan kaynak referansları")
    latency_seconds: float = Field(..., description="Toplam işlem süresi (Arama + LLM)")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Genel sistem durumu (healthy / unhealthy)")
    database_connected: bool = Field(..., description="Qdrant veritabanı bağlantı durumu")
    collection_exists: bool = Field(..., description="Hedef koleksiyon varlık durumu")
    vector_count: int = Field(..., description="Koleksiyondaki toplam vektör sayısı")
    llm_configured: bool = Field(..., description="LLM sağlayıcı anahtarlarından en az birinin yüklü olup olmadığı")
    latency_seconds: float = Field(..., description="Sağlık kontrolü sorgu süresi")

# --- DAY 10: NEW PRODUCTION SCHEMAS (Feedback & Ingestion) ---

class FeedbackRequest(BaseModel):
    question: str = Field(..., description="Kullanıcının sorduğu soru")
    answer: str = Field(..., description="Kullanıcıya verilen cevap")
    score: int = Field(..., ge=-1, le=1, description="Kullanıcı oyu: 1 (Thumbs Up), -1 (Thumbs Down)")
    feedback_text: str | None = Field(default=None, description="Kullanıcının isteğe bağlı yazılı geri bildirimi")

class IngestRequest(BaseModel):
    category: str = Field(default="astro-ph.CO+OR+cat:astro-ph.EP", description="arXiv arama kategorileri")
    max_results: int = Field(default=3, ge=1, le=10, description="Maksimum indirilecek makale sayısı")

class IngestResponse(BaseModel):
    status: str = Field(..., description="İşlem durumu (success / error)")
    papers_ingested: int = Field(..., description="Başarıyla Qdrant'a yüklenen makale sayısı")
    message: str = Field(..., description="Detaylı durum mesajı")

# --- 2. API ENDPOINTS ---

# --- Reranking (Cross-Encoder / Cohere Rerank) Mimarisi ---
_local_reranker = None

def get_local_reranker():
    global _local_reranker
    if _local_reranker is None:
        from sentence_transformers import CrossEncoder
        logger.info("Yerel Cross-Encoder reranker yükleniyor: cross-encoder/ms-marco-MiniLM-L-6-v2")
        _local_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _local_reranker

def rerank_documents(query: str, raw_results: list, limit: int) -> list:
    if not raw_results:
        return []
        
    cohere_key = os.getenv("COHERE_API_KEY", "")
    docs = [p.payload.get("text", "") for p in raw_results]
    
    # Cohere API Reranking (Premium / Cloud)
    if cohere_key:
        try:
            logger.info("Cohere Rerank API ile yeniden sıralanıyor...")
            import cohere
            co = cohere.Client(cohere_key)
            rerank_res = co.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=docs,
                top_n=limit
            )
            
            reranked_results = []
            for result in rerank_res.results:
                idx = result.index
                point = raw_results[idx]
                point.score = float(result.relevance_score)
                reranked_results.append(point)
            return reranked_results
        except Exception as ce:
            logger.warning(f"Cohere Rerank hatası: {ce}. Yerel reranker modeline geçiliyor...")
            
    # Yerel MiniLM Reranking (Local Fallback)
    try:
        model = get_local_reranker()
        pairs = [[query, doc] for doc in docs]
        scores = model.predict(pairs)
        
        scored_points = []
        for idx, score in enumerate(scores):
            point = raw_results[idx]
            point.score = float(score)
            scored_points.append(point)
            
        # Skorlara göre azalan sırada diz ve en alakalı 'limit' tanesini al
        scored_points.sort(key=lambda x: x.score, reverse=True)
        return scored_points[:limit]
    except Exception as le:
        logger.error(f"Yerel rerank işleminde hata oluştu: {le}. İlk sonuçlar doğrudan dönülüyor.")
        return raw_results[:limit]


@app.get("/", response_class=FileResponse)
def read_root():
    """
    Antispace RAG Arayüzü (Dashboard) ana sayfasını döner.
    """
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if not os.path.exists(html_path):
        return HTMLResponse(
            content="<h2>Antispace Dashboard Yükleniyor...</h2><p>Lütfen statik dosyaların hazırlandığından emin olun.</p>",
            status_code=404
        )
    return FileResponse(html_path)

@app.get("/api/v1/health", response_model=HealthResponse)
def health_check():
    """
    Sistem sağlığını, Qdrant bağlantısını ve veri hacmini doğrular.
    """
    start_time = time.time()
    db_connected = False
    coll_exists = False
    vec_count = 0
    
    try:
        coll_exists = store.client.collection_exists(collection_name=COLLECTION_NAME)
        db_connected = True
        
        if coll_exists:
            coll_info = store.client.get_collection(collection_name=COLLECTION_NAME)
            vec_count = coll_info.points_count
            
    except Exception as e:
        logger.error(f"Sağlık kontrolü sırasında veritabanı hatası: {e}")
        db_connected = False
        
    # LLM sağlayıcı anahtarlarının kontrolü
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "") or os.getenv("HF_TOKEN", "")
    llm_configured = bool(gemini_key or openrouter_key)
    
    status_str = "healthy" if db_connected and coll_exists else "unhealthy"
    latency = time.time() - start_time
    
    return HealthResponse(
        status=status_str,
        database_connected=db_connected,
        collection_exists=coll_exists,
        vector_count=vec_count,
        llm_configured=llm_configured,
        latency_seconds=round(latency, 4)
    )

@app.post("/api/v1/search", response_model=SearchResponse)
def search_documents(request: SearchRequest):
    """
    Qdrant üzerindeki akademik uzay belgelerinde semantik arama gerçekleştirir.
    """
    try:
        start_time = time.time()
        
        # Arama havuzunu genişletip (limit * 3) Reranking'e tabi tutuyoruz
        candidate_limit = request.limit * 3
        raw_results = store.search_documents(
            collection_name=COLLECTION_NAME,
            query=request.query,
            limit=candidate_limit,
            score_threshold=request.score_threshold
        )
        
        # Rerank işlemi
        raw_results = rerank_documents(request.query, raw_results, request.limit)
        
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


@app.post("/api/v1/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    RAG (Retrieval-Augmented Generation) kullanarak soruya sadece veritabanındaki belgelere dayanarak cevap üretir.
    """
    try:
        start_time = time.time()
        
        # 1. Adım: Qdrant'tan ilgili bağlam adaylarını (limit * 3) ara
        candidate_limit = request.limit * 3
        raw_results = store.search_documents(
            collection_name=COLLECTION_NAME,
            query=request.question,
            limit=candidate_limit,
            score_threshold=request.score_threshold
        )
        
        # 2. Adım: Cross-Encoder ile en alakalı makale parçalarını Rerank et
        raw_results = rerank_documents(request.question, raw_results, request.limit)
        
        # Eğer hiç alakalı belge bulunamadıysa doğrudan uydurmadan yanıt dön
        if not raw_results:
            return AskResponse(
                question=request.question,
                answer="Aranan bilgi indekslenmiş akademik belgelerde bulunamadı.",
                citations=[],
                latency_seconds=round(time.time() - start_time, 4)
            )
            
        # 2. Adım: Bağlamı (Context) ve Atıfları (Citations) oluştur
        context_parts = []
        citations = []
        
        for idx, point in enumerate(raw_results):
            source = point.payload.get("source", "unknown")
            page_number = point.payload.get("page_number", 0)
            text = point.payload.get("text", "")
            
            # LLM'e hangi bilginin nereden geldiğini net bildirmek için etiketliyoruz
            context_parts.append(f"--- [Kaynak #{idx+1}: {source}, Sayfa: {page_number}] ---\n{text}\n")
            citations.append(
                CitationItem(
                    source=source,
                    page_number=page_number,
                    score=point.score
                )
            )
            
        context_str = "\n".join(context_parts)
        
        # 3. Adım: Hugging Face API'ye gönderilecek prompt'u hazırla
        system_prompt = (
            "Sana sunulan belge içeriklerine dayanarak aşağıdaki soruyu cevapla.\n"
            "Cevabını sadece ve sadece verilen bağlam (context) bilgisine dayandır.\n"
            "Eğer verilen bağlam soruyu cevaplamak için yetersiz veya alakasızsa, "
            "kesinlikle kendi bilgilerini katma ve doğrudan 'Aranan bilgi indekslenmiş akademik belgelerde bulunamadı.' de.\n"
            "Cevap verirken mutlaka bilgi aldığın kaynak adını ve sayfa numarasını belirt."
        )

        # 4. Adım: LLM API Çağrısı (Hybrid Fallback: Gemini -> OpenRouter)
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "") or os.getenv("HF_TOKEN", "")
        
        ai_answer = None
        
        # 4a. Önce Gemini API'yi dene (eğer anahtar varsa)
        if gemini_key:
            model_name = "gemini-flash-latest"
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_key}"
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": system_prompt + "\n\nSORU:\n" + request.question}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.1
                }
            }
            
            logger.info(f"LLM üretimi deneniyor: {model_name} (Gemini)")
            
            max_retries = 2
            backoff_factor = 1.5
            response = None
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"Gemini Deneme {attempt+1}/{max_retries} yapılıyor...")
                    # Hızlı hata tespiti için timeout süresini 10 saniyeye düşürdük
                    response = requests.post(api_url, json=payload, timeout=10)
                    if response.status_code == 200:
                        break
                    
                    if response.status_code in [429, 500, 502, 503, 504]:
                        wait_time = backoff_factor ** (attempt + 1)
                        logger.warning(f"Gemini API geçici hata verdi ({response.status_code}). {wait_time:.1f} sn içinde tekrar denenecek...")
                        time.sleep(wait_time)
                    else:
                        logger.warning(f"Gemini API kritik hata verdi ({response.status_code}), fallback yapılacak.")
                        break
                except Exception as req_err:
                    logger.warning(f"Gemini API bağlantı hatası ({req_err}).")
                    if attempt == max_retries - 1:
                        break
                    wait_time = backoff_factor ** (attempt + 1)
                    time.sleep(wait_time)
            
            if response and response.status_code == 200:
                try:
                    response_json = response.json()
                    ai_answer = response_json["candidates"][0]["content"]["parts"][0]["text"]
                    logger.info("Gemini API yanıtı başarıyla alındı.")
                except Exception as parse_err:
                    logger.error(f"Gemini yanıtı ayrıştırılamadı: {parse_err}")
            else:
                logger.warning("Gemini API başarısız oldu veya zaman aşımına uğradı. OpenRouter Fallback devreye giriyor...")
        
        # 4b. Gemini başarısız olduysa veya yoksa OpenRouter yedek modelini dene
        if not ai_answer:
            if openrouter_key:
                # OpenRouter üzerinde sırayla denenecek stabil ücretsiz modeller
                models_to_try = [
                    "google/gemma-2-9b-it:free",
                    "qwen/qwen-2-7b-instruct:free",
                    "meta-llama/llama-3-8b-instruct:free"
                ]
                
                api_url = "https://openrouter.ai/api/v1/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {openrouter_key}",
                    "HTTP-Referer": "https://github.com/suleymansssardogan/Space-Paper",
                    "X-Title": "Antispace Space Science RAG"
                }
                
                for model_to_use in models_to_try:
                    logger.info(f"OpenRouter ile deneniyor: {model_to_use}")
                    
                    payload = {
                        "model": model_to_use,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": request.question}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 512
                    }
                    
                    try:
                        response = requests.post(api_url, headers=headers, json=payload, timeout=12)
                        if response.status_code == 200:
                            response_json = response.json()
                            ai_answer = response_json["choices"][0]["message"]["content"]
                            logger.info(f"OpenRouter ({model_to_use}) yanıtı başarıyla alındı.")
                            break
                        else:
                            logger.warning(f"OpenRouter ({model_to_use}) hata kodu döndürdü ({response.status_code}): {response.text}")
                    except Exception as e:
                        logger.warning(f"OpenRouter ({model_to_use}) bağlantı hatası: {e}")
            else:
                logger.error("OpenRouter API anahtarı bulunamadı (OPENROUTER_API_KEY veya HF_TOKEN eksik).")
        
        if not ai_answer:
            logger.info("API anahtarı bulunamadı veya LLM servisleri başarısız oldu, çevrimdışı mod cevabı oluşturuluyor...")
            ai_answer = (
                "⚠️ **[Çevrimdışı Mod - Yapay Zeka Cevabı Sentezlenemedi]**\n\n"
                "Sisteme geçerli bir **GEMINI_API_KEY** veya **OPENROUTER_API_KEY** tanımlanmadığı ya da LLM sağlayıcıları "
                "zaman aşımına uğradığı için yapay zeka cevap sentezi gerçekleştirilemedi. "
                "Ancak veritabanında semantik olarak en çok eşleşen akademik belgeler ve ilgili bulgular aşağıda sıralanmıştır:\n\n"
            )
            for idx, point in enumerate(raw_results):
                source = point.payload.get("source", "unknown")
                page_number = point.payload.get("page_number", 0)
                text = point.payload.get("text", "")
                
                # Format each chunk nicely
                ai_answer += f"**[{idx+1}] Kaynak: {source} (Sayfa: {page_number})**\n> {text.strip()}\n\n"
                
            ai_answer += (
                "💡 *İpucu: Tam yapay zeka cevap sentezini aktifleştirmek için lütfen projenin kök dizininde bir `.env` dosyası oluşturup içine "
                "`GEMINI_API_KEY=your_api_key` ekleyin ve sunucuyu yeniden başlatın.*"
            )
        
        latency = time.time() - start_time
        
        # Langfuse Tracing
        if langfuse:
            try:
                trace = langfuse.trace(
                    name="rag_ask_question",
                    input={"question": request.question},
                    output={"answer": ai_answer, "citations_count": len(citations)},
                    metadata={"latency_seconds": latency}
                )
                logger.info("Langfuse ask trace başarıyla gönderildi.")
            except Exception as trace_err:
                logger.warning(f"Langfuse trace gönderim hatası: {trace_err}")
        
        return AskResponse(
            question=request.question,
            answer=ai_answer,
            citations=citations,
            latency_seconds=round(latency, 4)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG işlemi sırasında sunucu hatası oluştu: {str(e)}")


@app.post("/api/v1/feedback")
def submit_feedback(request: FeedbackRequest):
    """
    Kullanıcılardan gelen beğeni/beğenmeme geri bildirimlerini toplar ve izleme (Langfuse) sistemine kaydeder.
    """
    logger.info(f"Kullanıcı geri bildirimi alındı: Soru='{request.question}' | Skor={request.score} | Yorum='{request.feedback_text}'")
    
    if langfuse:
        try:
            langfuse.score(
                name="user_feedback",
                value=float(request.score),
                comment=request.feedback_text or "General query feedback"
            )
            logger.info("Feedback skoru Langfuse'a başarıyla iletildi.")
        except Exception as fe:
            logger.warning(f"Feedback Langfuse'a gönderilemedi: {fe}")
            
    return {"status": "success", "message": "Geri bildiriminiz başarıyla kaydedildi."}


@app.post("/api/v1/ingest/daily", response_model=IngestResponse)
def trigger_daily_ingestion(request: IngestRequest):
    """
    arXiv API üzerinden günlük astrofizik makalelerini çeker ve veritabanını günceller.
    """
    try:
        logger.info(f"Arayüzden tetiklenen arXiv Ingestion başlatılıyor: {request.category}")
        from ingest_daily_arxiv import run_daily_ingestion
        
        count = run_daily_ingestion(category=request.category, max_results=request.max_results)
        
        return IngestResponse(
            status="success",
            papers_ingested=count,
            message=f"{count} adet yeni arXiv makalesi başarıyla çekildi, parçalandı ve Qdrant'a yüklendi."
        )
    except Exception as e:
        logger.error(f"Daily ingestion tetikleme hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Otomatik veri besleme hatası: {str(e)}")


# Static klasörünü FastAPI'ye bağla
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")
