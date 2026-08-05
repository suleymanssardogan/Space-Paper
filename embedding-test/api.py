import sys
import os

# Disable TensorFlow/Keras 3 conflicts and tokenizers deadlock warnings
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Script'in bulunduğu dizini Python arama yoluna ekle (Docker/Render ortamları için)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
# .env dosyasını yükle
load_dotenv()

import time
import requests
import logging
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from save_to_qdrant import SpaceScienceVectorStore


# Loglama ayarları
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fastapi")

# FastAPI uygulamasını başlat
app = FastAPI(
    title="Antispace RAG API Gateway",
    description="Space Science Academic Papers Search & Q&A API",
    version="1.2.0"
)

from fastapi.middleware.cors import CORSMiddleware

# CORS ayarları (Dışarıdan erişim için)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Gerekirse spesifik adresler eklenebilir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    source: str | None = Field(default=None, description="Filtrelenecek kaynak dosya adı (PDF)")

class SearchResultItem(BaseModel):
    text: str = Field(..., description="Belge parçası (chunk)")
    source: str = Field(..., description="Kaynak dosya adı (PDF)")
    page_number: int = Field(..., description="Sayfa numarası")
    score: float = Field(..., description="Semantik benzerlik skoru")

class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]
    latency_seconds: float = Field(..., description="Arama işleminin sunucu tarafındaki gecikme süresi")
    prefiltered_source: str | None = Field(default=None, description="Filtrelenen kaynak dosya")

# --- DAY 9: NEW SCHEMAS FOR RAG Q&A ---

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Uzay bilimleriyle ilgili sorunuz")
    limit: int = Field(default=3, ge=1, le=10, description="Bağlam olarak kullanılacak kaynak sayısı")
    score_threshold: float = Field(default=0.20, ge=0.0, le=1.0, description="Min benzerlik eşiği")
    source: str | None = Field(default=None, description="Filtrelenecek kaynak dosya adı (PDF)")

class CitationItem(BaseModel):
    source: str = Field(..., description="Kaynak dosya adı")
    page_number: int = Field(..., description="Sayfa numarası")
    score: float = Field(..., description="Benzerlik skoru")

class AskResponse(BaseModel):
    question: str
    answer: str = Field(..., description="Yapay zeka tarafından üretilen güvenilir cevap")
    citations: list[CitationItem] = Field(..., description="Cevap için kullanılan kaynak referansları")
    latency_seconds: float = Field(..., description="Toplam işlem süresi (Arama + LLM)")
    faithfulness: float | None = Field(default=None, description="Cevabın bağlama sadakat skoru (0-1)")
    answer_relevance: float | None = Field(default=None, description="Cevabın soruyla alaka skoru (0-1)")
    prefiltered_source: str | None = Field(default=None, description="Filtrelenen kaynak dosya")

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

class UploadResponse(BaseModel):
    status: str = Field(..., description="İşlem durumu (success / error)")
    filename: str = Field(..., description="Yüklenen PDF dosyasının adı")
    chunks_indexed: int = Field(..., description="Vektör veritabanına indekslenen parça sayısı")
    message: str = Field(..., description="Detaylı durum mesajı")

# --- 2. API ENDPOINTS ---

# --- Reranking (Cross-Encoder / Cohere Rerank) Mimarisi ---
_local_reranker = None

def get_local_reranker():
    global _local_reranker
    if _local_reranker is None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            logger.info("Yerel ONNX Cross-Encoder reranker yükleniyor: Xenova/ms-marco-MiniLM-L-6-v2")
            _local_reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2", threads=1)
        except Exception as e:
            logger.warning(f"Yerel ONNX reranker yüklenemedi: {e}")
            raise RuntimeError(f"Yerel ONNX reranker başlatılamadı: {e}")
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
        scores = list(model.rerank(query, docs))
        
        scored_points = []
        for idx, score in enumerate(scores):
            point = raw_results[idx]
            point.score = float(score)
            scored_points.append(point)
            
        # Skorlara göre azalan sırada diz
        scored_points.sort(key=lambda x: x.score, reverse=True)
        
        # Farklı sayfa/bölümlerden çeşitlilik sağlamak için unique (source, page) tercih et
        selected_points = []
        seen_pages = set()
        
        for p in scored_points:
            key = (p.payload.get("source", ""), p.payload.get("page_number", 0))
            if key not in seen_pages:
                seen_pages.add(key)
                selected_points.append(p)
            if len(selected_points) >= limit:
                break
                
        # Eğer yeterince farklı sayfa yoksa kalan en yüksek skorlularla tamamla
        if len(selected_points) < limit:
            for p in scored_points:
                if p not in selected_points:
                    selected_points.append(p)
                if len(selected_points) >= limit:
                    break
                    
        return selected_points
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

@app.get("/api/v1/sources", response_model=list[str])
def get_unique_sources():
    """
    Qdrant veritabanındaki tüm benzersiz kaynak (source) PDF dosyalarının listesini döner.
    """
    try:
        if not store.client.collection_exists(collection_name=COLLECTION_NAME):
            return []
        
        sources = set()
        offset = None
        while True:
            res, next_offset = store.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                with_payload=["source"],
                with_vectors=False,
                offset=offset
            )
            for point in res:
                if point.payload and "source" in point.payload:
                    sources.add(point.payload["source"])
            if not next_offset:
                break
            offset = next_offset
            
        return sorted(list(sources))
    except Exception as e:
        logger.error(f"Benzersiz kaynaklar listelenirken hata: {e}")
        raise HTTPException(status_code=500, detail=f"Kaynak listesi alınamadı: {str(e)}")

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
            score_threshold=request.score_threshold,
            source_filter=request.source
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
            latency_seconds=round(latency, 4),
            prefiltered_source=request.source if request.source else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Arama işlemi sırasında bir sunucu hatası oluştu: {str(e)}")


def evaluate_rag_response(question: str, context: str, answer: str) -> dict:
    """
    Evaluates RAG outputs using granular, critical claim verification and relevance criteria.
    Returns realistic faithfulness and answer_relevance scores between 0.00 and 1.00.
    """
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not gemini_key:
        logger.warning("Gemini API key missing, evaluation skipped.")
        return {"faithfulness": None, "answer_relevance": None}
        
    model_name = "gemini-2.5-flash"
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_key}"
    
    prompt = f"""
You are a rigorous scientific AI evaluator auditing a Retrieval-Augmented Generation (RAG) system.
Your job is to critically evaluate the provided RAG output based on the user's question, retrieved context snippets, and generated answer.

Evaluate the output on two metrics:

1. **Faithfulness (0.00 to 1.00)**:
   - Extract all factual claims in the generated answer.
   - Calculate the exact ratio of claims directly supported by the context versus unsupported or extrapolated claims.
   - If the answer contains general knowledge or details NOT explicitly present in the provided context snippets, deduct points proportionally.
   - If the answer states that the information is not found in the context or refuses to answer, score faithfulness as 0.00 (since no context evidence was utilized).
   - Score guideline:
     - 1.00: 100% of facts are explicitly substantiated by context without any extrapolation.
     - 0.70 - 0.90: Answer is accurate, but uses mild general phrasing or minor assumptions not explicitly in context.
     - 0.30 - 0.60: Significant unbacked claims, extrapolations, or partial hallucinations present.
     - 0.00: Refusal, complete hallucination, or context is totally un-utilized.

2. **Answer Relevance (0.00 to 1.00)**:
   - Measure how directly, completely, and specifically the generated answer addresses the exact user question.
   - Deduct points if the answer is too brief, misses key parts of the user's inquiry, or includes redundant fluff.
   - If the answer is a refusal ("information not found in context"), score answer_relevance as 0.00 to 0.20 (because the user's question went unanswered).
   - Score guideline:
     - 1.00: Directly, completely, and accurately answers every aspect of the user question.
     - 0.70 - 0.90: Answers the core question well, but omits secondary nuances or details requested by the user.
     - 0.30 - 0.60: Partially relevant, vague, or skirts around the primary topic.
     - 0.00 - 0.20: Entirely off-topic or refusal to answer.

Respond ONLY with a valid JSON object matching this exact schema:
{{"reasoning": "<short critical analysis>", "faithfulness": <float between 0.00 and 1.00>, "answer_relevance": <float between 0.00 and 1.00>}}

CONTEXT:
{context}

USER QUESTION:
{question}

GENERATED ANSWER:
{answer}
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json"
        }
    }
    
    try:
        logger.info("RAG cevabı Gemini ile değerlendiriliyor...")
        response = requests.post(api_url, json=payload, timeout=8)
        if response.status_code == 200:
            import json
            response_json = response.json()
            raw_text = response_json["candidates"][0]["content"]["parts"][0]["text"].strip()
            scores = json.loads(raw_text)
            logger.info(f"Gemini Değerlendirme Skorları & Analiz: {scores}")
            return {
                "faithfulness": round(float(scores.get("faithfulness", 0.0)), 2),
                "answer_relevance": round(float(scores.get("answer_relevance", 0.0)), 2)
            }
        else:
            logger.warning(f"Değerlendirme API hatası: {response.status_code} - {response.text}")
    except Exception as e:
        logger.warning(f"Değerlendirme sırasında hata oluştu: {e}")
        
    return {"faithfulness": None, "answer_relevance": None}


@app.post("/api/v1/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    RAG (Retrieval-Augmented Generation) kullanarak soruya sadece veritabanındaki belgelere dayanarak cevap üretir.
    """
    try:
        start_time = time.time()
        
        # 1. Adım: Qdrant'tan geniş bağlam adaylarını (limit * 5, min 15) ara
        candidate_limit = max(request.limit * 5, 15)
        raw_results = store.search_documents(
            collection_name=COLLECTION_NAME,
            query=request.question,
            limit=candidate_limit,
            score_threshold=request.score_threshold,
            source_filter=request.source
        )
        
        # 2. Adım: Cross-Encoder ile en alakalı makale parçalarını Rerank et
        raw_results = rerank_documents(request.question, raw_results, request.limit)
        
        # Eğer hiç alakalı belge bulunamadıysa doğrudan uydurmadan yanıt dön
        if not raw_results:
            return AskResponse(
                question=request.question,
                answer="Aranan bilgi indekslenmiş akademik belgelerde bulunamadı.",
                citations=[],
                latency_seconds=round(time.time() - start_time, 4),
                prefiltered_source=request.source if request.source else None
            )
            
        # 2. Adım: Bağlamı (Context) ve Atıfları (Citations) oluştur
        context_parts = []
        citations = []
        
        for idx, point in enumerate(raw_results):
            source = point.payload.get("source", "unknown")
            page_number = point.payload.get("page_number", 0)
            text = point.payload.get("text", "")
            
            # LLM'e hangi bilginin nereden geldiğini net bildirmek için etiketliyoruz
            context_parts.append(f"--- [Source #{idx+1}: {source}, Page: {page_number}] ---\n{text}\n")
            citations.append(
                CitationItem(
                    source=source,
                    page_number=page_number,
                    score=point.score
                )
            )
            
        context_str = "\n".join(context_parts)
        
        # 3. Adım: LLM Prompt'unu hazırla
        system_prompt = (
            "You are Antispace, an expert space science research assistant.\n"
            "Answer the user's question based strictly on the provided academic context snippets.\n"
            "Provide a clear, detailed, grounded scientific answer using the information in the context.\n"
            "Every fact or claim in your response MUST be cited inline using the exact format: (source_filename.pdf, Page: X).\n"
            "Respond in the same language as the user's query.\n"
            "If the context snippets do not contain sufficient evidence to answer the question, state: "
            "'The retrieved index documents do not contain specific evidence regarding this question.'"
        )

        # 4. Adım: LLM API Çağrısı (Hybrid Fallback: Gemini -> OpenRouter)
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "") or os.getenv("HF_TOKEN", "")
        
        ai_answer = None
        
        # 4a. Önce Gemini API'yi dene (eğer anahtar varsa)
        if gemini_key:
            model_name = "gemini-2.5-flash"
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_key}"
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": f"{system_prompt}\n\nBAĞLAM (CONTEXT):\n{context_str}\n\nSORU:\n{request.question}"}
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
                            {"role": "system", "content": f"{system_prompt}\n\nBAĞLAM (CONTEXT):\n{context_str}"},
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
        
        # Ragas benzeri değerlendirme yapalım
        eval_scores = {"faithfulness": None, "answer_relevance": None}
        # Sadece geçerli bir cevap alındıysa değerlendir
        if ai_answer and "Aranan bilgi indekslenmiş akademik belgelerde bulunamadı" not in ai_answer and citations:
            eval_scores = evaluate_rag_response(
                question=request.question,
                context=context_str,
                answer=ai_answer
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
                if eval_scores.get("faithfulness") is not None:
                    langfuse.score(
                        trace_id=trace.id,
                        name="faithfulness",
                        value=eval_scores["faithfulness"]
                    )
                if eval_scores.get("answer_relevance") is not None:
                    langfuse.score(
                        trace_id=trace.id,
                        name="answer_relevance",
                        value=eval_scores["answer_relevance"]
                    )
                logger.info("Langfuse ask trace ve değerlendirme skorları başarıyla gönderildi.")
            except Exception as trace_err:
                logger.warning(f"Langfuse trace gönderim hatası: {trace_err}")
        
        return AskResponse(
            question=request.question,
            answer=ai_answer,
            citations=citations,
            latency_seconds=round(latency, 4),
            faithfulness=eval_scores.get("faithfulness"),
            answer_relevance=eval_scores.get("answer_relevance"),
            prefiltered_source=request.source if request.source else None
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


@app.post("/api/v1/upload", response_model=UploadResponse)
async def upload_pdf_file(file: UploadFile = File(...)):
    """
    Kullanıcıların kendi PDF akademik makalelerini sisteme yüklemesini, 
    parçalamasını (chunking) ve Qdrant Cloud veritabanına anında vektörleştirmesini sağlar.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Lütfen sadece .pdf uzantılı akademik belgeler yükleyin.")
        
    try:
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, file.filename)
        
        logger.info(f"Yüklenen dosya kaydediliyor: {file_path}")
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
            
        from ingest_pdfs import DocumetPipeline
        from ingest_to_qdrant import bulk_upsert_chunks
        
        pipeline = DocumetPipeline(download_dir=data_dir)
        raw_chunks = pipeline.extract_text_from_pdf(file_path)
        
        if not raw_chunks:
            raise HTTPException(status_code=400, detail="PDF belgesinden metin çıkarılamadı.")
            
        logger.info(f"PDF'den {len(raw_chunks)} parça çıkarıldı, Qdrant'a yükleniyor...")
        bulk_upsert_chunks(store, COLLECTION_NAME, raw_chunks, batch_size=32)
        
        return UploadResponse(
            status="success",
            filename=file.filename,
            chunks_indexed=len(raw_chunks),
            message=f"'{file.filename}' belgesi başarıyla yüklendi ve {len(raw_chunks)} parça Qdrant'a indekslendi."
        )
    except Exception as e:
        logger.error(f"PDF yükleme ve vektörleştirme hatası: {e}")
        raise HTTPException(status_code=500, detail=f"PDF yükleme hatası: {str(e)}")


# Static klasörünü FastAPI'ye bağla
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")
