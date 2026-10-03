# 🌌 Antispace: Uzay Bilimleri RAG Asistanı

**Canlı Site:** https://space-paper.onrender.com

Antispace, arXiv makaleleri, NASA raporları ve JWST dokümantasyonu üzerinde çalışan bir **RAG (Retrieval-Augmented Generation)** asistanı. FastAPI, Qdrant (hibrit dense+sparse arama), FastEmbed/ONNX, Cohere Rerank ve çok katmanlı Gemini/OpenRouter LLM fallback zinciri üzerine kurulu; cevaplarını **sadece** veritabanındaki kaynaklara dayandırıp her iddiayı sayfa numarasıyla referanslıyor.

## 📖 Bu Doküman Nasıl Okunmalı?

Bu README bir kurulum kılavuzu değil — **projeyi bir mülakatta baştan sona anlatıyormuş gibi** yazıldı. Amaç sadece "ne yaptım" sorusunu geçmek değil, "neden böyle yaptım", "alternatifleri neden elemedim" ve "nerede zorlandım, nasıl çözdüm" sorularına da hazırlıklı olmak. Mülakat öncesi tekrar ederken her bölüm, karşına çıkabilecek tipik bir soruya karşılık geliyor:

| Bölüm | Karşılık geldiği mülakat sorusu |
|---|---|
| 1️⃣ Problem Neydi? | "Bu projeyi neden yaptın, hangi ihtiyaçtan doğdu?" |
| 2️⃣ Sistem Baştan Sona Nasıl Çalışıyor? | "Mimariyi uçtan uca anlatır mısın?" |
| 3️⃣ Kritik Teknik Kararlar ve Gerekçeleri | "Neden X değil de Y'yi seçtin?" |
| 4️⃣ Karşılaştığım Zorluklar ve Nasıl Çözdüm | "En çok nerede zorlandın, nasıl aştın?" |
| 5️⃣ Sonuçlar Nasıl Ölçülüyor? | "Başarıyı nasıl ölçtün, elinde kanıt var mı?" |
| 6️⃣ Ne Eksik / Gelecekte Ne Yapardım | "Zamanın/kaynağın olsa neyi değiştirir, neyi eklerdin?" |

Yani doküman yukarıdan aşağı okunduğunda, bir mülakatın doğal akışını (motivasyon → mimari → kararlar → zorluklar → sonuç → eleştiri) tekrar etmiş oluyorsun.

### Ekran Görüntüleri

**Grounded Q&A — sayfa atıflı, sadece kaynağa dayalı cevap:**
![Grounded Q&A cevabı](docs/screenshots/grounded-qa-answer.png)

**Gelişmiş arama filtreleri — kaynak PDF seçimi, top-k ve benzerlik eşiği:**
![Gelişmiş arama filtreleri](docs/screenshots/advanced-search-filters.png)

---

## 🎤 Tek Cümlelik Özet (Elevator Pitch)

> "Antispace, arXiv makaleleri, NASA raporları ve JWST dokümantasyonu üzerinde çalışan; cevaplarını **sadece** veritabanındaki kaynaklara dayandıran, her iddiayı sayfa numarasıyla kaynak gösteren, halüsinasyon riskini mimari seviyede minimize eden production-grade bir RAG (Retrieval-Augmented Generation) sistemi."

---

## 1️⃣ Problem Neydi?

Uzay bilimleri / astrofizik literatüründe bir LLM'e doğrudan soru sormanın üç somut riski var:

1. **Bilgi kesintisi:** Model, dün yayınlanan bir arXiv makalesinden habersiz.
2. **Halüsinasyon:** Sayısal veriler, formüller, misyon detayları gibi teknik konularda model kendinden emin ama yanlış cevaplar üretebiliyor.
3. **Doğrulanamazlık:** Cevabın hangi makaleye, hangi sayfaya dayandığı bilinmiyor — akademik/teknik bir bağlamda bu kabul edilemez.

Bunu şöyle özetliyorum: **"Model bilgiyi hatırlamaya değil, doğru yerden bulup okumaya zorlanmalı."** RAG mimarisini seçmemin temel gerekçesi bu.

---

## 2️⃣ Sistem Baştan Sona Nasıl Çalışıyor (Adım Adım)

Sistemi iki ana hat üzerinden anlatıyorum: **veri toplama (ingestion)** ve **sorgu/cevap üretimi (query pipeline)**.

```mermaid
flowchart TD
    subgraph Ingestion [1. Veri Toplama Hattı]
        direction TB
        Cron[GitHub Actions Cron / 05:00 TR] -->|arXiv API| Scrap[Yeni Makaleleri Bul]
        Scrap -->|PDF indir| PDF[PyPDF ile Metin Çıkar]
        PDF -->|800 karakter / 150 overlap| Split[RecursiveCharacterTextSplitter]
        Split -->|Dense + Sparse| FE[FastEmbed / ONNX Embedding]
        FE -->|UUID5 hash| UUID[Deterministik ID Üretimi]
        UUID -->|Upsert| QC[(Qdrant Cloud)]
    end

    subgraph Query [2. Sorgu ve Cevap Üretim Hattı]
        direction TB
        User([Kullanıcı Sorusu]) --> API[FastAPI Gateway]
        API -->|Dense + Sparse vektörleştir| FE_Q[FastEmbed]
        FE_Q -->|Prefetch x2 + RRF Fusion| QC
        QC -->|İlk ~15-20 aday| Rerank{Rerank}
        Rerank -->|Öncelik| Cohere[Cohere Rerank API]
        Rerank -->|Fallback| CE[Yerel ONNX Cross-Encoder]
        Cohere --> Prompt[Strict Grounding Prompt]
        CE --> Prompt
        Prompt --> LLM{LLM}
        LLM -->|1. Öncelik| Gemini[Gemini 2.5 Flash]
        LLM -->|2. Fallback| OR[OpenRouter Ücretsiz Modeller]
        LLM -->|3. Fallback| Offline[Ham Kaynak Metinleri]
        Gemini --> Eval[LLM-as-Judge: Faithfulness + Relevance]
        Eval --> Res[Kaynak Atıflı Cevap]
    end
```

### Adım 1 — Veri Toplama: Her sabah kendi kendine güncellenen bir veritabanı

Her gün TR saatiyle 05:00'te (UTC 02:00) bir **GitHub Actions cron job** tetikleniyor (`.github/workflows/daily_ingest.yml`). Bu job arXiv API'sini sorgulayıp `astro-ph.CO` ve `astro-ph.EP` kategorilerindeki en yeni makaleleri buluyor, PDF'lerini indirip metne çeviriyor.

*Neden GitHub Actions?* Ayrı bir sunucu/worker maliyetine girmeden, versiyon kontrolüyle birlikte yaşayan, ücretsiz ve izlenebilir bir cron altyapısı sağlıyor.

### Adım 2 — Chunking: Metni modele "sindirilebilir" parçalara bölmek

`RecursiveCharacterTextSplitter` ile her PDF **800 karakterlik, 150 karakter overlap'li** parçalara bölünüyor. Overlap, bir cümlenin/argümanın parça sınırında ikiye bölünüp anlamını kaybetmesini önlüyor.

### Adım 3 — Embedding: Hem "anlamı" hem "kelimeyi" yakalamak (Dense + Sparse)

Her parça için iki farklı vektör üretiliyor:
- **Dense vektör** (`all-MiniLM-L6-v2`, 384 boyut): semantik/anlamsal benzerlik için.
- **Sparse vektör** (`Qdrant/bm25`): tam kelime eşleşmesi (ör. "Stephan's Quintet" gibi özel isimler, kısaltmalar) için.

*Neden ikisi birden?* Dense arama parafrazları yakalamakta iyi ama nadir geçen özel terimlerde (misyon adları, enstrüman kodları) zayıf kalabiliyor. Sparse (BM25) tam tersi. İkisini birleştirmek tek başına hiçbirinin veremeyeceği bir kapsama alanı sağlıyor.

Embedding modelleri **FastEmbed** (ONNX runtime) ile çalıştırılıyor — PyTorch'a göre çok daha düşük bellek ayak izi, bu da Render'ın ücretsiz/düşük katmanındaki 512MB-1GB RAM sınırında hayati önem taşıyordu (aşağıda "zorluklar" bölümünde detaylandırıyorum).

### Adım 4 — Idempotent Yükleme: Aynı veriyi iki kere işlememek

Her chunk için kaynak dosya adı, sayfa numarası, chunk sırası ve metninden **deterministik bir UUID5** üretiliyor. Böylece farklı kaynaklardaki aynı metinler birbirinin atıf bilgisini ezmiyor. Böylece aynı makale ya da chunk ikinci kez işlense bile Qdrant'ta duplicate kayıt oluşmuyor — upsert doğal olarak "varsa güncelle, yoksa ekle" davranışı gösteriyor. Günlük cron'un sürekli çalıştığı bir sistemde bu, veri bütünlüğü için kritik.

### Adım 5 — Sorgu Zamanı: Hibrit Arama + RRF Füzyonu

Kullanıcı soru sorduğunda, aynı dense+sparse vektörleştirme sorguya da uygulanıyor. Qdrant'ın `query_points` API'sinde **iki paralel prefetch** çalıştırılıyor (dense top-N, sparse top-N) ve sonuçlar **RRF (Reciprocal Rank Fusion)** ile tek bir sıralı listede birleştiriliyor. Kullanıcı arayüzden belirli bir PDF seçtiyse, bu adımda `source` alanına göre **pre-filtering** de uygulanıyor (aramayı o dokümanla sınırlıyor).

### Adım 6 — Reranking: İlk sıradaki sonuçların gerçekten en alakalı olduğundan emin olmak

Hibrit aramadan gelen ilk ~15-20 aday, ikinci bir modelle yeniden puanlanıyor:
- **Öncelik:** Cohere Rerank API (`rerank-english-v3.0`) — daha güçlü, cloud tabanlı.
- **Fallback:** Cohere anahtarı yoksa veya API hata verirse, yerel bir **ONNX Cross-Encoder** (`Xenova/ms-marco-MiniLM-L-6-v2`) devreye giriyor.

*Neden ayrı bir rerank adımı?* İlk aşamadaki vektör araması hız için optimize; rerank ise doğruluk için — sorgu ve dokümanı birlikte (cross-attention) değerlendirdiği için çok daha isabetli ama daha yavaş. Bu yüzden önce ucuz/hızlı yöntemle adayları daraltıp, pahalı/yavaş yöntemi sadece o küçük kümeye uyguluyorum.

### Adım 7 — Cevap Üretimi: "Sadece bağlamdan oku" prensibi

Seçilen en iyi 3 parça, kaynak dosya adı ve sayfa numarasıyla etiketlenip LLM'e "strict grounding" bir sistem promptuyla veriliyor. Prompt açıkça şunu talep ediyor:
- Her iddia `(kaynak.pdf, Page: X)` formatında **inline** kaynak göstermeli.
- Bağlamda yeterli kanıt yoksa model **açıkça "bulunamadı" demeli**, uydurmamalı.

LLM tarafında **3 katmanlı fallback** var: **Gemini 2.5 Flash** (birincil, retry + exponential backoff ile) → başarısız olursa **OpenRouter** üzerinden ücretsiz modeller (Gemma, Qwen, Llama sırayla denenir) → o da yoksa **çevrimdışı mod** (ham, en alakalı kaynak metinleri doğrudan kullanıcıya gösterilir, hiç "uydurma" riski alınmaz).

*Neden tek bir LLM'e bağımlı kalmadım?* Ücretsiz/tek API kotalarına bağlı bir sistemde tek sağlayıcı = tek arıza noktası. Kademeli fallback, servis kesintisinde bile kullanıcıya boş ekran değil, en azından ham kaynak veriyi gösterebiliyor.

### Adım 8 — Kendi Kendini Denetleme: Runtime RAGAs Değerlendirmesi

Her cevap üretildikten sonra, **Gemini'yi bir "hakem" (LLM-as-a-judge) olarak** ikinci kez çağırıyorum: üretilen cevaptaki her iddianın bağlamda gerçekten var olup olmadığını (**Faithfulness**) ve cevabın soruyu ne kadar tam karşıladığını (**Answer Relevance**) 0-1 arası puanlıyor. Bu skorlar statik/sahte değil, her istekte anlık hesaplanıyor ve kullanıcıya gösteriliyor.

### Adım 9 — İzlenebilirlik: Langfuse + Kullanıcı Geri Bildirimi

Her sorgunun vektör arama süresi, rerank süresi, LLM çağrı süresi ve token kullanımı **Langfuse**'a asenkron olarak gönderiliyor. Kullanıcının beğen/beğenme butonları da aynı trace'e bağlanıyor — böylece "hangi tür sorularda sistem kötü cevap veriyor" sorusu geriye dönük analiz edilebiliyor.

---

## 3️⃣ Kritik Teknik Kararlar ve Gerekçeleri

Mülakatta "neden X değil de Y?" sorularına hazır olmak için:

| Karar | Alternatif | Neden bu seçim |
|---|---|---|
| Qdrant | Pinecone, Weaviate | Hem dense hem native sparse vektör + RRF fusion'ı tek sorguda destekliyor, self-host/cloud esnekliği var |
| FastEmbed (ONNX) | sentence-transformers (PyTorch) | Çok daha düşük RAM/CPU ayak izi — Render'ın kısıtlı belleğinde stabil çalışmak için zorunluydu |
| Hibrit arama (Dense+Sparse+RRF) | Sadece dense | Özel isim/kısaltma ağırlıklı akademik literatürde tek başına dense arama kelime eşleşmelerini kaçırıyor |
| İki aşamalı rerank (Cohere→local) | Tek sağlayıcı | Kota/kesinti durumunda sistemin tamamen durmaması |
| Çoklu LLM fallback | Tek LLM | Ücretsiz kota/rate-limit riskine karşı sürekli çalışırlık |
| UUID5 ile idempotent upsert | Auto-increment ID | Günlük cron aynı veriyi tekrar işlese bile duplicate oluşmasın diye |

---

## 4️⃣ Karşılaştığım Zorluklar ve Nasıl Çözdüm

Bunlar somut, anlatılabilir hikayeler:

**a) Render'da bellek yetersizliği (Out-of-Memory)**
Embedding ve reranker modellerini varsayılan ayarlarla çalıştırdığımda, düşük RAM'li instance'ta konteyner OOM (out-of-memory) hatasıyla çöküyordu. Çözüm: FastEmbed modellerini `threads=1` ile sınırlamak (paralel thread'lerin bellek/CPU patlamasını önlemek) ve ONNX tabanlı reranker'a geçmek. Ayrıca ingestion sonrası `gc.collect()` ile belleği anında serbest bırakıyorum.

**b) Deploy sonrası eski arayüzün önbellekten gelmesi**
UI'yi (Türkçe/koyu temadan İngilizce/açık temaya) yeniden tasarladıktan sonra bazı kullanıcılar hâlâ eski sürümü görüyordu. Kök neden: statik dosyalar (`index.html`, `style.css`, `app.js`) için `Cache-Control` header'ı yeterince katı değildi (`no-cache` revalidation gerektiriyor ama önceki deploy'larda hiç cache header'ı yokken önbelleğe alınmış eski kopyalar buna tabi değildi). Çözümü `no-store, no-cache, must-revalidate` + `Pragma`/`Expires` header'larına genişleterek, tarayıcının bu dosyaları **hiç önbelleğe almadan** her seferinde sunucudan taze çekmesini sağladım (`embedding-test/api.py`).

**c) Halüsinasyonu mimari seviyede engellemek**
Sadece "uydurma" demek yetmiyor — prompt'ta modele bağlam yetersizse **açıkça refuze etmesi** talimatı verildi, ve değerlendirme katmanında bir refusal cevabı otomatik olarak Faithfulness=0 alacak şekilde puanlanıyor. Yani sistem "kaçamak" cevapları da ölçülebilir kılıyor, sessizce görmezden gelmiyor.

**d) Hız (latency) ile doğruluk arasındaki denge**
Rerank adımı doğruluğu artırıyor ama gecikme ekliyor. Bunu, vektör aramada geniş bir aday havuzu (limit×4-5) çekip, sadece bu havuza pahalı rerank uygulayarak; nihai LLM'e ise sadece en iyi 3 parçayı göndererek dengelemeye çalıştım.

---

## 5️⃣ Sonuçlar Nasıl Ölçülüyor?

- **Faithfulness / Answer Relevance:** Her sorguda gerçek zamanlı, LLM-as-judge ile üretilen skorlar (statik değil).
- **Kaynak atıfları:** Her cevap `[dosya.pdf, Sayfa: X]` formatında doğrulanabilir referanslarla geliyor.
- **Langfuse trace'leri:** Bottleneck analizi (arama mı, rerank mi, LLM mi yavaş) ve kullanıcı geri bildirimiyle çapraz doğrulama.

---

## 6️⃣ Ne Eksik / Gelecekte Ne Yapardım

1. **Multimodal RAG:** Şu an grafik/tablo gibi görsel veriler sadece metin olarak okunuyor; JWST/Kepler makalelerindeki şekilleri de Gemini'nin görsel anlama yeteneğiyle vektörleştirmek isterdim.
2. **Agentic self-correction:** Faithfulness skoru düşük çıkan cevaplarda, kullanıcıya göstermeden önce sistemin otomatik olarak aramayı genişletip (query expansion) yanıtı düzeltmesi.
3. **Semantik chunking:** Şu an sabit karakter sayısına göre bölüyorum; paragraf/konu geçişlerine duyarlı semantik bölme, bağlam bütünlüğünü artırırdı.

---

## ⚙️ Teknik Referans (Hızlı Başlangıç)

### API Uç Noktaları
- `GET /api/v1/health` — Sağlık durumu ve veritabanı bağlantı kontrolü.
- `POST /api/v1/search` — Ham semantik/hibrit arama (kaynak filtreleme destekli).
- `POST /api/v1/ask` — Uçtan uca RAG sorgusu (kaynak atıfları + runtime RAGAs skorları döner).
- `POST /api/v1/feedback` — Kullanıcı geri bildirim kaydı.
- `POST /api/v1/ingest/daily` — Yeni arXiv makalelerini çekmek için manuel tetikleyici.

### Çevre Değişkenleri (`.env`)
```env
QDRANT_URL=https://your-qdrant-cluster.io
QDRANT_API_KEY=your_qdrant_api_key
GEMINI_API_KEY=your_gemini_api_key
COHERE_API_KEY=your_cohere_key (isteğe bağlı)
LANGFUSE_PUBLIC_KEY=your_public_key (isteğe bağlı)
LANGFUSE_SECRET_KEY=your_secret_key (isteğe bağlı)
```

### Docker ile Çalıştırma
```bash
docker compose up --build
```
Uygulama arayüzüne `http://localhost:8000` adresinden erişilebilir.

### Manuel Veri Yükleme ve Değerlendirme
```bash
# Yerel PDF'leri yüklemek için
python embedding-test/ingest_to_qdrant.py

# arXiv'den güncel makaleleri çekmek için
python embedding-test/ingest_daily_arxiv.py

# RAG performans değerlendirme testini çalıştırmak için
python embedding-test/evaluate_rag.py
```

### Regresyon kontrolleri

```bash
python -m unittest discover -s tests -v
node --check embedding-test/static/app.js
```

Testler model indirmeden ve canlı Qdrant verisine yazmadan; kaynak kimliğini, yarıda kalan PDF indirmelerinin temizlenmesini ve kaynak filtresinin iki arama koluna da uygulanmasını kontrol eder.

**Mevcut koleksiyonlar için:** Kaynak bilgisini koruyan yeni UUID biçimi eski metin temelli ID'lerden farklıdır. Eski koleksiyona yeniden ingestion yapmak eski kayıtların yanında yeni kayıtlar oluşturabilir. Tam geçiş için yedek alınarak boş bir koleksiyona yeniden indeksleme yapılmalıdır; uygulama mevcut kayıtları otomatik silmez.

### API koruması ve istek sınırı

`.env` / deploy ortamında aşağıdaki ayarlar kullanılır:

```env
API_ACCESS_KEY=
INGEST_API_KEY=
API_RATE_LIMIT=20
API_RATE_WINDOW_SECONDS=60
EVIDENCE_MIN_COSINE=0.35
```

- `API_ACCESS_KEY` doluysa arama, soru-cevap, kaynak listesi ve geri bildirim istekleri `X-API-Key` başlığı gerektirir. Boş bırakılması herkese açık demo modudur; istek sınırı bu modda da çalışır.
- Manuel ingestion ayrı `INGEST_API_KEY` gerektirir. Anahtar tanımlı değilse endpoint 503, yanlış veya eksik anahtarda 401 döner. GitHub Actions ingestion betiği bu HTTP korumasından bağımsız çalışır.
- Arayüzde Advanced Search altında iki ayrı anahtar alanı bulunur. Anahtarlar tarayıcı depolamasına kaydedilmez; sayfa yenilendiğinde silinir. Sunucu anahtarları arayüz koduna gömülmez.
- Varsayılan sınır istemci başına 60 saniyede 20 istektir; limit aşılırsa 429 ve `Retry-After` döner. Sağlık kontrolü ve OPTIONS istekleri muaf tutulur. İstek gövdesi en fazla 64 KiB olabilir.
- Limiter süreç belleğinde tutulur ve yeniden başlatmada sıfırlanır. Birden fazla worker/replica için Redis gibi ortak bir limiter gerekir. Proxy arkasında yalnızca güvenilen proxylerin adreslerini Uvicorn'un `forwarded-allow-ips` ayarıyla tanımlayın; internetten gelen her forwarding başlığına güvenmeyin. Aynı NAT IP'sini paylaşan kullanıcılar aynı kotayı paylaşır.

### Kanıt yetersizliğinde cevap vermeme

Soru-cevap hattı, yeniden sıralamadan önce aday chunk'ların gerçek dense cosine benzerliğini hesaplar. Eşik `max(EVIDENCE_MIN_COSINE, kullanıcının score_threshold değeri)` olarak uygulanır; kullanıcı sunucudaki alt sınırı düşüremez. Kaynak adı/sayfa bilgisi eksik veya metni boş chunk'lar kanıt olarak kabul edilmez. Uygun chunk yoksa hiçbir LLM çağrısı yapılmaz.

Üretilen cevabın en az bir `(dosya.pdf, Page: X)` atfı olmalı ve bulunan tüm atıflar gönderilen bağlamdaki dosya/sayfalarla eşleşmelidir. Geçersiz atıfta cevap yerine ret döner. Yanıt şeması `refused`, `refusal_reason` ve `answer_mode` (`generated`, `extractive`, `refusal`) alanlarını içerir. Çevrimdışı kaynak gösterimi üretilmiş cevap gibi değerlendirilmez.

Cosine eşiği ve atıf doğrulaması anlamsal doğruluğu garanti etmez. Yüksek benzerlikteki bir parça yine de sorunun cevabını içermeyebilir; eşik değerlendirme sonuçlarına göre kalibre edilmelidir. Atıf kontrolü her iddianın doğruluğunu ayrıca doğrulamaz.

### Karşılaştırmalı değerlendirme

`evaluation/questions.json` 20 kaynak sorusu ve 10 ret sorusundan oluşan bir başlangıç setidir. İki temel PDF (`jwst_performance.pdf`, `kepler_mission.pdf`) indekslenmiş olmalıdır. Etiketler kaynak dosyası düzeyindedir; yayınlanmış, insan tarafından doğrulanmış bir benchmark değildir. CV'de sonuç kullanmadan önce soruların gerçekten kaynaklarda cevaplanabildiğini elle kontrol edin ve seti geliştirin.

```bash
# Çalışan yerel API üzerinden retrieval karşılaştırması; LLM çağrısı yapmaz
python embedding-test/evaluate_rag.py --skip-rag

# Retrieval karşılaştırmasına ek olarak ret / cevap üretme ölçümü; LLM kotası kullanır
python embedding-test/evaluate_rag.py
```

Erişim anahtarı `.env` veya ortamdan okunur. Varsayılan 3.1 saniyelik istek aralığı demo kotasına uygundur. Önceden kullanılan kota varsa 429 görülebilir; araç bunları hata olarak raporlar. Kaynak dosyaları eksikse ölçüme başlamaz.

`/api/v1/search` isteğinde `retrieval_mode` alanı `dense`, `hybrid` veya `hybrid_rerank` olabilir (varsayılan sonuncusu). Araç aynı soru setini üç yöntemle karşılaştırıp `evaluation/results/latest.json` raporuna ham sonuçları ve şu ölçümleri kaydeder:

- **Source Hit@k:** beklenen dosyalardan en az birini ilk k chunk'ta bulan soruların oranı.
- **Source Recall@k:** bulunan beklenen dosyaların tüm beklenen dosyalara oranı.
- **Source MRR@k:** ilk doğru dosyanın chunk sırasının tersinin ortalaması.
- **Gecikme:** başarılı istekler için ortalama ve p95; hatalar ayrıca sayılır.
- **Refusal recall / false refusal rate:** cevaplanamaz soruları reddetme ve cevaplanabilir soruları yanlış reddetme oranı.
- **Answerable response rate:** cevaplanabilir sorularda başarılı, üretilmiş cevap oranı; extractive fallback ayrı sayılır.

Retrieval hataları başarı ölçümlerinin paydasında kalır. Ret ölçümlerinin yanında hata sayısını da değerlendirin. Bu ölçümler kaynak bulmayı ve ret davranışını ölçer; cevap doğruluğunu tek başına ölçmez. Üç yöntem aynı k ve retrieval eşiğiyle çalışır; RAG için sunucudaki kanıt alt sınırı ayrıca geçerlidir.

CI her push/PR'da dış servis veya model indirmesi gerektirmeyen testleri ve sözdizimi kontrollerini çalıştırır. Yerel test bağımlılıkları: `pip install -r requirements-test.txt`.
