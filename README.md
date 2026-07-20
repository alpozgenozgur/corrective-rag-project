# Corrective RAG (CRAG)

> A self-checking retrieval-augmented generation pipeline built with LangGraph. It grades its own retrieved documents, falls back to web search when they're irrelevant, and verifies its answers for hallucination and relevance before returning them.

[Türkçe](#türkçe) | [English](#english)

---

## English

### Why?

A plain RAG pipeline has a blind spot: it retrieves whatever is closest in vector space and answers from it — even when the retrieved documents are off-topic, and even when the model hallucinates beyond what they say. There's no check.

Corrective RAG adds that check. Before answering, it grades whether the retrieved documents are actually relevant. If they aren't, it doesn't force an answer from bad context — it searches the web instead. After generating, it verifies the answer twice: is it grounded in the documents (no hallucination), and does it actually address the question? Only then does it return.

### How it works

The pipeline is a **LangGraph state graph** — not a linear chain. Nodes can branch and loop based on grading decisions:
              Question
                 │
              [route_question]  ──► off-topic ──► web search
                 │ relevant
              [retrieve]
                 │
          [grade_documents]  ──► any irrelevant? ──► [web_search]
                 │ all relevant                          │
              [generate]  ◄────────────────────────────┘
                 │
    [hallucination check] ──► not grounded ──► regenerate
                 │ grounded
      [answer relevance] ──► doesn't answer ──► web search + retry
                 │ answers

Answer


Each decision point is a small LLM-based grader with a structured (Pydantic) output — a yes/no judgment the graph routes on.

### The graders

| Grader | Question it answers | Action if it fails |
|---|---|---|
| **Router** | Is this question about the indexed topics? | Send to web search |
| **Retrieval grader** | Is each retrieved document relevant? | Drop it; if any fail, trigger web search |
| **Hallucination grader** | Is the answer grounded in the documents? | Regenerate |
| **Answer grader** | Does the answer actually address the question? | Web search and retry |

### Design decisions

**Grade before answering, not after.** Filtering irrelevant documents before generation stops bad context from producing a confident-but-wrong answer in the first place.

**Web search as a fallback, not a default.** When the knowledge base doesn't cover a question, forcing an answer from weak retrieval is worse than admitting the gap. The graph routes to web search instead of hallucinating.

**Two-stage answer verification.** Grounding and relevance are separate failures: an answer can be faithful to the documents but not address the question, or address the question while drifting beyond the sources. Each is checked independently.

**Batched, quota-aware embedding.** The free-tier embedding quota is consumed per text sent, not per request — a single large batch can exhaust it instantly. Ingestion embeds in small batches (20 docs) with pacing delays and automatic retry-on-429, so it stays under the per-minute limit. Embeddings persist to `.chroma`; the source pages are scraped and embedded only once.

### Stack

- **LangGraph** — stateful, branching graph orchestration
- **LangChain** — RAG components
- **Google Gemini** — generation and grading
- **Gemini Embedding** — vectorization
- **ChromaDB** — vector store with disk persistence
- **Tavily** — web search fallback
- **Pydantic** — structured grader outputs

### Knowledge base

Indexed from three of Lilian Weng's posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.

### Setup

```bash
git clone https://github.com/alpozgenozgur/corrective-rag-project.git
cd corrective-rag-project

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

Create a `.env` file:

GOOGLE_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here


Run:

```bash
python main.py
```

The first run scrapes and embeds the source pages (paced to respect the embedding quota). Later runs reuse the persisted vector store.

### Limitations

- Grading adds latency and token cost — every query runs several LLM calls, not one.
- Retrieval quality is bounded by the indexed pages; web search covers gaps but isn't a full substitute.
- Free-tier embedding quota makes the first ingestion slow by design (intentional pacing).

---

## Türkçe

### Neden?

Sıradan bir RAG hattının kör noktası vardır: vektör uzayında en yakın olanı getirir ve ona göre cevap verir — getirilen belgeler konu dışı olsa bile, model belgelerin söylediğinin ötesine geçip uydursa bile. Hiçbir denetim yoktur.

Corrective RAG bu denetimi ekler. Cevap vermeden önce, getirilen belgelerin gerçekten alakalı olup olmadığını değerlendirir. Alakalı değilse, kötü bağlamdan cevap üretmeye zorlamaz — bunun yerine web'de arar. Cevabı ürettikten sonra iki kez doğrular: belgelere dayanıyor mu (hallüsinasyon yok mu) ve soruyu gerçekten yanıtlıyor mu? Ancak o zaman cevabı döndürür.

### Nasıl çalışıyor?

Hat, düz bir zincir değil, bir **LangGraph durum grafiği.** Düğümler, değerlendirme kararlarına göre dallanabilir ve döngü yapabilir:
                Soru
                 │
              [yönlendir]  ──► konu dışı ──► web araması

│ alakalı
[belge getir]
│
[belgeleri değerlendir] ──► alakasız var mı? ──► [web araması]
│ hepsi alakalı │
[cevap üret] ◄──────────────────────────────┘
│
[hallüsinasyon kontrolü] ──► dayanaksız ──► yeniden üret
│ dayanaklı
[cevap alaka kontrolü] ──► yanıtlamıyor ──► web araması + tekrar
│ yanıtlıyor
Cevap


Her karar noktası, yapılandırılmış (Pydantic) çıktı veren küçük bir LLM tabanlı değerlendiricidir — grafiğin yönlendirme yaptığı bir evet/hayır kararı.

### Değerlendiriciler

| Değerlendirici | Yanıtladığı soru | Başarısız olursa |
|---|---|---|
| **Yönlendirici** | Bu soru, indekslenen konularla ilgili mi? | Web aramasına gönder |
| **Belge değerlendirici** | Getirilen her belge alakalı mı? | Ele; biri bile başarısızsa web araması tetikle |
| **Hallüsinasyon değerlendirici** | Cevap belgelere dayanıyor mu? | Yeniden üret |
| **Cevap değerlendirici** | Cevap soruyu gerçekten yanıtlıyor mu? | Web araması yap ve tekrar dene |

### Tasarım kararları

**Cevaptan sonra değil, önce değerlendir.** Alakasız belgeleri üretimden önce elemek, kötü bağlamın en baştan "kendinden emin ama yanlış" bir cevaba dönüşmesini engeller.

**Web araması varsayılan değil, yedek.** Bilgi tabanı bir soruyu kapsamıyorsa, zayıf getirimden cevap üretmeye zorlamak, boşluğu kabul etmekten kötüdür. Grafik, uydurmak yerine web aramasına yönlenir.

**İki aşamalı cevap doğrulama.** Dayanak ve alaka ayrı hatalardır: bir cevap belgelere sadık olup soruyu yanıtlamıyor olabilir, ya da soruyu yanıtlarken kaynakların ötesine kayabilir. Her biri ayrı ayrı kontrol edilir.

**Toplu, kotaya duyarlı embedding.** Ücretsiz katman embedding kotası istek başına değil, gönderilen metin başına tükenir — tek bir büyük yığın onu anında bitirebilir. Ingestion, küçük yığınlar (20 belge) halinde, aralıklı gecikmeler ve 429 hatasında otomatik tekrar ile embed eder, böylece dakika başı limitin altında kalır. Embedding'ler `.chroma`'ya kaydedilir; kaynak sayfalar yalnızca bir kez çekilip embed edilir.

### Teknolojiler

- **LangGraph** — durumlu, dallanan grafik orkestrasyonu
- **LangChain** — RAG bileşenleri
- **Google Gemini** — üretim ve değerlendirme
- **Gemini Embedding** — vektörleştirme
- **ChromaDB** — disk kalıcılığı olan vektör deposu
- **Tavily** — web araması yedeği
- **Pydantic** — yapılandırılmış değerlendirici çıktıları

### Bilgi tabanı

Lilian Weng'in LLM ajanları, prompt mühendisliği ve LLM'lere yönelik saldırılar üzerine üç yazısından indekslenmiştir.

### Kurulum

```bash
git clone https://github.com/alpozgenozgur/corrective-rag-project.git
cd corrective-rag-project

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

`.env` dosyası oluşturun:

GOOGLE_API_KEY=buraya_anahtariniz
TAVILY_API_KEY=buraya_anahtariniz


Çalıştırın:

```bash
python main.py
```

İlk çalıştırmada kaynak sayfalar çekilip embed edilir (embedding kotasına saygı için aralıklı). Sonraki çalıştırmalar kayıtlı vektör deposunu kullanır.

### Sınırlar

- Değerlendirme, gecikme ve token maliyeti ekler — her sorgu tek değil, birkaç LLM çağrısı çalıştırır.
- Getirim kalitesi indekslenen sayfalarla sınırlıdır; web araması boşlukları kapatır ama tam ikame değildir.
- Ücretsiz katman embedding kotası, ilk ingestion'ı tasarım gereği yavaşlatır (kasıtlı aralıklama).

---

**Hüseyin Özgür Alpözgen** — [GitHub](https://github.com/alpozgenozgur)
