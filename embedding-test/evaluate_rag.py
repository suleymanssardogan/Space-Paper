import requests
import json
import os
import sys
import time

# API URL
API_URL = "http://127.0.0.1:8000/api/v1/ask"

# Benchmark soruları (Uzay bilimleri ve teleskoplar hakkında)
EVAL_QUESTIONS = [
    {
        "question": "What is the science performance of the James Webb Space Telescope?",
        "expected_source": "jwst_performance.pdf"
    },
    {
        "question": "What are the objectives and key findings of the Kepler Mission?",
        "expected_source": "kepler_mission.pdf"
    },
    {
        "question": "What instruments does the James Webb Space Telescope carry?",
        "expected_source": "jwst_performance.pdf"
    },
    {
        "question": "How to bake a pizza at home?", # Alakasız soru (Refusal / Grounding testi)
        "expected_source": None
    }
]

def run_evaluation():
    print("=" * 60)
    print("🚀 ANTISPACE RAG PIPELINE OFFLINE EVALUATION START")
    print("=" * 60)
    print(f"Hedef API: {API_URL}\n")
    
    results = []
    
    for i, item in enumerate(EVAL_QUESTIONS):
        q = item["question"]
        expected = item["expected_source"]
        
        print(f"[{i+1}/{len(EVAL_QUESTIONS)}] Değerlendiriliyor: '{q}'")
        
        payload = {
            "question": q,
            "limit": 3,
            "score_threshold": 0.30
        }
        
        try:
            start_time = time.time()
            response = requests.post(API_URL, json=payload, timeout=30)
            latency = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "")
                citations = data.get("citations", [])
                faithfulness = data.get("faithfulness")
                relevance = data.get("answer_relevance")
                
                citation_files = [c["source"] for c in citations]
                
                results.append({
                    "question": q,
                    "answer_snippet": answer[:100] + "..." if len(answer) > 100 else answer,
                    "citations": ", ".join(citation_files) if citation_files else "None",
                    "faithfulness": faithfulness,
                    "relevance": relevance,
                    "latency": round(latency, 2),
                    "status": "Success"
                })
                
                print(f"   -> Durum: Başarılı | Gecikme: {latency:.2f}sn")
                print(f"   -> Faithfulness (Sadakat): {faithfulness}")
                print(f"   -> Relevance (Soru Alakası): {relevance}\n")
            else:
                print(f"   -> HATA: API HTTP {response.status_code} döndürdü. Yanıt: {response.text}\n")
                results.append({
                    "question": q,
                    "answer_snippet": "N/A (Error)",
                    "citations": "N/A",
                    "faithfulness": None,
                    "relevance": None,
                    "latency": round(latency, 2),
                    "status": f"Error {response.status_code}"
                })
        except Exception as e:
            print(f"   -> İstek Hatası: {e}\n")
            results.append({
                "question": q,
                "answer_snippet": "N/A (Exception)",
                "citations": "N/A",
                "faithfulness": None,
                "relevance": None,
                "latency": 0.0,
                "status": f"Exception"
            })
            
    # Markdown Raporu Yazdırma
    print("\n" + "=" * 60)
    print("📊 EVALUATION REPORT SUMMARY")
    print("=" * 60)
    print("| Question | Citations | Answer Snippet | Faithfulness | Relevance | Latency (s) | Status |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    
    total_faithfulness = 0.0
    valid_f_count = 0
    total_relevance = 0.0
    valid_r_count = 0
    
    for res in results:
        f_str = f"{res['faithfulness']:.2f}" if res['faithfulness'] is not None else "N/A"
        r_str = f"{res['relevance']:.2f}" if res['relevance'] is not None else "N/A"
        
        if res['faithfulness'] is not None:
            total_faithfulness += res['faithfulness']
            valid_f_count += 1
        if res['relevance'] is not None:
            total_relevance += res['relevance']
            valid_r_count += 1
            
        print(f"| {res['question']} | {res['citations']} | {res['answer_snippet']} | {f_str} | {r_str} | {res['latency']} | {res['status']} |")
        
    print("-" * 60)
    avg_f = total_faithfulness / valid_f_count if valid_f_count > 0 else 0.0
    avg_r = total_relevance / valid_r_count if valid_r_count > 0 else 0.0
    
    print(f"📈 Ortalama Faithfulness (Sadakat) Skoru: {avg_f:.2f}")
    print(f"📈 Ortalama Soru Alakası (Relevance) Skoru: {avg_r:.2f}")
    print("=" * 60)

if __name__ == "__main__":
    run_evaluation()
