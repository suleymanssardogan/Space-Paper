import requests
import json

url = "http://127.0.0.1:8000/api/v1/ask"

questions = [
    "What is Stephan's Quintet?",
    "What is the Kepler Follow-up Observation Program?",
    "How to bake a pizza?"  # Grounding test: should be refused
]

for q in questions:
    print(f"\n=== SORGU: '{q}' ===")
    payload = {
        "question": q,
        "limit": 3,
        "score_threshold": 0.35
    }
    try:
        response = requests.post(url, json=payload, timeout=20)
        print("Status Code:", response.status_code)
        if response.status_code == 200:
            res_data = response.json()
            print("Cevap:")
            print(res_data.get("answer", ""))
            print("\nAtıflar (Citations):")
            for cit in res_data.get("citations", []):
                print(f"- [Skor: {cit['score']:.4f}] Kaynak: {cit['source']}, Sayfa: {cit['page_number']}")
        else:
            print("Hata Yanıtı:", response.text)
    except Exception as e:
        print("İstek hatası:", e)
