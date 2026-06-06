from sentence_transformers import SimilarityFunction,SentenceTransformer

# 2 model yükleyeceğiz

model_1 = SentenceTransformer("all-MiniLM-L6-v2")
model_2 = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Test Uzay cümleleri
sentences = [
    "The James Webb Space Telescope captured deep field images of early galaxies.",
    "Webb telescope observed ancient galaxies in the deep universe.",
    "A pizza recipe requires flour, water, yeast, and tomato sauce.",
] 


print("--- MODEL 1: all-MiniLM-L6-v2 ---")
vectors_1 = model_1.encode(sentences)
print(f"Cümle 1 Vektör Boyutu (Dimension): {len(vectors_1[0])}")
# İlk cümlenin bilgisayarın anladığı dildeki ilk 5 sayısını görelim(embedding yani)
print(f"Vektörden bir kesit: {vectors_1[0][:5]}\n")

print("--- MODEL 2: bge-small-en-v1.5 ---")
vectors_2 = model_2.encode(sentences)
print(f"Cümle 1 Vektör Boyutu (Dimension): {len(vectors_2[0])}")
print(f"Vektörden bir kesit: {vectors_2[0][:5]}\n")

# Semantik Benzerlik Testi (İlk model için)
similarity = model_1.similarity(vectors_1, vectors_1)
print("--- CÜMLELER ARASI BENZERLİK SKORLARI ---")
print(
    f"1. ve 2. Uzay Cümlesi Benzerliği: {similarity[0][1].item():.4f}"
)  # Yüksek çıkmalı
print(
    f"1. Uzay Cümlesi ile Pizza Cümlesi Benzerliği: {similarity[0][2].item():.4f}"
)  # Düşük çıkmalı