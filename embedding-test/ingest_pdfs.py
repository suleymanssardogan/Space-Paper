# Arxiv den uzay bilimiyle ilgili gerçek araştırma kağıtlarını arama ve indirme
from httpx import stream
from fastapi import requests
from urllib3 import response
from accelerate import logging
import os
import requests
import logging
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#Loglama Ayarları
logging.basicConfig(
    level=logging.INFO,
    format ="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class DocumetPipeline:
    def __init__(self,download_dir:str="data"):
        self.download_dir = download_dir
        os.makedirs(self.download_dir,exist_ok=True)

    def download_pdf(self,url:str,filename:str)->str:
        """
        Verilen URL'den PDF dosyasını indirir ve belirtilen filename ile kaydeder.

        Args:
            url: PDF dosyasının URL'si
            filename: Dosyanın kaydedileceği isim (örn:"paper_1.pdf")
        
        Returns:
            Dosyanın tam kayıtlı yolu


        """

        file_path= os.path.join(self.download_dir,filename)
        # Eğer dosya zaten varsa tekrar indirme (idempotency)
        if os.path.exists(file_path):
            logger.info(f"Dosya zaten mevcut, indirme atlanıyor: {file_path}")
            return file_path
        try:
            logger.info(f"İndiriliyor:{url}->{file_path}")
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0"}
            response = requests.get(url,headers=headers,stream=True)

            response.raise_for_status()

            #Chunk'lar halinde indirip dosyaya yazmak
            with open(file_path,"wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Başarılı şekilde indirildi ve kaydedildi: {file_path}")
            return file_path

        except requests.exceptions.RequestException as e:
            logger.error(f"İndirme hatası: {e}")
            raise
        except IOError as e:
            logger.error(f"Dosya yazma hatası: {e}")
            raise

    def extract_text_from_pdf(self, file_path: str) -> list[dict]:
        """PDF dosyasından metin ve sayfa numaralarını çıkarır."""
        logger.info(f"Metin çıkarılıyor: {file_path}")
        extracted_data = []
        
        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            logger.info(f"Toplam sayfa sayısı: {total_pages}")
            
            for page_num in range(total_pages):
                page = reader.pages[page_num]
                text = page.extract_text()
                
                if text and text.strip():
                    extracted_data.append({
                        "page_number": page_num + 1,
                        "text": text,
                        "char_count": len(text)
                    })
            
            logger.info(f"Başarıyla metin çıkarılan sayfa sayısı: {len(extracted_data)}")
            return extracted_data
        except Exception as e:
            logger.error(f"Metin çıkarma sırasında hata oluştu: {e}")
            raise e
    def chunk_text_data(self, pages_data: list[dict]) -> list[dict]:
        """Çıkarılan sayfa metinlerini akıllıca parçalara böler."""
        logger.info("Metinler akıllı parçalara (chunk) ayrılıyor...")
        
        # 800 karakterlik parçalar oluşturup 150 karakterlik kesişim (overlap) veriyoruz
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=len
        )
    
        all_chunks = []
        for page in pages_data:
            page_num = page["page_number"]
            page_text = page["text"]
            
            # Sayfa metnini bölüyoruz
            chunks = text_splitter.split_text(page_text)
            
            for i, chunk_text in enumerate(chunks):
                all_chunks.append({
                    "text": chunk_text,
                    "page_number": page_num,
                    "chunk_index": i + 1,
                    "char_count": len(chunk_text)
                })
                
        logger.info(f"Toplam sayfalardan {len(all_chunks)} adet parça (chunk) üretildi.")
        return all_chunks

if __name__ == "__main__":
    # Test makaleleri (arXiv uzay bilimi ve teleskop makaleleri)
    papers = {
        "jwst_performance.pdf": "https://arxiv.org/pdf/2207.13067", # JWST Bilimsel Performansı
        "kepler_mission.pdf": "https://arxiv.org/pdf/1001.0352"     # Kepler Gezegen Bulma Görevi
    }
    
    pipeline = DocumetPipeline(download_dir="data")
    
    for filename, url in papers.items():
        print(f"\n--- {filename} İşleme Başlanıyor ---")
        try:
            # 1. PDF İndir
            pdf_path = pipeline.download_pdf(url, filename)
            
            # 2. Metin Çıkar
            pages_data = pipeline.extract_text_from_pdf(pdf_path)
            
            # 3. Metinleri Parçala (Day 5 - Yeni eklenen kısım)
            chunks_data = pipeline.chunk_text_data(pages_data)
            print(f"-> Üretilen Toplam Chunk Sayısı: {len(chunks_data)}")
            
            # İlk 2 chunk'ı ekrana yazdırarak test edelim
            for i, chunk in enumerate(chunks_data[:2]):
                print(f"\n   [Chunk {chunk['chunk_index']} - Sayfa {chunk['page_number']} - Karakter: {chunk['char_count']}]:")
                print(f"   {chunk['text'][:200]}...\n")
                
        except Exception as e:
            print(f"Hata: {filename} işlenemedi: {e}")


    

                
