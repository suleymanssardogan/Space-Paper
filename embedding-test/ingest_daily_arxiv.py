import os
import requests
import logging
import xml.etree.ElementTree as ET
from ingest_pdfs import DocumetPipeline
from save_to_qdrant import SpaceScienceVectorStore
from ingest_to_qdrant import bulk_upsert_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ingest_daily_arxiv")

def fetch_recent_arxiv_papers(category: str = "astro-ph.CO", max_results: int = 3) -> list[dict]:
    """
    Queries the official arXiv API for the most recent papers in a specific category.
    """
    logger.info(f"arXiv API sorgulanıyor. Kategori: {category} | Maks Sonuç: {max_results}")
    
    # URL for query
    api_url = f"http://export.arxiv.org/api/query?search_query=cat:{category}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    
    try:
        response = requests.get(api_url, timeout=15)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"arXiv API bağlantı hatası: {e}")
        return []

    # Parse XML response
    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as pe:
        logger.error(f"XML ayrıştırma hatası: {pe}")
        return []

    namespaces = {'atom': 'http://www.w3.org/2005/Atom'}
    entries = root.findall('atom:entry', namespaces)
    
    papers = []
    for entry in entries:
        title = entry.find('atom:title', namespaces).text.strip().replace('\n', ' ')
        raw_id = entry.find('atom:id', namespaces).text
        
        # Clean arXiv ID (e.g. from http://arxiv.org/abs/2405.0001v1 to 2405.0001)
        arxiv_id = raw_id.split('/abs/')[-1].split('v')[0]
        filename = f"arxiv_{arxiv_id}.pdf"
        
        # Resolve PDF download link
        pdf_url = None
        for link in entry.findall('atom:link', namespaces):
            rel = link.attrib.get('rel')
            link_type = link.attrib.get('type')
            title_attr = link.attrib.get('title')
            
            if title_attr == 'pdf' or link_type == 'application/pdf':
                pdf_url = link.attrib.get('href')
                break
                
        # Fallback PDF url constructor if link wasn't parsed correctly
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
            
        if pdf_url:
            papers.append({
                "title": title,
                "filename": filename,
                "url": pdf_url,
                "id": arxiv_id
            })
            
    logger.info(f"arXiv API'den {len(papers)} adet makale başarıyla listelendi.")
    return papers

def run_daily_ingestion(category: str = "astro-ph.CO", max_results: int = 3):
    """
    Downloads, chunks, and indexes recent arXiv papers into Qdrant.
    """
    logger.info("--- Günlük Otomatik Ingestion Başlatıldı ---")
    
    papers_to_ingest = fetch_recent_arxiv_papers(category, max_results)
    if not papers_to_ingest:
        logger.warning("İndirilecek yeni makale bulunamadı.")
        return 0
        
    pipeline = DocumetPipeline(download_dir="data")
    store = SpaceScienceVectorStore()
    collection = "space_science_collection"
    
    # Ensure collection exists
    store.init_collection(collection)
    
    all_chunks = []
    success_count = 0
    
    for paper in papers_to_ingest:
        filename = paper["filename"]
        url = paper["url"]
        title = paper["title"]
        
        logger.info(f"İşleniyor: {title} ({filename})")
        
        try:
            # 1. Download
            pdf_path = pipeline.download_pdf(url, filename)
            
            # 2. Extract
            pages_data = pipeline.extract_text_from_pdf(pdf_path)
            
            # 3. Chunk
            chunks_data = pipeline.chunk_text_data(pages_data)
            
            # Enrich metadata
            for chunk in chunks_data:
                chunk["source"] = filename
                chunk["paper_title"] = title
                all_chunks.append(chunk)
                
            success_count += 1
            
        except Exception as e:
            logger.error(f"{filename} işlenirken hata oluştu, atlanıyor: {e}")
            
    # 4. Bulk Upsert into Qdrant
    if all_chunks:
        logger.info(f"Toplam {len(all_chunks)} adet chunk Qdrant'a yükleniyor...")
        bulk_upsert_chunks(store, collection, all_chunks, batch_size=32)
        logger.info(f"Ingestion tamamlandı. {success_count}/{len(papers_to_ingest)} makale başarıyla indekslendi.")
    else:
        logger.warning("İndirilen makalelerden geçerli metin parçası (chunk) çıkarılamadı.")
        
    return success_count

if __name__ == "__main__":
    # Query latest 3 papers from cosmology (astro-ph.CO) and planetary astrophysics (astro-ph.EP)
    run_daily_ingestion(category="astro-ph.CO+OR+cat:astro-ph.EP", max_results=3)
