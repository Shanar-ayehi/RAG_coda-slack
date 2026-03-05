import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()
CODA_API_TOKEN = os.environ.get("CODA_API_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {CODA_API_TOKEN}",
    "Content-Type": "application/json"
}

def export_page_to_markdown(doc_id: str, page_id: str) -> str:
    """Esporta una singola pagina Coda in formato Markdown."""
    print(f"🔄 Inizio esportazione per la pagina {page_id}...")
    
    export_url = f"https://coda.io/apis/v1/docs/{doc_id}/pages/{page_id}/export"
    payload = {"outputFormat": "markdown"}
    
    response = requests.post(export_url, headers=HEADERS, json=payload)
    response.raise_for_status() 
    
    request_id = response.json().get("id")
    status_url = f"{export_url}/{request_id}"
    
    while True:
        status_response = requests.get(status_url, headers=HEADERS)
        status_response.raise_for_status()
        data = status_response.json()
        status = data.get("status")
        
        if status == "complete":
            download_link = data.get("downloadLink")
            break
        elif status == "failed":
            raise Exception(f"Esportazione fallita per la pagina {page_id}.")
            
        time.sleep(3)

    markdown_response = requests.get(download_link)
    markdown_response.raise_for_status()
    return markdown_response.text

def get_all_pages_in_doc(doc_id: str) -> list:
    """Restituisce una lista con gli ID di tutte le pagine di un documento."""
    print(f"🔍 Cerco tutte le pagine nel documento {doc_id}...")
    url = f"https://coda.io/apis/v1/docs/{doc_id}/pages"
    
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    
    pages = response.json().get("items", [])
    page_ids = [page["id"] for page in pages]
    
    print(f"📚 Trovate {len(page_ids)} pagine da scaricare!")
    return page_ids