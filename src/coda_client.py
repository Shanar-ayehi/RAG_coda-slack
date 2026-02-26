import os
import time
import requests
from dotenv import load_dotenv

# Carica il token di Coda dal file .env
load_dotenv()
CODA_API_TOKEN = os.environ.get("CODA_API_TOKEN")

# Headers per l'autenticazione
HEADERS = {
    "Authorization": f"Bearer {CODA_API_TOKEN}",
    "Content-Type": "application/json"
}

def export_page_to_markdown(doc_id: str, page_id: str) -> str:
    """
    Gestisce l'intero processo di esportazione di una pagina Coda in Markdown.
    Ritorna il testo puro in Markdown.
    """
    print(f"🔄 Inizio esportazione per la pagina {page_id}...")
    
    # STEP 1: Richiedi l'esportazione
    export_url = f"https://coda.io/apis/v1/docs/{doc_id}/pages/{page_id}/export"
    payload = {"outputFormat": "markdown"}
    
    response = requests.post(export_url, headers=HEADERS, json=payload)
    response.raise_for_status() # Lancia un errore se il token è sbagliato o la pagina non esiste
    
    request_id = response.json().get("id")
    print(f"⏳ Richiesta presa in carico da Coda (ID: {request_id}). Attendo il file...")

    # STEP 2: Polling (Controlla lo stato ogni 3 secondi finché non è pronto)
    status_url = f"{export_url}/{request_id}"
    
    while True:
        status_response = requests.get(status_url, headers=HEADERS)
        status_response.raise_for_status()
        data = status_response.json()
        
        status = data.get("status")
        
        if status == "complete":
            download_link = data.get("downloadLink")
            print("✅ Esportazione completata! Scarico il file...")
            break
        elif status == "failed":
            raise Exception("L'esportazione su Coda è fallita.")
            
        # Se lo status è "inProgress", aspetta 3 secondi e riprova
        time.sleep(3)

    # STEP 3: Scarica il contenuto effettivo del file Markdown
    markdown_response = requests.get(download_link)
    markdown_response.raise_for_status()
    
    return markdown_response.text

# --- AREA DI TEST ---
# Se esegui questo file direttamente, farà un test di download.
if __name__ == "__main__":
    # Sostituisci questi con ID reali per fare una prova
    TEST_DOC_ID = "inserisci_qui_il_doc_id"
    TEST_PAGE_ID = "inserisci_qui_il_page_id"
    
    if CODA_API_TOKEN:
        try:
            testo_estratto = export_page_to_markdown(TEST_DOC_ID, TEST_PAGE_ID)
            print("\n--- ANTEPRIMA DEL MARKDOWN ESTRATTO ---")
            print(testo_estratto[:500]) # Stampa solo i primi 500 caratteri
        except Exception as e:
            print(f"❌ Errore durante il test: {e}")
    else:
        print("⚠️ CODA_API_TOKEN mancante nel file .env!")