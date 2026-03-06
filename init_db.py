import os
from dotenv import load_dotenv
from src.rag_engine import load_knowledge_base

# Carica le variabili (incluso il CODA_DOC_ID)
load_dotenv()
doc_id = os.environ.get("CODA_DOC_ID")

if not doc_id:
    print("❌ Errore: CODA_DOC_ID non trovato nel file .env")
else:
    print("🚀 Avvio il caricamento INIZIALE della Knowledge Base da Coda...")
    # Lancia la funzione massiva!
    load_knowledge_base(doc_id)
    print("🎉 Caricamento completato! Ora il Bot è intelligente.")