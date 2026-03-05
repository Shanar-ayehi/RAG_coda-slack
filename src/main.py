import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, BackgroundTasks
from slack_bolt import App
from slack_bolt.adapter.fastapi import SlackRequestHandler

from src.slack_events import register_events
from src.rag_engine import load_knowledge_base

# 1. Carica le variabili d'ambiente
load_dotenv()

# 2. Inizializza l'app Slack
slack_app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)
slack_handler = SlackRequestHandler(slack_app)

# 3. Collega gli eventi al Bot
register_events(slack_app)

# 4. Inizializza FastAPI
api_app = FastAPI()

# --- ENDPOINT 1: SLACK ---
@api_app.post("/slack/events")
async def slack_endpoint(req: Request):
    return await slack_handler.handle(req)

# --- ENDPOINT 2: CODA WEBHOOK ---
@api_app.post("/coda-update")
async def coda_webhook(background_tasks: BackgroundTasks):
    """
    Rotta chiamata dall'automazione di Coda.
    Avvia lo scaricamento massivo e l'aggiornamento su Pinecone in background.
    """
    print("🔔 Ricevuto segnale di aggiornamento da Coda!")
    
    doc_id = os.environ.get("CODA_DOC_ID")
    if not doc_id:
        return {"status": "error", "message": "CODA_DOC_ID mancante nel file .env"}
    
    # Esegue la funzione in background per non far andare Coda in timeout
    background_tasks.add_task(load_knowledge_base, doc_id)
    
    return {"status": "ok", "message": "Aggiornamento Knowledge Base avviato in background!"}