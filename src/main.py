from src.rag_engine import ask_bot, load_knowledge_base
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from slack_bolt import App
from slack_bolt.adapter.fastapi import SlackRequestHandler
from langchain_cohere import ChatCohere, CohereEmbeddings

# 1. Carica le variabili d'ambiente dal file .env
load_dotenv()

# 2. Inizializza l'app Slack
slack_app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)
slack_handler = SlackRequestHandler(slack_app)

# 3. Inizializza FastAPI (Il server web che ascolterà Slack)
api_app = FastAPI()

# 4. Inizializza il modello AI (Cohere)
llm = ChatCohere(
    model="command-r-plus-08-2024",
    cohere_api_key=os.environ.get("COHERE_API_KEY")
)


# --- IL WORKAROUND DEI 3 SECONDI (LAZY LISTENERS) ---

# Funzione A: Viene eseguita istantaneamente
def ack_mention(ack, say):
    # 'ack()' dice subito a Slack: "Ricevuto! Tutto ok (HTTP 200)"
    ack() 
    # Manda un messaggio temporaneo all'utente per fargli capire che stiamo lavorando
    say("Sto consultando la Knowledge Base di Coda... ⏳") 

# Funzione B: Viene eseguita in background (può durare anche 10-20 secondi)
# Funzione B: Viene eseguita in background
def process_rag_query(event, say):
    user_query = event["text"] 
    
    try:
        # Interroga il nostro RAG Engine
        ai_response = ask_bot(user_query)
        
        # Invia la risposta finale nel canale Slack
        say(ai_response)
    except Exception as e:
        say(f"Scusa, ho riscontrato un errore nel leggere la Knowledge Base: {e}")

# Diciamo a Slack di ascoltare quando il bot viene menzionato (@Bot)
slack_app.event("app_mention")(
    ack=ack_mention,        # Risponde subito in < 3 secondi
    lazy=[process_rag_query] # Lavora in background con l'AI
)
# --- FINE WORKAROUND ---


# 5. L'endpoint ("la porta") a cui Slack busserà per inviare i messaggi
@api_app.post("/slack/events")
async def endpoint(req: Request):
    return await slack_handler.handle(req)