import logging
from src.rag_engine import ask_bot

logger = logging.getLogger(__name__)

# Deduplicazione eventi: traccia eventi già processati
_processed_events: set = set()

def register_events(slack_app):
    """
    Registra tutti gli ascoltatori di eventi (mentions, messaggi, ecc.) 
    all'applicazione Slack principale.
    """

    # --- IL WORKAROUND DEI 3 SECONDI (LAZY LISTENERS) ---
    
    # Funzione A: Viene eseguita istantaneamente
    def ack_mention(ack, say):
        # 'ack()' dice subito a Slack: "Ricevuto! Tutto ok"
        ack() 
        say("Sto consultando la Knowledge Base su Pinecone... ☁️⏳") 

    # Funzione B: Viene eseguita in background
    def process_rag_query(event, say):
        global _processed_events
        
        # Deduplicazione: evita processing multipli dello stesso evento
        event_id = event.get("event_id") or f"{event.get('ts')}_{event.get('channel')}"
        if event_id in _processed_events:
            logger.info(f"⏭️ Evento già processato: {event_id}")
            return
        _processed_events.add(event_id)
        
        # Pulizia periodica del set (mantieni ultimi 1000 eventi)
        if len(_processed_events) > 1000:
            _processed_events.clear()
        
        user_query = event["text"] 
        thread_ts = event.get("thread_ts", event.get("ts"))  # Use thread_ts if exists, otherwise use ts
        channel = event["channel"]
        
        try:
            # Interroga il nostro RAG Engine collegato a Pinecone con contesto della conversazione
            ai_response = ask_bot(user_query, thread_ts=thread_ts, channel=channel)
            # Invia la risposta finale nel canale Slack
            say(ai_response)
        except Exception as e:
            say(f"Scusa, ho riscontrato un errore nel leggere la Knowledge Base: {e}")

    # Diciamo a Slack di ascoltare quando il bot viene menzionato (@Bot)
    slack_app.event("app_mention")(
        ack=ack_mention,        # Risponde subito in < 3 secondi
        lazy=[process_rag_query] # Lavora in background con l'AI
    )
    # 2. NUOVO: Ascolta i messaggi privati (Direct Messages)
    slack_app.event("message")(
        ack=ack_mention,
        lazy=[process_rag_query]
    )
