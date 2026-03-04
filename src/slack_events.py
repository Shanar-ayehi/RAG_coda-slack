from src.rag_engine import ask_bot

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
        user_query = event["text"] 
        
        try:
            # Interroga il nostro RAG Engine collegato a Pinecone
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