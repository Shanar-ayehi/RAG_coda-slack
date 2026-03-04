import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from slack_bolt import App
from slack_bolt.adapter.fastapi import SlackRequestHandler

# Importiamo la funzione che contiene le regole dal nuovo file
from src.slack_events import register_events

# 1. Carica le variabili d'ambiente dal file .env
load_dotenv()

# 2. Inizializza l'app Slack
slack_app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)
slack_handler = SlackRequestHandler(slack_app)

# 3. Collega gli eventi al Bot! (Qui chiamiamo il file slack_events.py)
register_events(slack_app)

# 4. Inizializza FastAPI (Il server web che ascolterà Slack)
api_app = FastAPI()

# 5. L'endpoint ("la porta") a cui Slack busserà per inviare i messaggi
@api_app.post("/slack/events")
async def endpoint(req: Request):
    return await slack_handler.handle(req)