# RAG Coda-Slack Bot

Un bot per Slack basato su architettura RAG (Retrieval-Augmented Generation) che permette agli utenti di interrogare la Knowledge Base aziendale ospitata su Coda.io direttamente tramite chat.

## Funzionalità Principali

* **Interrogazione in Linguaggio Naturale:** Gli utenti possono fare domande su Slack menzionando il bot (`@Bot come chiedo le ferie?`).
* **Workaround Timeout Slack:** Utilizza i *Lazy Listeners* di `slack-bolt` per bypassare il limite dei 3 secondi di Slack, garantendo il tempo necessario all'AI per elaborare la risposta.
* **Motore RAG Serverless:** Utilizza LangChain e **Pinecone** per la ricerca vettoriale sui documenti direttamente in cloud, garantendo persistenza dei dati anche se il server si riavvia.
* **LLM ad alte prestazioni:** Alimentato dai modelli **Cohere** (`command-r-plus-08-2024` per la generazione e `embed-multilingual-v3.0` per gli embeddings).
* **Estrazione Dati Coda:** Modulo dedicato per il polling asincrono e il download delle pagine Coda in formato Markdown.

---

## Stack Tecnologico

* **Linguaggio:** Python 3.12+
* **Gestione Dipendenze:** Poetry
* **Server Web:** FastAPI / Uvicorn
* **Integrazione Slack:** Slack Bolt Framework (`slack-bolt`)
* **AI & Orchestrazione:** LangChain, LangChain-Cohere, LangChain-Pinecone, LangChain-Classic
* **Database Vettoriale:** Pinecone (Cloud)

---

## Struttura del Progetto

```text
coda-slack-rag/
├── .env                 # Variabili d'ambiente sensibili (NON tracciato da Git)
├── .gitignore           # File ignorati da Git 
├── pyproject.toml       # Configurazione del progetto e librerie (Poetry)
├── poetry.lock          # Versioni esatte delle dipendenze bloccate
├── README.md            # Questo file di documentazione
├── mock_coda.md         # File Markdown di test per simulare l'export da Coda
│
└── src/                 # Sorgenti del codice
    ├── __init__.py      # Inizializzatore del modulo Python
    ├── main.py          # Entry point FastAPI e smistamento endpoint Slack
    ├── slack_events.py  # Gestione logica eventi Slack (mentions, lazy listeners)
    ├── rag_engine.py    # Logica AI: chunking, embedding, Pinecone DB e chain LangChain
    └── coda_client.py   # Logica per l'esportazione asincrona del Markdown da Coda API
```

## Setup e Installazione

### 1. Prerequisiti

* Python 3.12 installato sul sistema.

* Poetry installato (pip install poetry).

* Un account su Cohere per l'API Key.

* Un account su Pinecone (creare un Indice chiamato coda-rag-index con 1024 dimensions e metrica cosine).

* Un'App creata sulla Slack API Dashboard.

### 2. Installazione dipendenze

Clona la repository e installa le dipendenze tramite Poetry:

```Bash
poetry env use 3.12
poetry install
```

### 3. Variabili d'Ambiente

Crea un file .env nella root del progetto e inserisci le seguenti chiavi:

```Bash
SLACK_BOT_TOKEN=xoxb-tuo-token-slack-qui
SLACK_SIGNING_SECRET=tuo-signing-secret-slack-qui
COHERE_API_KEY=tua-api-key-cohere-qui
CODA_API_TOKEN=tuo-token-coda-qui
PINECONE_API_KEY=tua-api-key-pinecone-qui
PINECONE_INDEX_NAME=coda-rag-index
```

### Come avviare il Bot in locale

Durante lo sviluppo, il bot richiede due terminali aperti: uno per il server FastAPI e uno per esporre la porta locale su internet tramite Ngrok.

#### Terminale 1: Avvio del Server FastAPI

```Bash
poetry run uvicorn src.main:api_app --reload --port 3000
Terminale 2: Avvio del Tunnel (Ngrok)
```

#### Terminale 2: Avvio del Tunnel (Ngrok)

```Bash
ngrok http 3000
```

Copia l'URL generato da ngrok (es. `https://xxxx.ngrok-free.app`) e incollalo nella dashboard della tua App Slack sotto Event Subscriptions > Request URL, aggiungendo alla fine /slack/events.

### ☁️ Architettura di Produzione (Deploy)

Il bot è stato strutturato per essere "Cloud-Native". Non avendo database locali, è ideale per essere ospitato su piattaforme PaaS (Platform as a Service) gratuite.

* **Hosting Consigliato:** Render.com o Koyeb.

* **Database:** Pinecone garantisce che la Knowledge Base non venga persa durante i riavvii effimeri dei container cloud.

* **Keep-Awake:** Per evitare che il piano gratuito vada in "sleep" (causando un timeout di Slack al risveglio), è consigliato l'uso di un servizio come cron-job.org per effettuare un ping all'indirizzo del bot ogni 10 minuti.