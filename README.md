# 🤖 RAG Coda-Slack Bot

Un bot per Slack basato su architettura RAG (Retrieval-Augmented Generation) che permette agli utenti di interrogare la Knowledge Base aziendale ospitata su Coda.io direttamente tramite chat.

## 🚀 Funzionalità Principali

* **Interrogazione in Linguaggio Naturale:** Gli utenti possono fare domande su Slack menzionando il bot (`@Bot come chiedo le ferie?`).
* **Workaround Timeout Slack:** Utilizza i *Lazy Listeners* di `slack-bolt` per bypassare il limite dei 3 secondi di Slack, garantendo il tempo necessario all'AI per elaborare la risposta.
* **Motore RAG Avanzato:** Utilizza LangChain e ChromaDB per la ricerca vettoriale sui documenti.
* **LLM ad alte prestazioni:** Alimentato dai modelli **Cohere** (`command-r-plus-08-2024` per la generazione e `embed-multilingual-v3.0` per gli embeddings).
* **Estrazione Dati Coda:** Modulo dedicato per il polling asincrono e il download delle pagine Coda in formato Markdown.

---

## 🛠️ Stack Tecnologico

* **Linguaggio:** Python 3.12+
* **Gestione Dipendenze:** Poetry
* **Server Web:** FastAPI / Uvicorn
* **Integrazione Slack:** Slack Bolt Framework (`slack-bolt`)
* **AI & Orchestrazione:** LangChain, LangChain-Cohere, LangChain-Chroma, LangChain-Classic
* **Database Vettoriale:** ChromaDB (Locale)

---

## 📂 Struttura del Progetto

```text
coda-slack-rag/
├── .env                 # Variabili d'ambiente sensibili (NON tracciato da Git)
├── .gitignore           # File ignorati da Git (inclusi i database locali)
├── pyproject.toml       # Configurazione del progetto e librerie (Poetry)
├── poetry.lock          # Versioni esatte delle dipendenze bloccate
├── README.md            # Questo file di documentazione
├── mock_coda.md         # File Markdown di test per simulare l'export da Coda
│
└── src/                 # Sorgenti del codice
    ├── __init__.py      # Inizializzatore del modulo Python
    ├── main.py          # Entry point FastAPI, configurazione Slack e endpoint
    ├── rag_engine.py    # Logica AI: chunking, embedding, ChromaDB e chain LangChain
    └── coda_client.py   # Logica per l'esportazione asincrona del Markdown da Coda API
```

## ⚙️ Setup e Installazione

### 1. Prerequisiti
* Python 3.12 installato sul sistema.

* Poetry installato (pip install poetry).

* Un account su Cohere per l'API Key.

* Un'App creata su Slack API.

### 2. Installazione dipendenze

Clona la repository e installa le dipendenze tramite Poetry:

```bash
poetry env use 3.12
poetry install
```

### 3. Variabili d'Ambiente

Crea un file .env nella root del progetto e inserisci le seguenti chiavi:

```bash
SLACK_BOT_TOKEN=xoxb-tuo-token-slack-qui
SLACK_SIGNING_SECRET=tuo-signing-secret-slack-qui
COHERE_API_KEY=tua-api-key-cohere-qui
CODA_API_TOKEN=tuo-token-coda-qui
```

### 🏃‍♂️ Come avviare il Bot in locale

Durante lo sviluppo, il bot richiede due terminali aperti: uno per il server FastAPI e uno per esporre la porta locale su internet tramite Ngrok.

#### **Terminale 1: Avvio del Server FastAPI**

```Bash
poetry run uvicorn src.main:api_app --reload --port 3000
```

#### **Terminale 2: Avvio del Tunnel (Ngrok)**

```Bash
ngrok http 3000
```

Copia l'URL generato da ngrok (es. `https://xxxx.ngrok-free.app`) e incollalo nella dashboard della tua App Slack sotto Event Subscriptions > Request URL, aggiungendo alla fine /slack/events.
