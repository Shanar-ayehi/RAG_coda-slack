# RAG Coda-Slack Bot

Un bot per Slack basato su architettura RAG (Retrieval-Augmented Generation) che permette agli utenti di interrogare la Knowledge Base aziendale ospitata su Coda.io direttamente tramite chat.

## Funzionalità Principali

* **Interrogazione in Linguaggio Naturale:** Gli utenti possono fare domande su Slack menzionando il bot (`@Bot come chiedo le ferie?`).
* **Workaround Timeout Slack:** Utilizza i *Lazy Listeners* di `slack-bolt` per bypassare il limite dei 3 secondi di Slack, garantendo il tempo necessario all'AI per elaborare la risposta.
* **Motore RAG Serverless:** Utilizza LangChain e **Pinecone** per la ricerca vettoriale sui documenti direttamente in cloud, garantendo persistenza dei dati anche se il server si riavvia. Inoltre il motore è hostato tramite **Render**, una solusione PaaS.
* **LLM ad alte prestazioni:** Alimentato dai modelli **Cohere** (`command-r-plus-08-2024` per la generazione e `embed-multilingual-v3.0` per gli embeddings).
* **Estrazione Dati Coda:** Modulo dedicato per il polling asincrono e il download delle pagine Coda in formato Markdown.

---

## Stack Tecnologico

* **Linguaggio:** Python 3.12+
* **Gestione Dipendenze:** Poetry
* **Server Web:** FastAPI / Uvicorn
* **Hosting via PaaS:** Render
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

### Come Collegare Slack (Temporaneo con Ngrok)

#### Fase 1: Creare l'App (Il "Corpo" del Bot)

Vai sul sito ufficiale per sviluppatori: api.slack.com/apps e fai il login con il tuo account Slack.

Clicca sul bottone verde "Create New App".

Scegli "From scratch" (Da zero).

Dai un nome al tuo bot (es. Knowledge Base Bot o Coda RAG).

Passaggio fondamentale: Nel menu a tendina "Pick a workspace", seleziona il tuo Workspace lavorativo.

Clicca su "Create App".

#### Fase 2: Dare i permessi al Bot (Gli "Scopes")

Il bot appena nato è sordo e muto. Dobbiamo dirgli esplicitamente cosa può fare.

Nel menu a sinistra, clicca su "OAuth & Permissions".

Scorri in basso fino alla sezione "Scopes" e guarda sotto "Bot Token Scopes".

Clicca su "Add an OAuth Scope" e aggiungi questi tre permessi fondamentali:

app_mentions:read (Permette al bot di sentire quando qualcuno scrive @Bot)

chat:write (Permette al bot di scrivere e rispondere nel canale)

channels:read (Permette al bot di vedere in quali canali pubblici si trova)

#### Fase 3: Installarlo nel Workspace (Ottenere i Token)

Ora che ha i permessi, possiamo "assumerlo" in azienda.

Scorri in cima alla stessa pagina ("OAuth & Permissions") e clicca sul bottone "Install to Workspace".

Slack ti chiederà di confermare ("L'app richiede il permesso di accedere..."). Clicca su "Consenti" (Allow).

🎉 Ecco il tuo primo Token! Vedrai apparire una stringa che inizia con xoxb-... (Bot User OAuth Token). Copiala e incollala nel tuo file .env alla riga SLACK_BOT_TOKEN.

Per prendere il secondo segreto, vai nel menu a sinistra su "Basic Information", scorri fino a "App Credentials" e clicca su Show accanto a "Signing Secret". Copialo e incollalo nel tuo file .env alla riga SLACK_SIGNING_SECRET.

#### Fase 4: Collegare le orecchie (Event Subscriptions)

Ora dobbiamo dire a Slack di mandare i messaggi al tuo computer tramite Ngrok.

Assicurati che il tuo server FastAPI e ngrok siano accesi sul tuo PC.

Vai nel menu a sinistra su "Event Subscriptions".

Accendi l'interruttore "Enable Events" su On.

Nel campo "Request URL", incolla l'URL di ngrok e aggiungi /slack/events (es. `https://1234-abcd.ngrok-free.app/slack/events`). Aspetta un secondo e dovrebbe apparire la scritta verde "Verified".

Scorri in basso fino a "Subscribe to bot events", clicca su "Add Bot User Event" e aggiungi:

app_mention

Clicca sul bottone verde "Save Changes" in basso a destra. (Slack ti chiederà di cliccare su un banner giallo in alto "reinstall your app" per applicare le modifiche. Fallo).

Il tocco finale (Su Slack)
Ora il bot fa parte dell'azienda, ma non è ancora nelle stanze a chiacchierare!

Apri la tua applicazione di Slack (o il sito web).

Vai nel canale dove vuoi fare i test (es. #general o creane uno nuovo chiamato #test-bot).

Scrivi questo messaggio nel canale:
/invite @NomeDelTuoBot (sostituendo il nome con quello che gli hai dato).

Slack aggiungerà il bot al canale.

Ora scrivi: @NomeDelTuoBot Qual è il budget formativo per un dipendente part-time?

### Come Collegare Ngrok

#### Passaggio 1: Accendi il Server del Bot (Terminale 1)

Per prima cosa, dobbiamo accendere il tuo server web locale (FastAPI) in modo che sia pronto ad ascoltare.

Apri un terminale nella cartella del tuo progetto.

Lancia il server su una porta specifica (usiamo la 3000):

```Bash
poetry run uvicorn src.main:api_app --reload --port 3000
```

> Se vedi la scritta Application startup complete, il tuo bot è sveglio e in ascolto. Lascia questa finestra aperta.

#### Passaggio 2: Apri il Tunnel Ngrok (Terminale 2)

Ora dobbiamo prendere quella porta 3000 e "lanciarla" su internet.

Apri una nuova finestra del terminale (non chiudere l'altra).

Lancia il comando per avviare ngrok sulla stessa porta:

```Bash
ngrok http 3000
```

Il terminale cambierà schermata. Cerca la riga che inizia con Forwarding e copia l'URL sicuro (https). Sarà qualcosa di simile a: `https://123a-456b.ngrok-free.app`.

> :memo: **Nota importante:** nei piani gratuiti, ogni volta che spegni e riaccendi ngrok, questo URL cambia!

#### Passaggio 3: Diciamo a Slack dove trovarti

Ora andiamo ad incollare questo nuovo indirizzo nella dashboard di Slack.

Vai su `api.slack.com/apps` e clicca sul tuo bot.

Nel menu a sinistra, vai su "Event Subscriptions".

Alla voce "Request URL", incolla l'indirizzo che hai copiato da ngrok e aggiungi alla fine l'endpoint che abbiamo scritto nel codice, ovvero /slack/events.
Esempio esatto: `https://123a-456b.ngrok-free.app/slack/events`

Appena lo incolli, Slack manderà un "ping" invisibile per testarlo. Se tutto è acceso, vedrai apparire una bellissima scritta verde "Verified"!

Clicca su "Save Changes" in basso a destra (se ti chiede di reinstallare l'app con un banner giallo in alto, fallo).

È il momento della verità!
Se vedi la spunta verde "Verified", significa che Slack e il tuo computer stanno comunicando perfettamente.

Apri il tuo Slack aziendale, vai nel canale dove hai invitato il bot e scrivigli:
*@NomeDelTuoBot* Qual è il budget formativo per un dipendente part-time?

Se tutto è andato a buon fine:

Il bot risponderà istantaneamente con "Sto consultando la Knowledge Base su Pinecone... ☁️⏳".

Dietro le quinte, interrogherà Pinecone e Cohere.

Pochi secondi dopo ti scriverà la risposta corretta ("250€ all'anno").

Proviamo? Fammi sapere se ti dà "Verified" e come va il test su Slack!
