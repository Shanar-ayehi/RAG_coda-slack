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

Crea un file `.env` nella root del progetto e inserisci le seguenti chiavi:

```Bash
SLACK_BOT_TOKEN=xoxb-tuo-token-slack-qui
SLACK_SIGNING_SECRET=tuo-signing-secret-slack-qui
COHERE_API_KEY=tua-api-key-cohere-qui
CODA_API_TOKEN=tuo-token-coda-qui
PINECONE_API_KEY=tua-api-key-pinecone-qui
PINECONE_INDEX_NAME=coda-rag-index
```

### ⚠️ Nota per il Build (Errore pacchetto Poetry)

Se durante l'installazione o il deploy ricevi l'errore di build `No file/folder found for package rag-coda-slack`, significa che Poetry sta cercando di impacchettare il progetto come se fosse una libreria pubblica, basandosi sui dati inseriti in `pyproject.toml` (come `name = "rag-coda-slack"` e la versione Python `~3.12, <4.0.0`).

Per disabilitare questa funzione e usare Poetry solo per gestire le dipendenze, aggiungi questo blocco alla fine del file `pyproject.toml`:

```Ini, TOML
[tool.poetry]
package-mode = false
In alternativa, puoi lanciare il comando di installazione usando il flag --no-root.
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

1. Vai sul sito ufficiale per sviluppatori: `api.slack.com/apps` e fai il login con il tuo account Slack.

2. Clicca sul bottone verde **"Create New App"**.

3. Scegli **"From scratch"** (Da zero).

4. Dai un nome al tuo bot (es. Knowledge Base Bot o Coda RAG).

5. **Passaggio fondamentale:** Nel menu a tendina **"Pick a workspace"**, seleziona il tuo Workspace lavorativo.

6. Clicca su "Create App".

#### Fase 2: Dare i permessi al Bot (Gli "Scopes")

Il bot appena nato è sordo e muto. Dobbiamo dirgli esplicitamente cosa può fare.

1. Nel menu a sinistra, clicca su "OAuth & Permissions".

2. Scorri in basso fino alla sezione "Scopes" e guarda sotto "Bot Token Scopes".

3. Clicca su "Add an OAuth Scope" e aggiungi questi tre permessi fondamentali:

    * `app_mentions:read` Permette al bot di sentire quando qualcuno scrive `@Bot`.

    * `chat:write` Permette al bot di scrivere e rispondere nel canale

    * `channels:read` Permette al bot di vedere in quali canali pubblici si trova

#### Fase 3: Installarlo nel Workspace (Ottenere i Token)

Ora che ha i permessi, possiamo "assumerlo" in azienda.

1. Scorri in cima alla stessa pagina ("OAuth & Permissions") e clicca sul bottone "Install to Workspace".

2. Slack ti chiederà di confermare ("L'app richiede il permesso di accedere..."). Clicca su "Consenti" (Allow).

    🎉 Ecco il tuo primo Token! Vedrai apparire una stringa che inizia con `xoxb-... (Bot User OAuth Token)`. Copiala e incollala nel tuo file `.env` alla riga `SLACK_BOT_TOKEN`.

3. Per prendere il secondo segreto, vai nel menu a sinistra su "Basic Information"
4. Scorri fino a "App Credentials" e clicca su Show accanto a "Signing Secret".
5. Copialo e incollalo nel tuo file .env alla riga `SLACK_SIGNING_SECRET`.

#### Fase 4: Collegare le orecchie (Event Subscriptions)

Ora dobbiamo dire a Slack di mandare i messaggi al tuo computer tramite Ngrok. Assicurati che il tuo server FastAPI e ngrok siano accesi sul tuo PC.

1. Vai nel menu a sinistra su "Event Subscriptions".

2. Accendi l'interruttore "Enable Events" su On.

3. Nel campo "Request URL", incolla l'URL di ngrok e aggiungi /slack/events (es. `https://1234-abcd.ngrok-free.app/slack/events`). Aspetta un secondo e dovrebbe apparire la scritta verde "Verified".

4. Scorri in basso fino a "Subscribe to bot events", clicca su "Add Bot User Event" e aggiungi:

    * `app_mention`

5. Clicca sul bottone verde "Save Changes" in basso a destra. (Slack ti chiederà di cliccare su un banner giallo in alto "reinstall your app" per applicare le modifiche. Fallo).

##### Il tocco finale (Su Slack)

Ora il bot fa parte dell'azienda, ma non è ancora nelle stanze a chiacchierare!

1. Apri la tua applicazione di Slack (o il sito web).

2. Vai nel canale dove vuoi fare i test (es. #general o creane uno nuovo chiamato #test-bot).

3. Scrivi questo messaggio nel canale:

    >```text
    >/invite @NomeDelTuoBot (sostituendo il nome con quello che gli hai dato).
    >```

    Slack aggiungerà il bot al canale.

4. Ora scrivi:

    ```text
    @NomeDelTuoBot Qual è il budget formativo per un dipendente part-time?
   ```

### Come Collegare Ngrok

#### Passaggio 1: Accendi il Server del Bot (Terminale 1)

Per prima cosa, dobbiamo accendere il tuo server web locale (FastAPI) in modo che sia pronto ad ascoltare.

1. Apri un terminale nella cartella del tuo progetto.

2. Lancia il server su una porta specifica (usiamo la 3000):

    ```Bash
    poetry run uvicorn src.main:api_app --reload --port 3000
    ```

> *Se vedi la scritta Application startup complete, il tuo bot è sveglio e in ascolto. Lascia questa finestra aperta.*

#### Passaggio 2: Apri il Tunnel Ngrok (Terminale 2)

Ora dobbiamo prendere quella porta 3000 e "lanciarla" su internet.

1. Apri una nuova finestra del terminale (non chiudere l'altra).

2. Lancia il comando per avviare ngrok sulla stessa porta:

    ```Bash
    ngrok http 3000
    ```

3. Il terminale cambierà schermata. Cerca la riga che inizia con Forwarding e copia l'URL sicuro (https). Sarà qualcosa di simile a: `https://123a-456b.ngrok-free.app`.

> :memo: ***Nota importante:** nei piani gratuiti, ogni volta che spegni e riaccendi ngrok, questo URL cambia!*

#### Passaggio 3: Diciamo a Slack dove trovarti

Ora andiamo ad incollare questo nuovo indirizzo nella dashboard di Slack.

1. Vai su `api.slack.com/apps` e clicca sul tuo bot.

2. Nel menu a sinistra, vai su **"Event Subscriptions"**.

3. Alla voce **"Request URL"**, incolla l'indirizzo che hai copiato da ngrok e aggiungi alla fine l'endpoint che abbiamo scritto nel codice, ovvero `/slack/events`.
Esempio esatto: `https://123a-456b.ngrok-free.app/slack/events`

4. Appena lo incolli, Slack manderà un "ping" invisibile per testarlo. Se tutto è acceso, vedrai apparire una bellissima scritta verde <font color="green">*"Verified"*</font>!

5. Clicca su **"Save Changes"** in basso a destra (se ti chiede di reinstallare l'app con un banner giallo in alto, fallo).

6. È il momento della verità!
Se vedi la spunta verde <font color="green">*"Verified"*</font>, significa che Slack e il tuo computer stanno comunicando perfettamente.

7. Apri il tuo Slack aziendale, vai nel canale dove hai invitato il bot e scrivigli:
*@NomeDelTuoBot* Qual è il budget formativo per un dipendente part-time?

    Se tutto è andato a buon fine:

    * Il bot risponderà istantaneamente con "Sto consultando la Knowledge Base su Pinecone... ☁️⏳".

    * Dietro le quinte, interrogherà Pinecone e Cohere.

    * Pochi secondi dopo ti scriverà la risposta corretta ("250€ all'anno").

Proviamo? Fammi sapere se ti dà "Verified" e come va il test su Slack!

### 🚀 Deploy Definitivo su Render (Produzione)

Ngrok è perfetto per il test locale, ma per mettere il bot in produzione 24/7 devi spostarlo su un server cloud.

1. **Push su GitHub:** Carica tutto il codice su un repository GitHub privato (assicurati che il file .env sia ignorato dal .gitignore!).

2. **Crea il Web Service:** Vai su Render.com, crea un nuovo "Web Service" e collegalo alla tua repo.

3. **Configurazione:**

    * **Runtime:** `Python 3`

    * **Build Command:** `pip install poetry && poetry install --without dev --no-root`

    * **Start Command:** `poetry run uvicorn src.main:api_app --host 0.0.0.0 --port $PORT`

4. **Variabili d'Ambiente:** Nella scheda Environment Variables di Render, copia e incolla tutte le chiavi che hai nel tuo `.env` locale. Aggiungi anche una variabile `PYTHON_VERSION` impostata su `3.12.x` per allinearla al tuo `pyproject.toml`.

5. **Aggiornamento Slack:** Una volta completato il deploy, copia il nuovo URL fornito da Render (es. `https://tuo-bot.onrender.com`). Vai nella dashboard di Slack alla voce Event Subscriptions e sostituisci il vecchio link di Ngrok con `https://tuo-bot.onrender.com/slack/events.`

## Ottimizzazioni

### 🔄 Sincronizzazione Automatica della Knowledge Base (Coda Webhook)

Per evitare di lanciare l'aggiornamento manuale dal terminale, puoi istruire Coda ad inviare un segnale al bot ogni volta che i documenti HR vengono aggiornati.

1. Apri il tuo documento su Coda.

2. Inserisci il Pack ufficiale **"Webhooks"**.

3. Crea un'**Automation** (es. basata sul tempo "Ogni giorno alle 02:00" oppure attivata da un bottone).

4. Scegli l'azione **Post to URL** e inserisci l'URL del tuo bot Render con la rotta dedicata:
👉 `https://tuo-bot.onrender.com/coda-update`

Non appena l'automazione scatterà, il bot svuoterà autonomamente il database vettoriale Pinecone e ricaricherà i frammenti Markdown aggiornati.

### ☕ Evitare il Letargo del Server (Keep-Awake per Slack)

I piani gratuiti di hosting come Render vanno in "sleep" dopo 15 minuti di inattività per risparmiare risorse. Se il bot dorme, ci metterà circa 50 secondi a riaccendersi al primo messaggio del mattino. Slack, avendo un timeout severo di 3 secondi, darà errore al dipendente.

Per risolvere definitivamente il problema:

1. Crea un account gratuito su **cron-job.org**.

2. Crea un nuovo Cronjob che effettui una chiamata ogni 10 minuti.

3. Punta l'URL alla pagina della documentazione automatica del tuo bot:
👉 `https://tuo-bot.onrender.com/docs`

Questa pagina risponde sempre con un `200 OK` senza appesantire il server. Il traffico simulato ingannerà Render, mantenendo il bot sveglio, reattivo e pronto a rispondere in tempo reale.
