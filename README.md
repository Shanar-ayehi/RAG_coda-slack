# RAG_coda-slack

Progetto bot x slack; RAG su KB Coda

## Struttura File Tree

> ```text
> coda-slack-rag/
> ├── .env                 # Variabili d'Ambiemente
> ├── .gitignore           # L'elenco dei file che Git deve ignorare
> ├── pyproject.toml       # La "ricetta" delle tue librerie (creato da Poetry)
> ├── poetry.lock          # Le versioni esatte delle librerie (creato da Poetry)
> ├── README.md            # Le istruzioni del tuo progetto (utile per il futuro)
> │
> └── src/                 # La cartella che contiene tutto il tuo codice
>     ├── __init__.py      # File vuoto (dice a Python che questa è una cartella di codice)
>     ├── main.py          # Il file principale (avvia FastAPI e unisce i vari pezzi)
>     ├── slack_events.py  # Logica di Slack (risposte, messaggi, bottoni)
>     ├── rag_engine.py    # Logica AI (LangChain, Gemini e Database Vettoriale)
>     └── coda_client.py   # Contiene SOLO la logica per scaricare i Markdown da Coda
> ```

## Descrizione Progetto

## Struttura Progetto

### Root Folder

| File / Cartella | Descrizione |
| :--- | :--- |
| **`.env`** | File di configurazione contenente le variabili d'ambiente **sensibili** per l'uso del Bot; |
| **`.gitignore`** | File di configurazione, Per Python; |
| **`pyproject.toml`** | Configurazione Progetto Poetry; |
| **`poetry.lock`** | Dependencies Progetto Poetry; |
| **`README.md`** | Specifiche; |

### SRC Folder

| File / Cartella | Descrizione |
| :--- | :--- |
| **`coda_client.py`** | Contiene SOLO la logica per scaricare i Markdown da Coda; |
| **`rag_engine.py`** | Logica AI (LangChain, Gemini e Database Vettoriale); |
| **`slack_events.py`** | Logica di Slack (risposte, messaggi, bottoni); |
| **`main.py`** | Il file principale (avvia FastAPI e unisce i vari pezzi); |
