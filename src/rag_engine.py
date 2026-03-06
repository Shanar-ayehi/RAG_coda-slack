import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore

from src.coda_client import get_all_pages_in_doc, export_page_to_markdown

load_dotenv()
cohere_api_key = os.environ.get("COHERE_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
index_name = os.environ.get("PINECONE_INDEX_NAME", "coda-rag-index")

# Inizializza i modelli di Cohere
llm = ChatCohere(model="command-r-plus-08-2024", cohere_api_key=cohere_api_key)
embeddings = CohereEmbeddings(model="embed-multilingual-v3.0", cohere_api_key=cohere_api_key)

def load_knowledge_base(doc_id: str):
    """Scarica tutte le pagine da Coda, svuota Pinecone e carica i nuovi vettori"""
    print("☁️ Inizio sincronizzazione massiva verso Pinecone...")
    
    # 0. Svuotiamo il database Pinecone per evitare duplicati
    print("🧹 Pulizia del database vettoriale in corso...")
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(index_name)
        index.delete(delete_all=True)
    except Exception as e:
        # Se l'indice è già vuoto (o non esiste il namespace), ignoriamo l'errore e continuiamo
        print("ℹ️ Il database era già vuoto o appena creato. Procedo col caricamento...")
    
    # 1. Otteniamo la lista di tutte le pagine
    page_ids = get_all_pages_in_doc(doc_id)
    tutti_i_frammenti = []
    
    # 2. Prepariamo il "coltello" per tagliare il testo
    headers_to_split_on = [
        ("#", "Titolo Principale"),
        ("##", "Sottotitolo"),
        ("###", "Sezione")
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    # 3. Ciclo FOR: Scarichiamo e tagliamo ogni singola pagina
    for pid in page_ids:
        try:
            testo_md = export_page_to_markdown(doc_id, pid)
            frammenti_pagina = markdown_splitter.split_text(testo_md)
            tutti_i_frammenti.extend(frammenti_pagina)
            time.sleep(1) 
        except Exception as e:
            print(f"⚠️ Errore durante il download della pagina {pid}: {e}")

    if not tutti_i_frammenti:
        print("❌ Nessun testo trovato. Sincronizzazione annullata.")
        return None

    print(f"🔪 Testo diviso in {len(tutti_i_frammenti)} frammenti totali. Invio a Pinecone...")

    # 4. Salviamo l'intero malloppo nel database vettoriale
    vectorstore = PineconeVectorStore.from_documents(
        documents=tutti_i_frammenti,
        embedding=embeddings,
        index_name=index_name
    )
    print("✅ Intera Knowledge Base caricata su Pinecone ed è pronta!")
    return vectorstore

def ask_bot(user_query: str) -> str:
    """Cerca nel database e genera la risposta con l'LLM"""
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) 
    
    # Istruzioni di base per l'AI
    system_prompt = (
        "Sei l'assistente AI aziendale. Usa i frammenti di contesto recuperati per rispondere alla domanda. "
        "Se non conosci la risposta in base al contesto, di' semplicemente che non lo sai.\n\n"
        "Contesto recuperato:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    response = rag_chain.invoke({"input": user_query})
    return response["answer"]