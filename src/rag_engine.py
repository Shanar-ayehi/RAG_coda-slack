import os
import time
import re
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_cohere import ChatCohere, CohereEmbeddings, RerankCohere
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.coda_client import get_all_pages_in_doc, export_page_to_markdown

load_dotenv()
cohere_api_key = os.environ.get("COHERE_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
index_name = os.environ.get("PINECONE_INDEX_NAME", "coda-rag-index")

# Inizializza i modelli di Cohere
llm = ChatCohere(model="command-r-plus-08-2024", cohere_api_key=cohere_api_key)
embeddings = CohereEmbeddings(model="embed-multilingual-v3.0", cohere_api_key=cohere_api_key)
rerank = RerankCohere(model = "rerank-multilingual-v3.0", cohere_api_key=cohere_api_key)

def enhance_query(query: str) -> str:
    """Enhance query with synonyms and related terms for better retrieval"""
    # Simple query expansion for common HR terms
    query_expansions = {
        'ferie': 'ferie permessi congedo vacanze',
        'malattia': 'malattia assenza medica certificato',
        'rimborsi': 'rimborsi spese rimborsare',
        'formazione': 'formazione training corsi',
        'budget': 'budget finanziamento fondi',
        'part time': 'part-time tempo parziale',
        'full time': 'full-time tempo pieno'
    }
    
    enhanced_query = query.lower()
    for term, expansion in query_expansions.items():
        if term in enhanced_query:
            enhanced_query += f" {expansion}"
    
    return enhanced_query

def create_hybrid_chunks(text: str, page_id: str) -> List[Document]:
    """Create hybrid chunks using semantic boundaries and size limits"""
    # First pass: Split by semantic boundaries (headers)
    headers_to_split_on = [
        ("#", "Titolo Principale"),
        ("##", "Sottotitolo"),
        ("###", "Sezione")
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    semantic_chunks = markdown_splitter.split_text(text)
    
    # Second pass: Further split large chunks using character-based splitter
    final_chunks = []
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,  # Reduced from default for better precision
        chunk_overlap=100,  # Overlap for context preservation
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    for chunk in semantic_chunks:
        if len(chunk.page_content) > 1500:  # If chunk is too large
            sub_chunks = char_splitter.split_documents([chunk])
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)
    
    # Add metadata for better filtering
    for i, chunk in enumerate(final_chunks):
        chunk.metadata.update({
            'chunk_id': f"{page_id}_{i}",
            'page_id': page_id,
            'chunk_size': len(chunk.page_content),
            'source_type': 'coda_page'
        })
    
    return final_chunks

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
    
    # 2. Processiamo ogni pagina con chunking ibrido
    for pid in page_ids:
        try:
            testo_md = export_page_to_markdown(doc_id, pid)
            frammenti_pagina = create_hybrid_chunks(testo_md, pid)
            tutti_i_frammenti.extend(frammenti_pagina)
            time.sleep(1) 
        except Exception as e:
            print(f"⚠️ Errore durante il download della pagina {pid}: {e}")

    if not tutti_i_frammenti:
        print("❌ Nessun testo trovato. Sincronizzazione annullata.")
        return None

    print(f"🔪 Testo diviso in {len(tutti_i_frammenti)} frammenti ottimizzati. Invio a Pinecone...")

    # 4. Salviamo l'intero malloppo nel database vettoriale
    vectorstore = PineconeVectorStore.from_documents(
        documents=tutti_i_frammenti,
        embedding=embeddings,
        index_name=index_name
    )
    print("✅ Intera Knowledge Base caricata su Pinecone ed è pronta!")
    return vectorstore

def get_slack_thread_history(thread_ts: str, channel: str) -> List[Dict[str, Any]]:
    """Retrieve conversation history from Slack thread for context"""
    try:
        # Import Slack SDK here to avoid circular imports
        from slack_sdk import WebClient
        import os
        
        client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))
        
        # Get conversation history
        response = client.conversations_replies(
            channel=channel,
            ts=thread_ts,
            limit=10  # Limit to last 10 messages to avoid context overflow
        )
        
        messages = []
        for msg in response["messages"]:
            if msg.get("text") and not msg.get("bot_id"):  # Exclude bot messages to avoid loops
                messages.append({
                    "user": msg.get("user", "unknown"),
                    "text": msg["text"],
                    "ts": msg["ts"]
                })
        
        return messages
    except Exception as e:
        print(f"⚠️ Errore nel recupero della storia della conversazione: {e}")
        return []

def create_enhanced_retriever():
    """Create an enhanced retriever with query expansion and ensemble methods"""
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    
    # Create multiple retrievers for ensemble
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 25})
    
    # BM25 retriever for keyword matching
    # Note: This would need documents to be loaded for BM25
    # For now, we'll use just the vector retriever with enhanced query
    
    return vector_retriever

def log_query_metrics(query: str, response: str, retrieval_time: float, llm_time: float, 
                     retrieved_docs: List[Document], success: bool):
    """Log query metrics for monitoring and debugging"""
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "response_length": len(response),
        "retrieval_time": retrieval_time,
        "llm_time": llm_time,
        "retrieved_docs_count": len(retrieved_docs),
        "success": success,
        "avg_chunk_size": sum(len(doc.page_content) for doc in retrieved_docs) / max(len(retrieved_docs), 1) if retrieved_docs else 0
    }
    
    # Log to file for analysis
    with open("rag_metrics.log", "a") as f:
        f.write(json.dumps(metrics) + "\n")
    
    logger.info(f"Query processed: {query[:50]}... - Success: {success}")

def check_response_confidence(response: str) -> bool:
    """Basic confidence check for the response"""
    # Check for uncertainty indicators
    uncertainty_phrases = [
        "non sono sicuro", "non so", "non ho informazioni", 
        "non posso rispondere", "non ho trovato", "non disponibile"
    ]
    
    response_lower = response.lower()
    for phrase in uncertainty_phrases:
        if phrase in response_lower:
            return False
    
    # Check if response is too short (might indicate failure)
    if len(response.strip()) < 20:
        return False
    
    return True

def ask_bot(user_query: str, thread_ts: str = None, channel: str = None) -> str:
    """Enhanced RAG function with query expansion, context handling, and quality assurance"""
    
    start_time = time.time()
    
    try:
        # 1. Enhance the query for better retrieval
        enhanced_query = enhance_query(user_query)
        
        # 2. Get conversation history if available
        conversation_history = []
        if thread_ts and channel:
            conversation_history = get_slack_thread_history(thread_ts, channel)
        
        # 3. Create enhanced retriever and measure retrieval time
        retrieval_start = time.time()
        base_retriever = create_enhanced_retriever()
        
        # 4. Configure reranker with improved parameters
        cohere_rerank = RerankCohere(
            model="rerank-multilingual-v3.0", 
            top_n=3,  # Reduced from 4 to 3 for more focused context
            cohere_api_key=os.getenv("COHERE_API_KEY")
        )
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=cohere_rerank, 
            base_retriever=base_retriever
        )
        
        # Get retrieved documents for metrics
        retrieved_docs = compression_retriever.invoke(enhanced_query)
        retrieval_time = time.time() - retrieval_start
        
        # 5. Enhanced system prompt with better ambiguity handling
        system_prompt = (
            "Sei l'assistente virtuale HR dell'azienda. Il tuo compito è rispondere alle domande dei dipendenti "
            "basandoti ESCLUSIVAMENTE sui documenti aziendali forniti nel contesto qui sotto.\n\n"
            "REGOLE FONDAMENTALI DI COMPORTAMENTO:\n"
            "1. VERIDICITÀ: Se la risposta non è contenuta nel contesto, dì chiaramente che non hai questa informazione. Non inventare, non dedurre e non usare conoscenze esterne.\n"
            "2. GESTIONE DELL'AMBIGUITÀ: Se l'utente scrive una domanda troppo breve o vaga, NON cercare di indovinare. "
            "Chiedi gentilmente all'utente di specificare meglio cosa vuole sapere. Esempi:\n"
            "   - Utente: 'ferie' → Bot: 'Ti riferisci alla procedura per richiederle o al saldo disponibile?'\n"
            "   - Utente: 'malattia' → Bot: 'Vuoi sapere come richiedere il certificato o i giorni di permesso?'\n"
            "   - Utente: 'rimborsi' → Bot: 'Di quale tipo di rimborso stai parlando? Viaggi, formazione, o altro?'\n"
            "3. RISPOSTE PRECISE: Quando trovi la risposta nel contesto, fornisci informazioni specifiche e dettagliate.\n"
            "4. TONO: Mantieni un tono cortese, professionale, chiaro e conciso.\n\n"
            "Contesto aziendale trovato:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # 6. Create enhanced chain with better error handling
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(compression_retriever, question_answer_chain)
        
        # 7. Generate response and measure LLM time
        llm_start = time.time()
        response = rag_chain.invoke({"input": enhanced_query})
        llm_time = time.time() - llm_start
        
        final_response = response["answer"]
        
        # 8. Quality assurance checks
        confidence = check_response_confidence(final_response)
        
        if not confidence:
            final_response = (
                "Mi dispiace, ma non ho trovato informazioni sufficienti nella Knowledge Base per rispondere "
                "alla tua domanda in modo preciso. Potresti provare a:\n"
                "1. Specificare meglio la tua richiesta\n"
                "2. Usare termini più specifici\n"
                "3. Contattare l'ufficio HR direttamente per assistenza"
            )
        
        # 9. Log metrics
        total_time = time.time() - start_time
        log_query_metrics(user_query, final_response, retrieval_time, llm_time, retrieved_docs, confidence)
        
        return final_response
        
    except Exception as e:
        # Fallback response for errors
        error_response = f"Mi dispiace, ho riscontrato un problema durante l'elaborazione della tua richiesta. Per favore riprova tra qualche minuto. Errore: {str(e)}"
        log_query_metrics(user_query, error_response, 0, 0, [], False)
        return error_response
