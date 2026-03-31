import os
import time
import asyncio
import re
import json
import logging
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from pinecone import Pinecone
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_cohere import ChatCohere, CohereEmbeddings, CohereRerank as RerankCohere
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.coda_client import get_all_pages_in_doc, export_page_to_markdown

load_dotenv()
cohere_api_key = os.environ.get("COHERE_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
index_name = os.environ.get("PINECONE_INDEX_NAME", "coda-rag-index")

# Global cache for BM25 documents (stateless-safe, rebuilt on each load)
_kb_documents: List[Document] = []
_query_counter: int = 0

# Inizializza i modelli di Cohere
llm = ChatCohere(model="command-r-plus-08-2024", cohere_api_key=cohere_api_key)
embeddings = CohereEmbeddings(model="embed-multilingual-v3.0", cohere_api_key=cohere_api_key)
rerank = RerankCohere(model = "rerank-multilingual-v3.0", cohere_api_key=cohere_api_key)

def enhance_query(query: str) -> str:
    """Enhance query with synonyms and related terms for better retrieval"""
    # Enhanced query expansion for marketing/tech tools
    marketing_tool_expansions = {
        # HubSpot
        'workflow': 'workflow automation automazione flusso',
        'list': 'list contact list segmentazione contatti',
        'pipeline': 'sales pipeline funnel di vendita',
        'property': 'contact property proprietà contatto',
        'form': 'form modulo landing page',
        'email': 'email newsletter campagna',
        'analytics': 'analytics metriche dati report',
        
        # Coda
        'formula': 'formula calcolo espressione',
        'pack': 'pack integrazione connessione',
        'automation': 'automation automazione processo',
        'table': 'table tabella database',
        'button': 'button pulsante azione',
        'view': 'view visualizzazione filtro',
        
        # Typeform
        'logic': 'logic jump salto logica',
        'theme': 'theme design grafica',
        'question': 'question domanda campo',
        'hidden': 'hidden fields campi nascosti',
        
        # Marketing Inbound
        'funnel': 'conversion funnel customer journey',
        'lead': 'lead prospect potenziale cliente',
        'mql': 'marketing qualified lead',
        'sql': 'sales qualified lead',
        'cta': 'call to action bottone azione',
        'landing': 'landing page pagina atterraggio',
        'seo': 'search engine optimization posizionamento',
        'crm': 'customer relationship management',
        
        # UX/UI
        'user experience': 'user experience esperienza utente',
        'ui': 'user interface interfaccia utente',
        'ux': 'user experience esperienza utente',
        'design': 'design grafica progettazione',
        
        # ADV
        'adv': 'adv advertising pubblicità',
        'campagna': 'campagna campaign marketing',
        'target': 'targeting pubblico',
        'budget': 'budget finanziamento fondi',
        
        # General Tech
        'configurazione': 'configurazione setup impostazioni',
        'best practice': 'best practice migliori pratiche',
        'troubleshooting': 'troubleshooting risoluzione problemi',
        'tutorial': 'tutorial guida passo passo',
        'esempio': 'esempio example demo'
    }
    
    enhanced_query = query.lower()
    for term, expansion in marketing_tool_expansions.items():
        if term in enhanced_query:
            enhanced_query += f" {expansion}"
    
    return enhanced_query

def create_hybrid_chunks(text: str, page_id: str) -> List[Document]:
    """Create hybrid chunks using semantic boundaries and size limits with marketing/tech optimization"""
    
    # Define tool-specific patterns for better chunking
    tool_patterns = {
        'hubspot': [
            r'(?i)workflow|automation|list|pipeline|property|form|email|analytics',
            r'(?i)contact.*property|deal.*pipeline|landing.*page|thank.*you',
        ],
        'coda': [
            r'(?i)formula|pack|automation|table|button|view',
            r'(?i)column.*formula|row.*action|pack.*integration',
        ],
        'typeform': [
            r'(?i)logic.*jump|theme|question|hidden.*field',
            r'(?i)thank.*you.*screen|results.*page|field.*logic',
        ]
    }
    
    # Define functional section patterns
    functional_patterns = {
        'configuration': r'(?i)configurazione|setup|impostazioni|parametri',
        'operations': r'(?i)operazioni|procedura|passo.*passo|workflow',
        'best_practice': r'(?i)best.*practice|raccomandazioni|linee.*guida',
        'troubleshooting': r'(?i)troubleshooting|risoluzione.*problemi|errori',
        'examples': r'(?i)esempi|demo|use.*case|caso.*d.*uso'
    }
    
    # First pass: Split by semantic boundaries (headers)
    headers_to_split_on = [
        ("#", "Titolo Principale"),
        ("##", "Sottotitolo"),
        ("###", "Sezione"),
        ("####", "Sottosezione")
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    semantic_chunks = markdown_splitter.split_text(text)
    
    # Second pass: Further split large chunks using character-based splitter with tech optimization
    final_chunks = []
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,  # Reduced from default for better precision
        chunk_overlap=100,  # Overlap for context preservation
        separators=[
            "\n\n## ",  # Major sections
            "\n\n### ",  # Subsections
            "\n\n#### ",  # Sub-subsections
            "\n\n**",     # Bold headings
            "\n\n*",      # List items
            "\n\n- ",     # Dash lists
            "\n\n1. ",    # Numbered lists
            "\n\n",       # Paragraphs
            ". ",         # Sentences
            " ",          # Words
            ""            # Characters
        ]
    )
    
    for chunk in semantic_chunks:
        if len(chunk.page_content) > 1500:  # If chunk is too large
            sub_chunks = char_splitter.split_documents([chunk])
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)
    
    # Third pass: Specialized splitting for technical content
    specialized_chunks = []
    for chunk in final_chunks:
        content = chunk.page_content
        
        # Check for code blocks, commands, or technical snippets
        if any(pattern in content.lower() for pattern in ['```', 'code', 'command', 'script', 'json', 'yaml', 'config']):
            # Split technical content more aggressively
            tech_splitter = RecursiveCharacterTextSplitter(
                chunk_size=768,  # Smaller chunks for technical content
                chunk_overlap=50,
                separators=["\n\n", "\n", ";", ".", " ", ""]
            )
            tech_chunks = tech_splitter.split_documents([chunk])
            specialized_chunks.extend(tech_chunks)
        else:
            specialized_chunks.append(chunk)
    
    # Add enriched metadata for better filtering and retrieval
    for i, chunk in enumerate(specialized_chunks):
        # Detect tool type
        tool_type = 'generic'
        content_lower = chunk.page_content.lower()
        for tool, patterns in tool_patterns.items():
            if any(re.search(pattern, content_lower) for pattern in patterns):
                tool_type = tool
                break
        
        # Detect content type
        content_type = 'generic'
        for content_cat, pattern in functional_patterns.items():
            if re.search(pattern, content_lower):
                content_type = content_cat
                break
        
        # Detect complexity level
        complexity = 'basic'
        if len(chunk.page_content) > 2000:
            complexity = 'advanced'
        elif len(chunk.page_content) > 1000:
            complexity = 'intermediate'
        
        # Calculate technical density
        technical_terms = [
            'api', 'endpoint', 'parameter', 'query', 'response', 'json', 'yaml',
            'config', 'script', 'command', 'function', 'variable', 'loop', 'condition'
        ]
        tech_density = sum(1 for term in technical_terms if term in content_lower)
        
        chunk.metadata.update({
            'chunk_id': f"{page_id}_{i}",
            'page_id': page_id,
            'chunk_size': len(chunk.page_content),
            'source_type': 'coda_page',
            'tool_type': tool_type,
            'content_type': content_type,
            'complexity': complexity,
            'technical_density': tech_density,
            'has_code_block': '```' in chunk.page_content,
            'has_commands': any(cmd in content_lower for cmd in ['command', 'script', 'bash', 'cli'])
        })
    
    return specialized_chunks

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def load_knowledge_base(doc_id: str):
    """Scarica tutte le pagine da Coda, svuota Pinecone e carica i nuovi vettori (async, with retry)"""
    global _kb_documents
    
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
    
    # 1. Otteniamo la lista di tutte le pagine (async)
    page_ids = await get_all_pages_in_doc(doc_id)
    tutti_i_frammenti = []
    
    # 2. Processiamo ogni pagina con chunking ibrido (async)
    for pid in page_ids:
        try:
            testo_md = await export_page_to_markdown(doc_id, pid)
            frammenti_pagina = create_hybrid_chunks(testo_md, pid)
            tutti_i_frammenti.extend(frammenti_pagina)
            await asyncio.sleep(1) 
        except Exception as e:
            print(f"⚠️ Errore durante il download della pagina {pid}: {e}")

    if not tutti_i_frammenti:
        print("❌ Nessun testo trovato. Sincronizzazione annullata.")
        return None

    print(f"🔪 Testo diviso in {len(tutti_i_frammenti)} frammenti ottimizzati. Invio a Pinecone...")

    # 3. Cache globale per BM25 Retriever
    _kb_documents = tutti_i_frammenti
    logger.info(f"📚 Cached {len(_kb_documents)} documents for BM25 retriever")

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
    """Create an enhanced retriever with Hybrid Search (BM25 + Vector Ensemble)"""
    global _kb_documents
    
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    
    # Vector retriever for semantic search
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 25})
    
    # BM25 retriever for keyword/exact match search
    if _kb_documents:
        bm25_retriever = BM25Retriever.from_documents(_kb_documents, k=25)
        logger.info(f"🔍 Hybrid Search enabled: BM25 ({len(_kb_documents)} docs) + Vector")
        
        # Ensemble Retriever: 40% keyword (BM25) + 60% semantic (Vector)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6]
        )
        return ensemble_retriever
    else:
        logger.warning("⚠️ No cached documents for BM25, falling back to vector-only search")
        return vector_retriever

def create_specialized_reranker():
    """Create a specialized reranker with weights for technical marketing content"""
    return RerankCohere(
        model="rerank-multilingual-v3.0", 
        top_n=3,  # Reduced from 4 to 3 for more focused context
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )

def apply_specialized_weights(retrieved_docs: List[Document], query: str) -> List[Document]:
    """Apply specialized weights for technical marketing content"""
    weighted_docs = []
    
    # Define weights for different content types
    weights = {
        'tool_specificity': 1.5,  # Documenti specifici per tool richiesto
        'step_by_step': 1.3,     # Guide passo-passo per query procedurali
        'examples': 1.2,         # Documenti con esempi pratici
        'recent_docs': 1.1,      # Documenti più recenti (tool si aggiornano)
        'inbound_relevance': 1.4 # Documenti su tematiche inbound specifiche
    }
    
    query_lower = query.lower()
    
    for doc in retrieved_docs:
        score_multiplier = 1.0
        
        # Tool specificity weight
        if hasattr(doc, 'metadata') and 'tool_type' in doc.metadata:
            tool_type = doc.metadata['tool_type']
            if tool_type != 'generic' and tool_type in query_lower:
                score_multiplier *= weights['tool_specificity']
        
        # Step-by-step weight for procedural queries
        procedural_keywords = ['come faccio', 'procedura', 'passo', 'configurare', 'setup', 'creare']
        if any(keyword in query_lower for keyword in procedural_keywords):
            if hasattr(doc, 'metadata') and doc.metadata.get('content_type') == 'operations':
                score_multiplier *= weights['step_by_step']
        
        # Examples weight
        if hasattr(doc, 'metadata') and doc.metadata.get('content_type') == 'examples':
            score_multiplier *= weights['examples']
        
        # Inbound relevance weight
        inbound_keywords = ['inbound', 'lead', 'funnel', 'mql', 'sql', 'nurturing', 'conversion']
        if any(keyword in query_lower for keyword in inbound_keywords):
            if hasattr(doc, 'metadata') and 'inbound' in doc.metadata.get('content_type', ''):
                score_multiplier *= weights['inbound_relevance']
        
        # Technical content weight
        if hasattr(doc, 'metadata') and doc.metadata.get('technical_density', 0) > 5:
            score_multiplier *= 1.1
        
        # Code block weight for technical queries
        technical_query_keywords = ['codice', 'script', 'comando', 'config', 'json', 'yaml']
        if any(keyword in query_lower for keyword in technical_query_keywords):
            if hasattr(doc, 'metadata') and doc.metadata.get('has_code_block', False):
                score_multiplier *= 1.2
        
        weighted_docs.append((doc, score_multiplier))
    
    # Sort by weighted score (assuming we can access the original score)
    # For now, just return the docs with metadata indicating weights
    return [doc for doc, _ in sorted(weighted_docs, key=lambda x: x[1], reverse=True)]

def classify_query(query: str) -> Dict[str, Any]:
    """Classify query type and tool focus for intelligent routing"""
    query_lower = query.lower()
    
    # Classify query type
    query_type = 'generic'
    if any(keyword in query_lower for keyword in ['configurare', 'setup', 'creare', 'come faccio', 'procedura']):
        query_type = 'procedural'
    elif any(keyword in query_lower for keyword in ['differenza', 'confronto', 'vs', 'meglio']):
        query_type = 'comparative'
    elif any(keyword in query_lower for keyword in ['best practice', 'migliori pratiche', 'raccomandazioni']):
        query_type = 'best_practice'
    elif any(keyword in query_lower for keyword in ['errore', 'problema', 'troubleshooting', 'non funziona']):
        query_type = 'troubleshooting'
    elif any(keyword in query_lower for keyword in ['esempio', 'demo', 'use case', 'caso']):
        query_type = 'example'
    
    # Classify tool focus
    tool_focus = 'generic'
    for tool in ['hubspot', 'coda', 'typeform']:
        if tool in query_lower:
            tool_focus = tool
            break
    
    # Calculate complexity
    complexity = 'basic'
    if len(query.split()) > 15:
        complexity = 'advanced'
    elif len(query.split()) > 8:
        complexity = 'intermediate'
    
    return {
        'query_type': query_type,
        'tool_focus': tool_focus,
        'complexity': complexity,
        'needs_examples': query_type in ['example', 'comparative'],
        'needs_steps': query_type == 'procedural',
        'needs_troubleshooting': query_type == 'troubleshooting'
    }

def apply_query_routing_weights(retrieved_docs: List[Document], query_classification: Dict[str, Any]) -> List[Document]:
    """Apply routing-based weights to retrieved documents"""
    routed_docs = []
    
    for doc in retrieved_docs:
        score_boost = 1.0
        
        if hasattr(doc, 'metadata'):
            # Boost procedural docs for procedural queries
            if query_classification['needs_steps'] and doc.metadata.get('content_type') == 'operations':
                score_boost *= 1.4
            
            # Boost examples for example/comparative queries
            if query_classification['needs_examples'] and doc.metadata.get('content_type') == 'examples':
                score_boost *= 1.3
            
            # Boost troubleshooting for problem queries
            if query_classification['needs_troubleshooting'] and doc.metadata.get('content_type') == 'troubleshooting':
                score_boost *= 1.5
            
            # Boost tool-specific docs
            if query_classification['tool_focus'] != 'generic':
                if doc.metadata.get('tool_type') == query_classification['tool_focus']:
                    score_boost *= 1.3
        
        routed_docs.append((doc, score_boost))
    
    return [doc for doc, _ in sorted(routed_docs, key=lambda x: x[1], reverse=True)]

def log_query_metrics(query: str, response: str, retrieval_time: float, llm_time: float, 
                     retrieved_docs: List[Document], success: bool):
    """Log query metrics for monitoring and debugging"""
    
    # Classify query
    query_classification = classify_query(query)
    
    # Analyze retrieved documents for specialized metrics
    tool_specificity = 0
    step_by_step_count = 0
    examples_count = 0
    technical_density_avg = 0
    code_blocks_count = 0
    
    if retrieved_docs:
        tool_types = [doc.metadata.get('tool_type', 'generic') for doc in retrieved_docs if hasattr(doc, 'metadata')]
        tool_specificity = len([t for t in tool_types if t != 'generic']) / len(tool_types) if tool_types else 0
        
        step_by_step_count = len([doc for doc in retrieved_docs 
                                 if hasattr(doc, 'metadata') and doc.metadata.get('content_type') == 'operations'])
        
        examples_count = len([doc for doc in retrieved_docs 
                             if hasattr(doc, 'metadata') and doc.metadata.get('content_type') == 'examples'])
        
        technical_density_avg = sum([doc.metadata.get('technical_density', 0) for doc in retrieved_docs 
                                    if hasattr(doc, 'metadata')]) / len(retrieved_docs)
        
        code_blocks_count = len([doc for doc in retrieved_docs 
                               if hasattr(doc, 'metadata') and doc.metadata.get('has_code_block', False)])
    
    query_type = query_classification['query_type']
    tool_focus = query_classification['tool_focus']
    
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "query_type": query_type,
        "tool_focus": tool_focus,
        "response_length": len(response),
        "retrieval_time": retrieval_time,
        "llm_time": llm_time,
        "retrieved_docs_count": len(retrieved_docs),
        "success": success,
        "avg_chunk_size": sum(len(doc.page_content) for doc in retrieved_docs) / max(len(retrieved_docs), 1) if retrieved_docs else 0,
        # Specialized technical metrics
        "tool_specificity": tool_specificity,
        "step_by_step_count": step_by_step_count,
        "examples_count": examples_count,
        "technical_density_avg": technical_density_avg,
        "code_blocks_count": code_blocks_count,
        "query_enhanced": enhance_query(query) != query.lower()
    }
    
    # Log to file for analysis
    with open("rag_metrics.log", "a") as f:
        f.write(json.dumps(metrics) + "\n")
    
    logger.info(f"Query processed: {query[:50]}... - Success: {success} - Type: {query_type} - Tool: {tool_focus}")

def log_technical_metrics_summary():
    """Log a summary of technical metrics for performance analysis"""
    try:
        with open("rag_metrics.log", "r") as f:
            lines = f.readlines()
        
        if not lines:
            return
        
        metrics = [json.loads(line) for line in lines]
        
        # Calculate technical-specific metrics
        avg_tool_specificity = sum(m.get('tool_specificity', 0) for m in metrics) / len(metrics)
        procedural_success_rate = sum(1 for m in metrics if m['query_type'] == 'procedural' and m['success']) / max(1, sum(1 for m in metrics if m['query_type'] == 'procedural'))
        code_block_utilization = sum(m.get('code_blocks_count', 0) for m in metrics) / len(metrics)
        technical_query_success = sum(1 for m in metrics if m.get('technical_density_avg', 0) > 3 and m['success']) / max(1, sum(1 for m in metrics if m.get('technical_density_avg', 0) > 3))
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(metrics),
            "avg_tool_specificity": avg_tool_specificity,
            "procedural_success_rate": procedural_success_rate,
            "code_block_utilization": code_block_utilization,
            "technical_query_success": technical_query_success,
            "tool_distribution": {
                tool: sum(1 for m in metrics if m.get('tool_focus') == tool) 
                for tool in ['hubspot', 'coda', 'typeform', 'generic']
            },
            "query_type_distribution": {
                qtype: sum(1 for m in metrics if m.get('query_type') == qtype)
                for qtype in ['procedural', 'comparative', 'best_practice', 'troubleshooting', 'example', 'generic']
            }
        }
        
        with open("technical_metrics_summary.log", "a") as f:
            f.write(json.dumps(summary) + "\n")
        
        logger.info(f"Technical metrics summary: {summary}")
        
    except Exception as e:
        logger.error(f"Error generating technical metrics summary: {e}")

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

def generate_hypothetical_answer(query: str) -> str:
    """Generate a hypothetical answer for HyDE (Hypothetical Document Embeddings)"""
    try:
        hyde_prompt = ChatPromptTemplate.from_messages([
            ("system", "Genera una risposta ipotetica breve e tecnica a questa domanda. La risposta deve contenere termini e concetti che sarebbero presenti nei documenti aziendali reali. Rispondi in italiano con massimo 100 parole."),
            ("human", "{query}")
        ])
        hyde_chain = hyde_prompt | llm | StrOutputParser()
        hypothetical = hyde_chain.invoke({"query": query})
        logger.info(f"🔮 HyDE generated hypothetical answer ({len(hypothetical)} chars)")
        return hypothetical
    except Exception as e:
        logger.warning(f"⚠️ HyDE failed, proceeding without: {e}")
        return ""

def ask_bot(user_query: str, thread_ts: str = None, channel: str = None) -> str:
    """Enhanced RAG function with Hybrid Search, HyDE, specialized weights, and quality assurance"""
    
    start_time = time.time()
    _query_counter += 1
    
    try:
        # 1. Classify query for routing
        query_classification = classify_query(user_query)
        
        # 2. Enhance the query for better retrieval
        enhanced_query = enhance_query(user_query)
        
        # 3. HyDE: Generate hypothetical answer for complex queries
        if query_classification['complexity'] in ['intermediate', 'advanced']:
            hypothetical = generate_hypothetical_answer(user_query)
            if hypothetical:
                enhanced_query = f"{enhanced_query} {hypothetical}"
        
        # 2. Get conversation history if available
        conversation_history = []
        if thread_ts and channel:
            conversation_history = get_slack_thread_history(thread_ts, channel)
        
        # 3. Create enhanced retriever (Hybrid: BM25 + Vector) and measure retrieval time
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
        
        # 5. Retrieve documents and apply specialized weights
        retrieved_docs = compression_retriever.invoke(enhanced_query)
        
        # Apply specialized weights for marketing/tech content
        weighted_docs = apply_specialized_weights(retrieved_docs, user_query)
        logger.info(f"📊 Applied specialized weights to {len(weighted_docs)} documents")
        
        retrieval_time = time.time() - retrieval_start
        
        # 6. Enhanced system prompt with better ambiguity handling
        system_prompt = (
            "Sei un esperto di Marketing Inbound e strumenti tecnici (HubSpot, Coda, Typeform). "
            "Rispondi ESCLUSIVAMENTE basandoti sui documenti forniti.\n\n"
            "TIPI DI RICHIESTE COMUNI:\n"
            "1. CONFIGURAZIONE TOOL: 'Come configuro X in HubSpot?'\n"
            "   → Fornisci passaggi specifici, screenshot se presenti nel contesto\n\n"
            "2. BEST PRACTICE: 'Quali sono le best practice per Y?'\n"
            "   → Evidenzia linee guida, esempi concreti, metriche\n\n"
            "3. DIFFERENZE/CONFRONTI: 'Differenza tra A e B?'\n"
            "   → Tabella comparativa, use case specifici\n\n"
            "4. OPERAZIONI PASSO-PASSO: 'Come faccio a Z?'\n"
            "   → Istruzioni dettagliate, prerequisiti, risultati attesi\n\n"
            "FORMATTAZIONE SPECIFICA:\n"
            "- Comandi/Configurazioni: usa code block\n"
            "- Passaggi: numerazione chiara\n"
            "- Best practice: elenchi puntati\n"
            "- Confronti: tabelle strutturate\n"
            "- Errori: sezione troubleshooting dedicata\n\n"
            "GESTIONE DELL'AMBIGUITÀ:\n"
            "Se l'utente scrive una domanda troppo breve o vaga, NON cercare di indovinare. "
            "Chiedi gentilmente all'utente di specificare meglio cosa vuole sapere. Esempi:\n"
            "   - Utente: 'workflow' → Bot: 'Vuoi sapere come creare un workflow o come ottimizzarlo?'\n"
            "   - Utente: 'form' → Bot: 'Ti riferisci alla creazione del form o all'analisi dei dati?'\n"
            "   - Utente: 'analytics' → Bot: 'Cerchi metriche specifiche o report generali?'\n\n"
            "REGOLE FONDAMENTALI:\n"
            "1. VERIDICITÀ: Se la risposta non è contenuta nel contesto, dì chiaramente che non hai questa informazione.\n"
            "2. PRECISIONE: Fornisci informazioni specifiche e dettagliate quando disponibili.\n"
            "3. TONO: Mantieni un tono cortese, professionale, chiaro e conciso.\n\n"
            "Contesto aziendale trovato:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # 7. Generate response using weighted documents directly
        llm_start = time.time()
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        
        # Format context from weighted docs
        context = "\n\n".join([doc.page_content for doc in weighted_docs])
        response = question_answer_chain.invoke({
            "context": context,
            "input": enhanced_query
        })
        llm_time = time.time() - llm_start
        
        final_response = response
        
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
        log_query_metrics(user_query, final_response, retrieval_time, llm_time, weighted_docs, confidence)
        
        # 10. Generate technical metrics summary every 10 queries
        if _query_counter % 10 == 0:
            log_technical_metrics_summary()
        
        return final_response
        
    except Exception as e:
        # Fallback response for errors
        error_response = f"Mi dispiace, ho riscontrato un problema durante l'elaborazione della tua richiesta. Per favore riprova tra qualche minuto. Errore: {str(e)}"
        log_query_metrics(user_query, error_response, 0, 0, [], False)
        return error_response
