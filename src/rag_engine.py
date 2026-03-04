import os
from dotenv import load_dotenv
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore

load_dotenv()
cohere_api_key = os.environ.get("COHERE_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
index_name = os.environ.get("PINECONE_INDEX_NAME", "coda-rag-index")

# 1. Inizializza i modelli di Cohere
llm = ChatCohere(model="command-r-plus-08-2024", cohere_api_key=cohere_api_key)
embeddings = CohereEmbeddings(model="embed-multilingual-v3.0", cohere_api_key=cohere_api_key)

# 2. Configura il Database Vettoriale Locale (ChromaDB)
persist_directory = "./chroma_db"

def load_knowledge_base():
    """Legge il file finto, lo taglia e lo salva in ChromaDB"""
    print("🧠 Inizializzazione Knowledge Base in corso...")
    
    # Leggiamo il nostro file finto
    with open("mock_coda.md", "r", encoding="utf-8") as f:
        markdown_document = f.read()

    # Diciamo a LangChain di tagliare il testo ogni volta che trova un "#" o "##"
    headers_to_split_on = [
        ("#", "Titolo Principale"),
        ("##", "Sottotitolo"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_document)

    # Salviamo i pezzetti nel database vettoriale Chroma
    vectorstore = PineconeVectorStore.from_documents(
        documents=md_header_splits,
        embedding=embeddings,
        index_name=index_name
    )
    print("✅ Knowledge Base caricata e pronta!")
    return vectorstore

def ask_bot(user_query: str) -> str:
    """Cerca nel database e genera la risposta con l'LLM"""
    # Carichiamo il database salvato
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # Prende i 2 pezzi più rilevanti
    
    # Creiamo le istruzioni per Gemini
    system_prompt = (
        "Sei l'assistente AI aziendale. Usa i frammenti di contesto recuperati per rispondere alla domanda. "
        "Se non conosci la risposta in base al contesto, di' semplicemente che non lo sai.\n\n"
        "Contesto recuperato:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Uniamo il tutto in una "Catena" (Chain)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # Eseguiamo la domanda
    response = rag_chain.invoke({"input": user_query})
    return response["answer"]

# --- TEST LOCALE ---
if __name__ == "__main__":
    # La prima volta carichiamo i dati
    load_knowledge_base()
    
    # Facciamo una domanda di test
    domanda = "Qual è il budget formativo per un dipendente part-time?"
    print(f"\n❓ Domanda: {domanda}")
    risposta = ask_bot(domanda)
    print(f"🤖 Risposta: {risposta}")