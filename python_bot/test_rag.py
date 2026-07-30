import os
import sys
import logging
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_rag(query: str):
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not found in .env file.")
        return

    doc_path = "document.pdf"
    if not os.path.exists(doc_path):
        doc_path = "document.txt"
        if not os.path.exists(doc_path):
            print(f"Error: Could not find document.pdf or document.txt in {os.getcwd()}")
            return

    print(f"1. Loading document: {doc_path}")
    if doc_path.endswith('.pdf'):
        loader = PyPDFLoader(doc_path)
    else:
        loader = TextLoader(doc_path, encoding='utf-8')
    docs = loader.load()

    print("2. Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"   Created {len(splits)} chunks.")

    print("3. Generating embeddings and storing in local ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=OpenAIEmbeddings(api_key=OPENAI_API_KEY),
        persist_directory="./chroma_db_test"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    print("4. Initializing the LLM...")
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)
    
    system_prompt = (
        "You are an assistant for answering questions about the A.Di.S.U.R.C. Call for Applications document. "
        "Use the following pieces of retrieved context to answer the user's question accurately. "
        "If the answer is not in the context, say that you don't know based on the document. "
        "Keep your answer concise and accurate.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print(f"\n--- Testing Query ---")
    print(f"Question: {query}")
    print("Retrieving and generating answer...\n")
    
    response = rag_chain.invoke({"input": query})
    
    print("--- Answer ---")
    print(response.get("answer"))
    
    print("\n--- Source Documents Retrieved ---")
    for i, doc in enumerate(response.get("context", [])):
        print(f"Source {i+1}:")
        print(doc.page_content.strip())
        print("-" * 40)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python test_rag.py "your question here"')
    else:
        test_rag(sys.argv[1])
