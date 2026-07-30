import os
import logging
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Global variable for the RAG chain
rag_chain = None

def initialize_rag():
    global rag_chain
    logging.info("Initializing RAG system...")
    
    # Load the document (Supports PDF or TXT)
    doc_path = "document.pdf"
    if not os.path.exists(doc_path):
        doc_path = "document.txt"
        if not os.path.exists(doc_path):
            logging.error("No document.pdf or document.txt found. Please add the file.")
            return False
            
    logging.info(f"Loading document from {doc_path}...")
    if doc_path.endswith('.pdf'):
        loader = PyPDFLoader(doc_path)
    else:
        loader = TextLoader(doc_path, encoding='utf-8')
        
    docs = loader.load()
    
    # Split the document into semantic chunks
    logging.info("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    splits = text_splitter.split_documents(docs)
    
    # Create Vector Store
    logging.info("Creating vector store with OpenAI Embeddings...")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=OpenAIEmbeddings(api_key=OPENAI_API_KEY),
        persist_directory="./chroma_db"
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # Set up LLM and Prompts
    logging.info("Setting up LLM and Retrieval Chain...")
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
    
    logging.info("RAG system initialized successfully.")
    return True

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id, 
        text="Hello! I am the A.Di.S.U.R.C. Assistant Bot. You can ask me any question about the Call for Applications document!"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_query = update.message.text
    logging.info(f"Received query: {user_query}")
    
    if not rag_chain:
        await context.bot.send_message(
            chat_id=update.effective_chat.id, 
            text="Sorry, the document system is not initialized properly."
        )
        return
        
    try:
        # Show typing indicator
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
        
        # Invoke RAG chain
        response = rag_chain.invoke({"input": user_query})
        answer = response.get("answer", "Sorry, I couldn't generate an answer.")
        
        await context.bot.send_message(
            chat_id=update.effective_chat.id, 
            text=answer
        )
    except Exception as e:
        logging.error(f"Error handling message: {e}")
        await context.bot.send_message(
            chat_id=update.effective_chat.id, 
            text="Sorry, I encountered an error while processing your request."
        )

if __name__ == '__main__':
    if not TELEGRAM_BOT_TOKEN or not OPENAI_API_KEY:
        print("Error: Please set TELEGRAM_BOT_TOKEN and OPENAI_API_KEY in the .env file.")
        exit(1)
        
    if not initialize_rag():
        print("Failed to initialize RAG system. Make sure document.pdf or document.txt exists.")
        exit(1)
        
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
    print("Bot is starting. Press Ctrl+C to stop.")
    application.run_polling()
