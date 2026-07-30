# A.Di.S.U.R.C RAG Telegram Bot

This repository contains a Retrieval-Augmented Generation (RAG) system built to answer questions regarding the A.Di.S.U.R.C. Call for Applications document. The bot operates on Telegram and uses AI embeddings and LLMs to provide highly accurate, document-grounded answers.

## Project Structure

This project has been set up with two different architectural paths depending on your deployment needs:

1. **`/python_bot`**: The original Python implementation using local `ChromaDB` for the vector store. This is ideal if you are hosting the bot on a standard 24/7 server (like Render, Railway, or a VPS) that has persistent disk storage.
2. **`/telegram_serverless_bot`**: A JavaScript/Node.js refactor designed specifically for **Telegram Serverless** or webhook-based Serverless deployments (like Vercel or Cloudflare Workers). It uses a cloud-based vector database (`Pinecone`) because serverless environments are stateless.

---

## 🚀 How to Run the Serverless Bot (JavaScript)

### Prerequisites
- Node.js installed
- A **Pinecone** API Key and Index name
- An **OpenAI** API Key
- A **Telegram Bot Token** (from `@BotFather`)

### 1. Setup Environment
Navigate to the `telegram_serverless_bot` folder and copy the `.env` template:
```bash
cd telegram_serverless_bot
cp .env.example .env
```
Fill in the `.env` file with your actual keys.

### 2. Install Dependencies
```bash
npm install
```

### 3. Ingest the Document
You must upload the PDF chunks to your Pinecone index. Ensure your document is named `document.pdf` and located in the root of the repository, then run:
```bash
npm run ingest
```

### 4. Deploy
You can now deploy `bot.js` to Vercel, Cloudflare, or directly to Telegram Serverless. Don't forget to set up your Telegram webhook URL pointing to your deployed function!

### 5. Admin & Secret Commands
The serverless bot includes built-in protection against abuse (via Upstash Redis rate limiting) and allows admins to track usage:

- **Admin Notifications**: Add `ADMIN_CHAT_ID` in your Vercel Environment Variables. When someone asks the bot a question, you will receive a direct message in Telegram containing both the user's question and the bot's generated answer.
- `/delamDeltange`: A secret command that only the bot admin knows. Send this to the bot to receive a statistics dashboard showing the total number of unique users who have ever interacted with the bot.
- `/deltangamLimit <number>`: The bot defaults to a strict limit of 3 questions per user per day. To change this globally across all users, send this secret command (e.g., `/deltangamLimit 10`).

---

## 🐍 How to Run the Local Bot (Python)

### Prerequisites
- Python 3.10+
- An **OpenAI** API Key
- A **Telegram Bot Token**

### 1. Setup Environment
Navigate to the `python_bot` folder and copy the `.env` template:
```bash
cd python_bot
cp .env.example .env
```
Fill in the `.env` file with your keys.

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Test or Run
- To test the RAG retrieval locally in your terminal:
  ```bash
  python test_rag.py "What are the requirements for an independent student?"
  ```
- To start the Telegram bot (polling mode):
  ```bash
  python bot.py
  ```
