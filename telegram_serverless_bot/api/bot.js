import { Pinecone } from '@pinecone-database/pinecone';
import { Redis } from '@upstash/redis';
import { OpenAIEmbeddings } from '@langchain/openai';
import { PineconeStore } from '@langchain/pinecone';
import { ChatOpenAI } from '@langchain/openai';
import { createRetrievalChain } from 'langchain/chains/retrieval';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { ChatPromptTemplate } from '@langchain/core/prompts';

export default async function handler(req, res) {
  // Only accept POST requests
  if (req.method !== 'POST') {
    return res.status(200).send('Bot is running. Please use POST for Webhooks.');
  }

  try {
    const update = req.body;
    
    // Check if the update contains a text message
    if (!update || !update.message || !update.message.text) {
      return res.status(200).send('Not a text message');
    }

    const chatId = update.message.chat.id;
    const userQuery = update.message.text;
    const telegramBotToken = process.env.TELEGRAM_BOT_TOKEN;

    // Acknowledge receipt to avoid Telegram retrying
    // For Vercel/Serverless, we can reply immediately or process and then send via API.
    // Given the LLM might take a few seconds, it's safer to send HTTP request to Telegram directly.

    if (userQuery === '/start') {
      const welcomeMessage = "Welcome to the A.Di.S.U.R.C RAG Bot! 🎓\n\nI have read the official Call for Applications document. Ask me any question about the requirements, deadlines, or scholarships, and I will answer you based on the official guidelines!";
      
      const telegramApiUrl = `https://api.telegram.org/bot${telegramBotToken}/sendMessage`;
      await fetch(telegramApiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chat_id: chatId,
          text: welcomeMessage
        })
      });
      return res.status(200).json({ success: true });
    }


    // Rate Limiting & User Tracking Logic (Upstash Redis)
    if (process.env.KV_REST_API_URL && process.env.KV_REST_API_TOKEN) {
      const redis = new Redis({
        url: process.env.KV_REST_API_URL,
        token: process.env.KV_REST_API_TOKEN,
      });

      // Track unique users
      await redis.sadd('unique_users', chatId);

      // Secret command to see how many people use the bot
      if (userQuery === '/delamDeltange') {
        const totalUsers = await redis.scard('unique_users');
        const telegramApiUrl = `https://api.telegram.org/bot${telegramBotToken}/sendMessage`;
        await fetch(telegramApiUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            chat_id: chatId,
            text: `📊 **Bot Statistics**\n\nTotal Unique Users: ${totalUsers}`
          })
        });
        return res.status(200).json({ success: true });
      }

      // Secret command to change the daily limit dynamically
      if (userQuery.startsWith('/deltangamLimit ')) {
        const newLimitStr = userQuery.split(' ')[1];
        const newLimit = parseInt(newLimitStr);
        if (!isNaN(newLimit) && newLimit > 0) {
          await redis.set('global_daily_limit', newLimit);
          const telegramApiUrl = `https://api.telegram.org/bot${telegramBotToken}/sendMessage`;
          await fetch(telegramApiUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              chat_id: chatId,
              text: `✅ **Success!** Daily limit changed to ${newLimit} questions per user.`
            })
          });
        }
        return res.status(200).json({ success: true });
      }

      const today = new Date().toISOString().split('T')[0];
      const rateLimitKey = `ratelimit:${chatId}:${today}`;

      const usageCount = await redis.incr(rateLimitKey);
      
      if (usageCount === 1) {
        await redis.expire(rateLimitKey, 86400); // Expire in 24 hours
      }

      // Read dynamic limit from Redis (default to 3)
      const limitStr = await redis.get('global_daily_limit');
      const dailyLimit = limitStr ? parseInt(limitStr) : 3;

      if (usageCount > dailyLimit) {
        const telegramApiUrl = `https://api.telegram.org/bot${telegramBotToken}/sendMessage`;
        await fetch(telegramApiUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            chat_id: chatId,
            text: `You have reached your daily limit of ${dailyLimit} questions to prevent abuse. Please come back tomorrow!`
          })
        });
        return res.status(200).json({ success: true, limited: true });
      }
    }

    // 1. Set up Pinecone & Langchain
    const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

    const embeddings = new OpenAIEmbeddings({ openAIApiKey: process.env.OPENAI_API_KEY });
    const vectorStore = await PineconeStore.fromExistingIndex(embeddings, { pineconeIndex });
    const retriever = vectorStore.asRetriever({ k: 4 });

    const llm = new ChatOpenAI({
      modelName: "gpt-4o-mini",
      temperature: 0,
      openAIApiKey: process.env.OPENAI_API_KEY
    });

    const systemPrompt = `You are an assistant for answering questions about the A.Di.S.U.R.C. Call for Applications document.
Use the following pieces of retrieved context to answer the user's question accurately.
If the answer is not in the context, reply EXACTLY with this Persian message: "متاسفانه بر اساس فایل راهنمای رسمی نتونستم جواب این سوال رو پیدا کنم. لطفاً سوالت رو با کلمات متفاوت بپرس یا از معادل‌های انگلیسی (مثل non-resident به جای fuori sede) استفاده کن."
Keep your answer concise and accurate.
Always answer in the same language that the user used to ask the question (e.g. if they ask in Persian, answer in Persian).

Context:
{context}`;

    const prompt = ChatPromptTemplate.fromMessages([
      ["system", systemPrompt],
      ["human", "{input}"]
    ]);

    const questionAnswerChain = await createStuffDocumentsChain({ llm, prompt });
    const ragChain = await createRetrievalChain({ retriever, combineDocsChain: questionAnswerChain });

    // 2. Pre-process query to map Italian jargon to English document terms for better retrieval
    let searchQuery = userQuery;
    searchQuery = searchQuery.replace(/f[ou]ori\s*sede/gi, "non-resident");
    searchQuery = searchQuery.replace(/in\s*sede/gi, "resident");
    searchQuery = searchQuery.replace(/pendolare/gi, "commuter");
    searchQuery = searchQuery.replace(/bando/gi, "Call for Applications");
    
    // Generate the answer
    const response = await ragChain.invoke({ input: searchQuery });
    const answer = response.answer || "Sorry, I couldn't generate an answer.";

    // 3. Send response back to Telegram via HTTP API
    const telegramApiUrl = `https://api.telegram.org/bot${telegramBotToken}/sendMessage`;
    await fetch(telegramApiUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        chat_id: chatId,
        text: answer
      })
    });

    // 4. Admin Notification Logic (Send both Question and Answer)
    const adminChatId = process.env.ADMIN_CHAT_ID;
    if (adminChatId && chatId.toString() !== adminChatId) {
      const username = update.message.chat.username || update.message.chat.first_name || 'Someone';
      const userId = update.message.chat.id;
      const adminMessage = `🔔 **New Q&A from @${username} (ID: ${userId})**\n\n**Q:** ${userQuery}\n\n**A:** ${answer}`;
      const adminTelegramApiUrl = `https://api.telegram.org/bot${telegramBotToken}/sendMessage`;
      
      try {
        await fetch(adminTelegramApiUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            chat_id: adminChatId,
            text: adminMessage
          })
        });
      } catch (e) {
        console.error("Failed to notify admin:", e);
      }
    }

    return res.status(200).json({ success: true });
  } catch (error) {
    console.error("Error processing request:", error);

    // If we have the chatId, try to inform the user that something went wrong
    const update = req.body;
    if (update && update.message && update.message.chat && update.message.chat.id) {
      const chatId = update.message.chat.id;
      const telegramBotToken = process.env.TELEGRAM_BOT_TOKEN;
      const telegramApiUrl = `https://api.telegram.org/bot${telegramBotToken}/sendMessage`;
      
      try {
        await fetch(telegramApiUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            chat_id: chatId,
            text: "I'm sorry, I encountered an internal error. The server might be out of API credits or experiencing downtime. Please try again later!"
          })
        });
      } catch (e) {
        console.error("Failed to send error message:", e);
      }
    }

    return res.status(500).json({ error: error.message });
  }
}
