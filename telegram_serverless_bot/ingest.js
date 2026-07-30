import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { Pinecone } from '@pinecone-database/pinecone';
import { OpenAIEmbeddings } from '@langchain/openai';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import pdf from 'pdf-parse';
import * as dotenv from 'dotenv';

// Load environment variables
dotenv.config();

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

if (!PINECONE_API_KEY || !PINECONE_INDEX_NAME || !OPENAI_API_KEY) {
  console.error("Missing required environment variables in .env");
  process.exit(1);
}

async function ingestData() {
  console.log("Starting ingestion process...");
  
  let text = '';
  const pdfPath = path.join(__dirname, '..', 'document.pdf');
  const txtPath = path.join(__dirname, '..', 'document.txt');

  if (fs.existsSync(pdfPath)) {
    console.log("Loading document.pdf...");
    const dataBuffer = fs.readFileSync(pdfPath);
    const pdfData = await pdf(dataBuffer);
    text = pdfData.text;
  } else if (fs.existsSync(txtPath)) {
    console.log("Loading document.txt...");
    text = fs.readFileSync(txtPath, 'utf8');
  } else {
    console.error("No document.pdf or document.txt found in the parent directory.");
    process.exit(1);
  }

  console.log("Splitting document into chunks...");
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const output = await splitter.createDocuments([text]);
  console.log(`Created ${output.length} chunks.`);

  console.log("Initializing OpenAI Embeddings...");
  const embeddings = new OpenAIEmbeddings({ openAIApiKey: OPENAI_API_KEY });

  console.log("Connecting to Pinecone...");
  const pc = new Pinecone({ apiKey: PINECONE_API_KEY });
  const index = pc.Index(PINECONE_INDEX_NAME);

  console.log("Uploading vectors to Pinecone (this may take a minute)...");
  
  // Create embeddings and format for Pinecone
  const batchSize = 100;
  for (let i = 0; i < output.length; i += batchSize) {
    const batch = output.slice(i, i + batchSize);
    
    // Generate embeddings for the batch
    const texts = batch.map(doc => doc.pageContent);
    const vectors = await embeddings.embedDocuments(texts);
    
    // Prepare pinecone format
    const pineconeVectors = batch.map((doc, j) => ({
      id: `chunk_${i + j}`,
      values: vectors[j],
      metadata: { text: doc.pageContent }
    }));
    
    await index.upsert(pineconeVectors);
    console.log(`Uploaded batch ${Math.floor(i/batchSize) + 1}/${Math.ceil(output.length/batchSize)}`);
  }

  console.log("Ingestion complete!");
}

ingestData().catch(console.error);
