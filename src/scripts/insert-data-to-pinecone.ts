import FirecrawlApp from "@mendable/firecrawl-js";
import dotenv from "dotenv";
import { Logger } from "@/utils/logger";
import { Pinecone } from "@pinecone-database/pinecone";
import { GoogleGenAI } from "@google/genai";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { cacheTag } from "next/dist/server/use-cache/cache-tag";

dotenv.config();

const ai = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!);
const model = ai.getGenerativeModel({ model: 'gemini-embedding-001'});
// const ai = new GoogleGenAI({
//   apiKey: process.env.GEMINI_API_KEY,
// });

const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY! });

const logger = new Logger("InsertDataToPinecone");

async function main() {
  const app = new FirecrawlApp({
    apiKey: process.env.FIRECRAWL_API_KEY,
  });

  const scrapeUrls = [
    "https://www.aven.com/support",
    "https://www.aven.com",
    "https://www.aven.com/education",
    "https://www.aven.com/about",
  ];

  for (const scrapeUrl of scrapeUrls) {
    const scrapeResult = await app.scrapeUrl(scrapeUrl, {
      formats: ["markdown"],
      onlyMainContent: true,
    });

    logger.info("Scrape result:", scrapeResult);

    if (!scrapeResult.success || !scrapeResult.markdown) {
      throw new Error("Failed to scrape content or no markdown found");
    }

    // Split the markdown into chunks (e.g., by paragraphs or every N characters)
    function chunkText(text: string, chunkSize: number = 2000): string[] {
      const chunks: string[] = [];
      let start = 0;
      while (start < text.length) {
        let end = start + chunkSize;
        // Try to break at a newline if possible
        if (end < text.length) {
          const nextNewline = text.lastIndexOf("\n", end);
          if (nextNewline > start) end = nextNewline;
        }
        chunks.push(text.slice(start, end));
        start = end;
      }
      return chunks;
    }

    const chunks = chunkText(scrapeResult.markdown, 2000);
    logger.info(`Split markdown into ${chunks.length} chunks`);

    const namespace = pc.index("company-data").namespace("aven");
    const embeddingModel = ai.getGenerativeModel({ model: "gemini-embedding-001" });
    // Call ai.embedContent directly in the loop

    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      // Get embedding for this chunk
      const response = await embeddingModel.embedContent(chunk);
      const embedding = response.embedding;
      logger.info(`Embedding for chunk ${i + 1}/${chunks.length}`, embedding);

      // Insert into Pinecone
      const pineconeResponse = await namespace.upsert([
        {
          id: `${scrapeUrl}-${Date.now()}-${i}`,
          values: embedding?.values,
          metadata: {
            chunk_text: chunk,
            category: "website",
            url: scrapeUrl,
            chunk_index: i,
          },
        },
      ]);
      logger.info(`Pinecone Response for chunk ${i + 1}`, pineconeResponse);
    }
  }
}
main();


