import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";
import { Logger } from "@/utils/logger";
import { env } from "@/config/env";
import { Pinecone } from "@pinecone-database/pinecone";
import { GoogleGenAI } from "@google/genai";
import { GoogleGenerativeAI } from "@google/generative-ai";

const logger = new Logger("API:Chat");

const pc = new Pinecone({ apiKey: env.PINECONE_API_KEY });
const namespace = pc.index("company-data").namespace("aven");

const ai = new GoogleGenerativeAI(env.GEMINI_API_KEY);
const embeddingmodel = ai.getGenerativeModel({ model: "gemini-embedding-001" });

const gemini = new OpenAI({
  apiKey: env.GEMINI_API_KEY,
  baseURL: "https://generativelanguage.googleapis.com/v1beta/openai/",
});

export async function POST(req: NextRequest) {
  if (req.method !== "POST") {
    return NextResponse.json({ message: "Not Found" }, { status: 404 });
  }

  try {
    const body = await req.json();
    // logger.info("Received request body:");

    const {
      model,
      messages,
      max_tokens,
      temperature,
      stream,
      call,
      ...restParams
    } = body;

    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return NextResponse.json(
        { error: "Messages array is required" },
        { status: 400 }
      );
    }

    const lastMessage = messages[messages.length - 1];
    if (!lastMessage?.content) {
      return NextResponse.json(
        { error: "Last message must have content" },
        { status: 400 }
      );
    }
    logger.info("Creating prompt modification");

    const query = lastMessage.content;

    logger.info("Query:", query);

    const embedding = await embeddingmodel.embedContent(query);

    logger.info("Logger:", embedding);

    const response = await namespace.query({
      vector: embedding.embedding.values,
      topK: 2,
      includeMetadata: true,
      includeValues: true,
    });

    logger.info("Pinecone Response:", response);

    const context = response.matches
      ?.map(match => match.metadata?.chunk_text)
      .join("\n\n");

    logger.info("Context:", context);

    const geminiPrompt = `Answer my question based on the following context:
    ${context}
    
    question:${query}
    Answer:`;

    // Get a modified prompt from Gemini
    const promptResponse = await gemini.chat.completions.create({
      model: "gemini-2.0-flash-lite",
      messages: [
        {
          role: "user",
          content: geminiPrompt,
        },
      ],
      max_tokens: 500,
      temperature: 0.7,
    });

    logger.info(promptResponse.model)

    const modifiedPrompt = promptResponse.choices?.[0]?.message?.content;

    if (!modifiedPrompt) {
      return NextResponse.json({ error: "Failed to generate modified prompt" });
    }

    const modifiedMessages = [
      ...messages.slice(0, messages.length - 1),
      { ...lastMessage, content: modifiedPrompt },
    ];
    logger.info("Creating completion");

    if (stream) {
      logger.info("inside if")
      const completionStream = await gemini.chat.completions.create({
        model: "gemini-2.0-flash-lite",
        ...restParams,
        messages: modifiedMessages,
        max_tokens: max_tokens || 150,
        temperature: temperature ?? 0.7,
        stream: true,
      } as OpenAI.Chat.ChatCompletionCreateParamsStreaming);

      logger.info("after")

      // Stream the response as NDJSON
      const encoder = new TextEncoder();
      const streamBody = new ReadableStream({
        async start(controller) {
          try {
            for await (const chunk of completionStream) {
              const data = `data: ${JSON.stringify(chunk)}\n\n`;
              controller.enqueue(encoder.encode(data));
            }
            controller.enqueue(encoder.encode("data: [Done]\n\n"));
          } catch (error) {
            logger.error("Streaming error:", error);
            controller.error(error);
          } finally {
            controller.close();
          }
        },
      });

      return new NextResponse(streamBody, {
        headers: {
          "Content-Type": "text/plain",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
      });
    } else {
      logger.info("Inside else")
      const completion = await gemini.chat.completions.create({
        model: "gemini-2.0-flash-lite",
        ...restParams,
        messages: modifiedMessages,
        max_tokens: max_tokens || 150,
        temperature: temperature ?? 0.7,
        stream: false,
      });
      logger.info("Completion created successfully.");
      return NextResponse.json(completion);
    }
  } catch (error: any) {
    logger.error("Error in chat completion", {
      message: error?.message,
      code: error?.code,
      status: error?.status,
      response: error?.response?.data || error?.response,
      error
    });

    // If using OpenAI's APIError, check like this:
    if (error instanceof OpenAI.APIError) {
      return NextResponse.json(
        { error: `API error: ${error.message}`, code: error.code },
        { status: error.status || 500 }
      );
    }

    return NextResponse.json(
      { error: error?.message || "Internal Server Error" },
      { status: 500 }
    );
  }
}
