#!/usr/bin/env node
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const BASE_URL = process.env.VISION_BASE_URL;
const API_KEY = process.env.VISION_API_KEY;
const MODEL = process.env.VISION_MODEL;

if (!BASE_URL || !API_KEY || !MODEL) {
  console.error(
    "Missing env vars: VISION_BASE_URL, VISION_API_KEY, VISION_MODEL must be set"
  );
  process.exit(1);
}

const EXT_MIME = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".webp": "image/webp",
};

/** Converts a local path or an already-existing data:/http(s): URI into a data URL the vision API can consume. */
async function toDataUrl(imageInput) {
  if (imageInput.startsWith("data:")) return imageInput;

  if (imageInput.startsWith("http://") || imageInput.startsWith("https://")) {
    const res = await fetch(imageInput);
    if (!res.ok) throw new Error(`Failed to download image: HTTP ${res.status}`);
    const buf = Buffer.from(await res.arrayBuffer());
    const ext = path.extname(new URL(imageInput).pathname).toLowerCase();
    const mime = EXT_MIME[ext] || "image/png";
    return `data:${mime};base64,${buf.toString("base64")}`;
  }

  // Treat as local file path (also handles file:// URIs)
  const filePath = imageInput.startsWith("file://")
    ? fileURLToPath(imageInput)
    : imageInput;
  const bytes = await readFile(filePath);
  const ext = path.extname(filePath).toLowerCase();
  const mime = EXT_MIME[ext];
  if (!mime) throw new Error(`Unsupported image extension: ${ext}`);
  return `data:${mime};base64,${bytes.toString("base64")}`;
}

/** Calls the OpenAI-compatible vision endpoint WITHOUT any "tools" field, avoiding tool_choice errors. */
async function analyzeImage(dataUrl, question) {
  const res = await fetch(`${BASE_URL}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${API_KEY}`,
    },
    body: JSON.stringify({
      model: MODEL,
      max_tokens: 2048,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: question || "Describe this image in full detail, including all visible text, UI elements and layout." },
            { type: "image_url", image_url: { url: dataUrl } },
          ],
        },
      ],
      // Deliberately no "tools" / "tool_choice" field here.
    }),
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`Vision provider error: HTTP ${res.status} ${body}`);
  }

  const data = await res.json();
  const text = data?.choices?.[0]?.message?.content;
  if (!text) throw new Error("Vision provider returned no text content");
  return text;
}

const server = new Server(
  { name: "analyze-image-mcp", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "analyze_image",
      description:
        "Analyzes an image using a dedicated vision model and returns a detailed text description. Use this whenever the user attaches or references an image and the active model cannot see images itself. Accepts a local file path, file:// URI, http(s) URL, or data: URL.",
      inputSchema: {
        type: "object",
        properties: {
          image_path: {
            type: "string",
            description: "Local path, file:// URI, http(s) URL, or data: URL of the image to analyze.",
          },
          question: {
            type: "string",
            description: "Optional specific question about the image. If omitted, a full generic description is returned.",
          },
        },
        required: ["image_path"],
      },
    },
  ],
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name !== "analyze_image") {
    throw new Error(`Unknown tool: ${request.params.name}`);
  }

  const { image_path, question } = request.params.arguments ?? {};
  if (!image_path || typeof image_path !== "string") {
    return {
      content: [{ type: "text", text: "Error: image_path is required." }],
      isError: true,
    };
  }

  try {
    const dataUrl = await toDataUrl(image_path);
    const description = await analyzeImage(dataUrl, question);
    return { content: [{ type: "text", text: description }] };
  } catch (error) {
    return {
      content: [{ type: "text", text: `Error analyzing image: ${error.message}` }],
      isError: true,
    };
  }
});

const transport = new StdioServerTransport();
await server.connect(transport);
