/**
 * createQdrantAgent
 * =================
 * One-liner factory that wires QdrantManager → LangChain tools →
 * CodeInterpreterMiddleware (PTC) → createDeepAgent.
 *
 * Peer deps required: deepagents, @langchain/quickjs, @langchain/core
 */

import { createDeepAgent } from "deepagents";
import { createCodeInterpreterMiddleware } from "@langchain/quickjs";
import { HumanMessage } from "@langchain/core/messages";
import type { StructuredToolInterface } from "@langchain/core/tools";
import { QdrantClient } from "@qdrant/js-client-rest";
import { z } from "zod";
import { QdrantManager } from "./indexManager.js";
import { makeQdrantTools } from "./tools.js";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/** Options for createQdrantAgent. */
export interface QdrantAgentOptions {
  /**
   * Model string or BaseChatModel instance, e.g. "openai:gpt-4o",
   * "anthropic:claude-sonnet-4-6", or a LangChain chat model object.
   */
  model: string;
  /** Qdrant server URL (default: "http://localhost:6333"). */
  url?: string;
  /** Qdrant Cloud API key. */
  apiKey?: string;
  /** Override the built-in system prompt. */
  systemPrompt?: string;
  /**
   * Allowlist of tool names to expose inside the QuickJS interpreter via PTC.
   * Omit to expose all ten Qdrant tools.
   */
  ptc?: string[];
  /** Per-eval timeout in milliseconds (default: 10 000). */
  executionTimeoutMs?: number;
  /** QuickJS heap cap in bytes (default: 64 MB). */
  memoryLimitBytes?: number;
  /** Max tools.* bridge calls per eval (default: 256). */
  maxPtcCalls?: number;
  /** Max characters returned from result/stdout blocks (default: 4 000). */
  maxResultChars?: number;
  /** Whether console.log output is captured (default: true). */
  captureConsole?: boolean;
  /** Expose the built-in task() global for dynamic subagents (default: true). */
  subagents?: boolean;
}

/** Handle returned by createQdrantAgent. */
export interface QdrantAgentHandle {
  /**
   * Run the agent with a natural-language query. Reusing the same threadId
   * continues the REPL session across calls (interpreter state persists).
   */
  invoke(query: string, threadId?: string): Promise<string>;
  /** Stream LangGraph events for the query. */
  stream(query: string, threadId?: string): AsyncIterable<unknown>;
  /** The shared Qdrant client used by all tools. */
  client: QdrantClient;
  /** Direct access to the index manager. */
  manager: QdrantManager;
  /** All registered LangChain tools (Qdrant tools + eval tool). */
  tools: StructuredToolInterface[];
}

// ---------------------------------------------------------------------------
// System prompt
// ---------------------------------------------------------------------------

const DEFAULT_SYSTEM_PROMPT = `\
You are an expert Qdrant vector-database agent. You create and manage Qdrant \
collections using a sandboxed JavaScript interpreter as your primary working environment.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 INTERPRETER  ("eval" tool)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The eval tool runs JavaScript inside a QuickJS sandbox:
• State (variables, functions) persists across eval calls within a session.
• The last expression is the result returned to you.
• console.log() output appears in <stdout> blocks.
• No filesystem, network, or shell — only the Qdrant tools below.
• Use await before every tools.* call — they are async.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 QDRANT TOOLS  (tools.*)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  await tools.createQdrantCollection({ collection_name, use_case, vector_size?,
      expected_count?, priority?, sparse_vectors?, memory_gb_available? })
  await tools.ensureQdrantCollection({ collection_name, use_case, vector_size? })
  await tools.listQdrantCollections()
  await tools.describeQdrantCollection({ collection_name })
  await tools.deleteQdrantCollection({ collection_name })
  await tools.upsertQdrantPoints({ collection_name, points })   // points = JSON string
  await tools.searchQdrantCollection({ collection_name, query_vector, limit?,
      score_threshold?, filter_conditions? })
  await tools.createQdrantPayloadIndex({ collection_name, field_name, field_type? })
  await tools.recommendQdrantParams({ use_case, vector_size?, expected_count?, priority? })
  await tools.optimizeQdrantCollection({ collection_name, priority? })

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 GUIDELINES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Prefer ONE eval call that batches related operations with Promise.all().
• Keep intermediate data in JS variables — return only the final summary.
• Call ensureQdrantCollection before assuming a collection exists.
• Create payload indexes for every field the user intends to filter on.
• After creating a collection explain the chosen parameters in plain English.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 EXAMPLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
\`\`\`javascript
const [col, catIdx, priceIdx] = await Promise.all([
  tools.createQdrantCollection({
    collection_name: "products",
    use_case: "semantic search with OpenAI ada-002, 500k products",
    expected_count: 500_000,
    priority: "recall",
  }),
  tools.createQdrantPayloadIndex({ collection_name: "products", field_name: "category", field_type: "keyword" }),
  tools.createQdrantPayloadIndex({ collection_name: "products", field_name: "price",    field_type: "float" }),
]);
col
\`\`\`
`;

// ---------------------------------------------------------------------------
// Response parsing schema
// ---------------------------------------------------------------------------

const AgentResultSchema = z.object({
  messages: z.array(
    z.object({
      content: z.union([z.string(), z.array(z.unknown())]),
    }),
  ),
});

function extractLastMessage(result: unknown): string {
  const parsed = AgentResultSchema.parse(result);
  const last = parsed.messages.at(-1);
  if (!last) return "";
  const { content } = last;
  if (typeof content === "string") return content;
  return content
    .filter(
      (block): block is Record<string, unknown> =>
        block !== null && typeof block === "object" && "text" in block,
    )
    .map((block) => String(block["text"]))
    .join("\n");
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/**
 * Create a Deep Agent wired to Qdrant with a QuickJS interpreter and PTC bridge.
 *
 * @example
 * ```typescript
 * import { createQdrantAgent } from "qdrant-deepagent";
 *
 * const { invoke } = createQdrantAgent({ model: "openai:gpt-4o" });
 * const reply = await invoke(
 *   "Create a semantic search collection for 500k product descriptions using ada-002"
 * );
 * console.log(reply);
 * ```
 */
export function createQdrantAgent(options: QdrantAgentOptions): QdrantAgentHandle {
  const {
    model,
    url = "http://localhost:6333",
    apiKey,
    systemPrompt = DEFAULT_SYSTEM_PROMPT,
    ptc,
    executionTimeoutMs = 10_000,
    memoryLimitBytes = 64 * 1024 * 1024,
    maxPtcCalls = 256,
    maxResultChars = 4_000,
    captureConsole = true,
    subagents = true,
  } = options;

  const client = new QdrantClient(
    apiKey != null ? { url, apiKey } : { url },
  );
  const manager = new QdrantManager(client);
  const qdrantTools = makeQdrantTools(manager);

  const ptcTools: StructuredToolInterface[] =
    ptc == null
      ? qdrantTools
      : qdrantTools.filter((t) => ptc.includes(t.name));

  const middleware = createCodeInterpreterMiddleware({
    ptc: ptcTools,
    executionTimeoutMs,
    memoryLimitBytes,
    maxPtcCalls,
    maxResultChars,
    captureConsole,
    subagents,
  });

  const agent = createDeepAgent({
    model,
    tools: qdrantTools,
    middleware: [middleware],
    systemPrompt,
  });

  // Collect all tools including the eval tool added by middleware
  const allTools: StructuredToolInterface[] = [
    ...qdrantTools,
    ...(Array.isArray(middleware.tools) ? middleware.tools : []),
  ];

  return {
    async invoke(query: string, threadId = "default"): Promise<string> {
      const result = await agent.invoke(
        { messages: [new HumanMessage(query)] },
        { configurable: { thread_id: threadId } },
      );
      return extractLastMessage(result);
    },

    async *stream(query: string, threadId = "default"): AsyncGenerator<unknown> {
      const iter = await agent.stream(
        { messages: [new HumanMessage(query)] },
        { configurable: { thread_id: threadId } },
      );
      for await (const chunk of iter) {
        yield chunk;
      }
    },

    client,
    manager,
    tools: allTools,
  };
}
