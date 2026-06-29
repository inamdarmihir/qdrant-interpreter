/**
 * qdrant-deepagent
 * ================
 * Qdrant vector-database plugin for Deep Agents.
 *
 * Three ways to use:
 *
 * 1. Full agent (requires deepagents + @langchain/quickjs peer deps):
 *    import { createQdrantAgent } from "qdrant-deepagent";
 *    const { invoke } = createQdrantAgent({ model: "openai:gpt-4o" });
 *
 * 2. Tools only (for any LangChain agent):
 *    import { QdrantManager, makeQdrantTools } from "qdrant-deepagent";
 *    const manager = new QdrantManager(new QdrantClient({ url }));
 *    const tools = makeQdrantTools(manager);
 *
 * 3. Interpreter skill (declared in a SKILL.md with module: ./index.ts):
 *    const qdrant = await import("@/skills/qdrant-manager");
 *    await qdrant.createCollection("products", { useCase: "..." });
 */

// Core classes and advisors
export { QdrantManager } from "./indexManager.js";
export type { AutoCreateOptions, CollectionSummary, HealthInfo, PayloadIndexResult, OptimizeResult } from "./indexManager.js";
export { UseCaseParamAdvisor, pickOptimizers } from "./paramAdvisor.js";
export type { PriorityHint, CollectionParamAdvice, AdviseInput, Distance, QuantizationMode } from "./paramAdvisor.js";

// LangChain tools factory
export { makeQdrantTools } from "./tools.js";

// Full agent factory (requires deepagents + @langchain/quickjs)
export { createQdrantAgent } from "./agent.js";
export type { QdrantAgentOptions, QdrantAgentHandle } from "./agent.js";
