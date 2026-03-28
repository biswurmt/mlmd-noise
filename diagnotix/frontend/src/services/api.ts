// API_BASE is empty in development — Vite proxies /api/* to localhost:8000.
// Set VITE_API_URL in a .env.production file for deployed environments.
const API_BASE = import.meta.env.VITE_API_URL ?? "";

// ── Shared types ──────────────────────────────────────────────────────────────

export interface GraphNode {
  id: string;
  node_type?: string;
  label?: string;
  [key: string]: unknown;
}

export interface GraphEdge {
  source: string;
  target: string;
  relationship?:           string;
  guideline_source?:       string; // renamed from "source" to avoid collision
  literature_weight?:      number;
  test_literature_weight?: number;
  trial_count?:            number;
  source_url?:             string;
  [key: string]: unknown;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface AddTestResponse {
  success: boolean;
  diagnostic_test: string;
  new_rules_added: number;
  new_nodes: GraphNode[];
  new_edges: GraphEdge[];
  total_nodes: number;
  total_edges: number;
  message: string;
}

// ── Helper ────────────────────────────────────────────────────────────────────

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// ── Endpoints ─────────────────────────────────────────────────────────────────

export interface TestNode {
  id: string;
  label: string;
}

/** Return the list of all Diagnostic_Test nodes (for the sidebar picker). */
export function getTests(): Promise<TestNode[]> {
  return apiFetch<TestNode[]>("/api/tests");
}

/**
 * Load graph data from the backend.
 * If *pathway* is provided, the server returns only that pathway's subgraph.
 * Omit (or pass undefined) to receive the full graph.
 */
export function getGraph(pathway?: string): Promise<GraphData> {
  const qs = pathway ? `?pathway=${encodeURIComponent(pathway)}` : "";
  return apiFetch<GraphData>(`/api/graph${qs}`);
}

/** Run the LLM pipeline for a diagnostic test; returns only the new cluster. */
export function addTest(diagnosticTest: string): Promise<AddTestResponse> {
  return apiFetch<AddTestResponse>("/api/add_test", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ diagnostic_test: diagnosticTest }),
  });
}

// ── Chat ──────────────────────────────────────────────────────────────────────

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatContext {
  nodes: GraphNode[];
  edges: GraphEdge[];
  pathway: string | null;
}

/** Send a chat message with conversation history and the current KG context. */
export function sendChatMessage(
  message: string,
  history: ChatMessage[],
  context: ChatContext,
): Promise<{ content: string }> {
  return apiFetch<{ content: string }>("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history, context }),
  });
}
