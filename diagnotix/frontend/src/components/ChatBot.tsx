import { type FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { DEFAULT_NODE_COLOR, NODE_COLORS } from "../constants/nodeColors";
import {
  type ChatContext,
  type ChatMessage,
  type GraphNode,
  sendChatMessage,
} from "../services/api";

interface Props {
  context: ChatContext;
  onHoverNode: (nodeId: string | null) => void;
}

// ── Node label index ───────────────────────────────────────────────────────────

interface NodeEntry {
  label: string;
  nodeId: string;
  type: string;
}

function buildNodeIndex(nodes: GraphNode[]): NodeEntry[] {
  return nodes
    .map((n) => ({
      label: n.id.replace(/^[^:]+:\s*/, ""),
      nodeId: n.id,
      type: (n.node_type ?? "") as string,
    }))
    .filter((e) => e.label.length > 2) // skip very short labels to avoid false matches
    .sort((a, b) => b.label.length - a.label.length); // longest-first
}

// ── Text enrichment ────────────────────────────────────────────────────────────

type Segment =
  | { kind: "text"; value: string }
  | { kind: "node"; label: string; nodeId: string; type: string };

function enrichText(text: string, index: NodeEntry[]): Segment[] {
  // Build a single regex that matches any node label (case-insensitive, word boundary)
  if (index.length === 0) return [{ kind: "text", value: text }];

  const pattern = index
    .map((e) => e.label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
    .join("|");
  const re = new RegExp(`(${pattern})`, "gi");

  const parts = text.split(re);
  const labelMap = new Map(index.map((e) => [e.label.toLowerCase(), e]));

  return parts
    .filter((p) => p.length > 0)
    .map((part) => {
      const entry = labelMap.get(part.toLowerCase());
      if (entry) return { kind: "node" as const, label: part, nodeId: entry.nodeId, type: entry.type };
      return { kind: "text" as const, value: part };
    });
}

// ── Enriched message renderer ─────────────────────────────────────────────────

function EnrichedText({
  text,
  index,
  onHoverNode,
}: {
  text: string;
  index: NodeEntry[];
  onHoverNode: (id: string | null) => void;
}) {
  const segments = useMemo(() => enrichText(text, index), [text, index]);

  return (
    <>
      {segments.map((seg, i) => {
        if (seg.kind === "text") return <span key={i}>{seg.value}</span>;
        const color = NODE_COLORS[seg.type] ?? DEFAULT_NODE_COLOR;
        return (
          <span
            key={i}
            className="chat-node-chip"
            style={{ borderColor: color, color }}
            onMouseEnter={() => onHoverNode(seg.nodeId)}
            onMouseLeave={() => onHoverNode(null)}
          >
            {seg.label}
          </span>
        );
      })}
    </>
  );
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function ChatBot({ context, onHoverNode }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const nodeIndex = useMemo(() => buildNodeIndex(context.nodes), [context.nodes]);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || loading) return;

    const userMsg: ChatMessage = { role: "user", content: trimmed };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setError(null);
    setLoading(true);

    try {
      const res = await sendChatMessage(trimmed, messages, context);
      setMessages((prev) => [...prev, { role: "assistant", content: res.content }]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  const pathwayLabel = context.pathway ?? "All Pathways";

  return (
    <div className="chat-panel">
      {/* Context badge */}
      <div className="chat-context-badge">
        <span className="chat-context-dot" />
        {pathwayLabel}
      </div>

      {/* Message history */}
      <div className="chat-messages" ref={scrollRef}>
        {messages.length === 0 && (
          <p className="chat-empty-hint">
            Ask about this pathway — symptoms, conditions, recommended tests, or clinical thresholds.
          </p>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`chat-msg chat-msg-${msg.role}`}>
            {msg.role === "assistant" ? (
              <EnrichedText text={msg.content} index={nodeIndex} onHoverNode={onHoverNode} />
            ) : (
              msg.content
            )}
          </div>
        ))}
        {loading && (
          <div className="chat-msg chat-msg-assistant chat-typing">
            <span className="chat-typing-dot" />
            <span className="chat-typing-dot" />
            <span className="chat-typing-dot" />
          </div>
        )}
        {error && <p className="chat-error">{error}</p>}
      </div>

      {/* Input */}
      <form className="chat-form" onSubmit={handleSubmit}>
        <input
          className="chat-input"
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about this pathway…"
          disabled={loading}
          autoComplete="off"
          spellCheck={false}
        />
        <button className="chat-submit" type="submit" disabled={loading || !input.trim()}>
          {loading ? (
            <span className="spinner-sm" />
          ) : (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          )}
        </button>
      </form>
    </div>
  );
}
