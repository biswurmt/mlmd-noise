import React, { type FormEvent, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { DEFAULT_NODE_COLOR, NODE_COLORS } from "../constants/nodeColors";
import {
  type ChatContext,
  type ChatMessage,
  type GraphNode,
  sendChatMessage,
} from "../services/api";

interface Props {
  context: ChatContext;
  messages: ChatMessage[];
  onMessagesChange: (msgs: ChatMessage[]) => void;
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

// Void HTML elements — React 19 throws if you pass children to these
const VOID_ELEMENTS = new Set([
  "area","base","br","col","embed","hr","img","input",
  "link","meta","param","source","track","wbr",
]);

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

// ── Enriched markdown renderer ────────────────────────────────────────────────

/** Enrich a single string child: split on node labels → chips + plain text. */
function enrichString(
  str: string,
  index: NodeEntry[],
  onHoverNode: (id: string | null) => void,
  keyPrefix: string,
) {
  return enrichText(str, index).map((seg, i) => {
    if (seg.kind === "text") return <span key={`${keyPrefix}-${i}`}>{seg.value}</span>;
    const color = NODE_COLORS[seg.type] ?? DEFAULT_NODE_COLOR;
    return (
      <span
        key={`${keyPrefix}-${i}`}
        className="chat-node-chip"
        style={{ borderColor: color, color }}
        onMouseEnter={() => onHoverNode(seg.nodeId)}
        onMouseLeave={() => onHoverNode(null)}
      >
        {seg.label}
      </span>
    );
  });
}

/** Recursively enrich React children: strings get chipified, elements pass through. */
function enrichChildren(
  children: React.ReactNode,
  index: NodeEntry[],
  onHoverNode: (id: string | null) => void,
  depth = 0,
): React.ReactNode {
  return (Array.isArray(children) ? children : [children]).flatMap((child, i) => {
    if (typeof child === "string") return enrichString(child, index, onHoverNode, `d${depth}-${i}`);
    if (child && typeof child === "object" && "props" in (child as object)) {
      const el = child as React.ReactElement<{ children?: React.ReactNode }>;
      // Void elements (br, hr, img, etc.) cannot have children in React 19 — pass through unchanged
      if (typeof el.type === "string" && VOID_ELEMENTS.has(el.type)) return el;
      return React.cloneElement(el, {}, enrichChildren(el.props.children, index, onHoverNode, depth + 1));
    }
    return child;
  });
}

function makeComponents(
  index: NodeEntry[],
  onHoverNode: (id: string | null) => void,
): Components {
  function Enrich({ children }: { children?: React.ReactNode }) {
    return <>{enrichChildren(children, index, onHoverNode)}</>;
  }
  return {
    p:  ({ children }) => <p><Enrich>{children}</Enrich></p>,
    li: ({ children }) => <li><Enrich>{children}</Enrich></li>,
    a:  ({ href, children }) => <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>,
  };
}

function MarkdownMessage({
  text,
  index,
  onHoverNode,
}: {
  text: string;
  index: NodeEntry[];
  onHoverNode: (id: string | null) => void;
}) {
  const components = useMemo(() => makeComponents(index, onHoverNode), [index, onHoverNode]);
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
      {text}
    </ReactMarkdown>
  );
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function ChatBot({ context, messages, onMessagesChange, onHoverNode }: Props) {
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
    const withUser = [...messages, userMsg];
    onMessagesChange(withUser);
    setInput("");
    setError(null);
    setLoading(true);

    try {
      const res = await sendChatMessage(trimmed, messages, context);
      onMessagesChange([...withUser, { role: "assistant", content: res.content }]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="chat-panel">
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
              <MarkdownMessage text={msg.content} index={nodeIndex} onHoverNode={onHoverNode} />
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
