import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import ChatBot from "./components/ChatBot";
import ChatInput from "./components/ChatInput";
import GraphCanvas from "./components/GraphCanvas";
import { DEFAULT_NODE_COLOR, NODE_COLORS } from "./constants/nodeColors";
import { addTest, getGraph, getTests, type ChatContext, type GraphEdge, type GraphNode, type TestNode } from "./services/api";

const DEFAULT_PATHWAY = "Abdominal Ultrasound";

// Node types visible on initial load. All others start hidden.
const DEFAULT_VISIBLE_TYPES = new Set(["Symptom", "Condition", "Diagnostic_Test"]);
const INITIAL_HIDDEN_TYPES  = new Set(
  Object.keys(NODE_COLORS).filter((t) => !DEFAULT_VISIBLE_TYPES.has(t))
);

export default function App() {
  const [nodes, setNodes]           = useState<GraphNode[]>([]);
  const [edges, setEdges]           = useState<GraphEdge[]>([]);
  const [hiddenTypes, setHiddenTypes] = useState<Set<string>>(INITIAL_HIDDEN_TYPES);
  const [allTestNodes, setAllTestNodes] = useState<TestNode[]>([]);
  const [newNodeIds, setNewNodeIds] = useState<Set<string>>(new Set());
  const [loading, setLoading]       = useState(false);
  const [graphLoading, setGraphLoading] = useState(true);
  const [selectedTest, setSelectedTest] = useState<string | null>(`Test: ${DEFAULT_PATHWAY}`);
  const [toast, setToast] = useState<{ type: "success" | "error"; text: string } | null>(null);
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [focusedNodeId, setFocusedNodeId] = useState<string | null>(null);
  const [chatMode, setChatMode] = useState(false);
  const [chatContext, setChatContext] = useState<ChatContext | null>(null);
  const [chatHoveredNodeId, setChatHoveredNodeId] = useState<string | null>(null);

  // ── Load sidebar test list on mount ───────────────────────────────────────
  useEffect(() => {
    getTests().then(setAllTestNodes).catch(console.error);
  }, []);

  // ── Fetch graph whenever the selected pathway changes ─────────────────────
  useEffect(() => {
    setGraphLoading(true);
    const pathway = selectedTest ? selectedTest.replace(/^[^:]+:\s*/, "") : undefined;
    getGraph(pathway)
      .then((data) => {
        setNodes(data.nodes);
        setEdges(data.edges);
      })
      .catch(console.error)
      .finally(() => setGraphLoading(false));
  }, [selectedTest]);

  // ── Toast helper ───────────────────────────────────────────────────────────
  function showToast(type: "success" | "error", text: string) {
    if (toastTimer.current) clearTimeout(toastTimer.current);
    setToast({ type, text });
    toastTimer.current = setTimeout(() => setToast(null), 6_000);
  }

  // ── Add test handler ───────────────────────────────────────────────────────
  async function handleAddTest(diagnosticTest: string) {
    setLoading(true);
    try {
      const res = await addTest(diagnosticTest);

      // Mark new nodes for highlight animation before the re-fetch lands
      setNewNodeIds(new Set(res.new_nodes.map((n) => n.id)));

      // Add any new test nodes to the sidebar list
      const newTests = res.new_nodes
        .filter((n) => (n.node_type ?? n.type) === "Diagnostic_Test")
        .map((n) => ({ id: n.id, label: n.id.replace(/^[^:]+:\s*/, "") }));
      if (newTests.length > 0) {
        setAllTestNodes((prev) => {
          const existingIds = new Set(prev.map((t) => t.id));
          const incoming = newTests.filter((t) => !existingIds.has(t.id));
          return [...prev, ...incoming].sort((a, b) => a.label.localeCompare(b.label));
        });
      }

      // Auto-select the new test — triggers the pathway-fetch useEffect
      const newTestNode = res.new_nodes.find(
        (n) => (n.node_type ?? n.type) === "Diagnostic_Test"
      );
      if (newTestNode) setSelectedTest(newTestNode.id);

      showToast(
        "success",
        `Added ${res.new_rules_added} rules for "${diagnosticTest}". ` +
          `Graph: ${res.total_nodes} nodes · ${res.total_edges} edges.`
      );
    } catch (err) {
      showToast("error", err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  // ── Toggle a node type's visibility ───────────────────────────────────────
  function toggleType(type: string) {
    setHiddenTypes((prev) => {
      const next = new Set(prev);
      if (next.has(type)) next.delete(type);
      else next.add(type);
      return next;
    });
  }

  // ── Filtered graph data ────────────────────────────────────────────────────
  const visibleNodes = useMemo(
    () => nodes.filter((n) => !hiddenTypes.has((n.node_type ?? n.type) as string ?? "")),
    [nodes, hiddenTypes]
  );

  const visibleNodeIds = useMemo(
    () => new Set(visibleNodes.map((n) => n.id)),
    [visibleNodes]
  );

  const visibleEdges = useMemo(
    () => edges.filter((e) => {
      const src = typeof e.source === "string" ? e.source : (e.source as any).id;
      const tgt = typeof e.target === "string" ? e.target : (e.target as any).id;
      return visibleNodeIds.has(src) && visibleNodeIds.has(tgt);
    }),
    [edges, visibleNodeIds]
  );

  // ── Search results ─────────────────────────────────────────────────────────
  const searchResults = useMemo(() => {
    if (!searchQuery.trim()) return [];
    const q = searchQuery.toLowerCase();
    return nodes.filter((n) => {
      const label = n.id.replace(/^[^:]+:\s*/, "").toLowerCase();
      return label.includes(q);
    });
  }, [nodes, searchQuery]);

  // ── Focused 1-hop subgraph (client-side) ──────────────────────────────────
  const focusedNodes = useMemo(() => {
    if (!focusedNodeId) return null;
    const neighborIds = new Set<string>([focusedNodeId]);
    edges.forEach((e) => {
      const src = typeof e.source === "string" ? e.source : (e.source as any).id;
      const tgt = typeof e.target === "string" ? e.target : (e.target as any).id;
      if (src === focusedNodeId) neighborIds.add(tgt);
      if (tgt === focusedNodeId) neighborIds.add(src);
    });
    return nodes.filter((n) => neighborIds.has(n.id));
  }, [focusedNodeId, nodes, edges]);

  const focusedEdges = useMemo(() => {
    if (!focusedNodeId) return null;
    return edges.filter((e) => {
      const src = typeof e.source === "string" ? e.source : (e.source as any).id;
      const tgt = typeof e.target === "string" ? e.target : (e.target as any).id;
      return src === focusedNodeId || tgt === focusedNodeId;
    });
  }, [focusedNodeId, edges]);

  // ── Node-type counts for legend (always from full unfiltered set) ──────────
  const typeCounts = nodes.reduce<Record<string, number>>((acc, n) => {
    const t = (n.node_type as string | undefined) ?? "Unknown";
    acc[t] = (acc[t] ?? 0) + 1;
    return acc;
  }, {});

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="brand">
          <div className="brand-badge">Dx</div>
          <div className="brand-text">
            <span className="brand-name">Diagnotix</span>
            <span className="brand-tagline">Medical Knowledge Graph Builder</span>
          </div>
        </div>
        <div className="header-stats">
          <span className="stat-pill">{nodes.length} nodes</span>
          <span className="stat-pill">{edges.length} edges</span>
          {loading && <span className="stat-pill loading-pill">Processing…</span>}
        </div>
      </header>

      {/* ── Body ── */}
      <div className="body">
        {/* Sidebar */}
        <aside className={`sidebar${chatMode ? " sidebar--analyze" : ""}`}>

          {/* ── Mode toggle ── */}
          <div className="sidebar-mode-toggle">
            <button
              className={`mode-btn${!chatMode ? " active" : ""}`}
              onClick={() => { setChatMode(false); setChatHoveredNodeId(null); }}
            >
              Navigate
            </button>
            <button
              className={`mode-btn${chatMode ? " active" : ""}`}
              onClick={() => {
                setChatContext({
                  nodes: focusedNodes ?? visibleNodes,
                  edges: focusedEdges ?? visibleEdges,
                  pathway: selectedTest ? selectedTest.replace(/^[^:]+:\s*/, "") : null,
                });
                setChatMode(true);
              }}
            >
              Analyze
            </button>
          </div>

          {chatMode && chatContext ? (
            <ChatBot
              context={chatContext}
              onHoverNode={setChatHoveredNodeId}
            />
          ) : (
            <>
              {/* ── Node search ── */}
              <p className="sidebar-title">Search</p>
              <div className="search-section">
                <input
                  className="node-search-input"
                  type="text"
                  placeholder="Search nodes…"
                  value={searchQuery}
                  onChange={(e) => {
                    setSearchQuery(e.target.value);
                    setFocusedNodeId(null);
                  }}
                />
                {searchQuery.trim() && (
                  <div className="search-results">
                    {searchResults.length === 0 ? (
                      <p className="search-no-results">No nodes found</p>
                    ) : (
                      searchResults.map((n) => {
                        const label = n.id.replace(/^[^:]+:\s*/, "");
                        const color = NODE_COLORS[n.node_type ?? ""] ?? DEFAULT_NODE_COLOR;
                        return (
                          <button
                            key={n.id}
                            className={`search-result-btn${focusedNodeId === n.id ? " active" : ""}`}
                            onClick={() => setFocusedNodeId(focusedNodeId === n.id ? null : n.id)}
                          >
                            <span className="search-result-dot" style={{ background: color }} />
                            {label}
                          </button>
                        );
                      })
                    )}
                  </div>
                )}
              </div>
              <div className="sidebar-divider" />

              {/* ── Node type legend ── */}
              <p className="sidebar-title">Node Types</p>
              <ul className="legend">
                {Object.entries(NODE_COLORS).map(([type, color]) => {
                  const hidden = hiddenTypes.has(type);
                  return (
                    <li
                      key={type}
                      className={`legend-item${hidden ? " legend-item--hidden" : ""}`}
                      onClick={() => toggleType(type)}
                      title={hidden ? `Show ${type.replace(/_/g, " ")}` : `Hide ${type.replace(/_/g, " ")}`}
                    >
                      <span
                        className="legend-dot"
                        style={{ background: hidden ? "var(--border)" : color }}
                      />
                      <span className="legend-label">{type.replace(/_/g, " ")}</span>
                      <span className="legend-count">{typeCounts[type] ?? 0}</span>
                    </li>
                  );
                })}
              </ul>

              {/* ── Pathway picker ── */}
              {allTestNodes.length > 0 && (
                <>
                  <div className="sidebar-divider" />
                  <p className="sidebar-title">Pathways</p>
                  <div className="pathway-list">
                    <button
                      className={`pathway-btn${selectedTest === null ? " active" : ""}`}
                      onClick={() => setSelectedTest(null)}
                    >
                      All Pathways
                      <span className="pathway-count">{allTestNodes.length}</span>
                    </button>
                    {allTestNodes.map(({ id, label }) => (
                      <button
                        key={id}
                        className={`pathway-btn${selectedTest === id ? " active" : ""}`}
                        onClick={() => setSelectedTest(selectedTest === id ? null : id)}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                </>
              )}

              <div className="sidebar-divider" />

              {/* ── Add pathway ── */}
              <p className="sidebar-title">Add Pathway</p>
              <ChatInput onSubmit={handleAddTest} disabled={loading} />
            </>
          )}
        </aside>

        {/* Graph canvas */}
        <main className="canvas-area">
          {nodes.length === 0 && !graphLoading ? (
            <div className="canvas-placeholder">
              <p className="placeholder-hint">
                No graph data yet. Enter a diagnostic test below to generate the first pathway.
              </p>
            </div>
          ) : (
            <div style={{ position: "relative", width: "100%", height: "100%" }}>
              {visibleNodes.length > 0 && (
                <GraphCanvas
                  nodes={focusedNodes ?? visibleNodes}
                  edges={focusedEdges ?? visibleEdges}
                  newNodeIds={newNodeIds}
                  activePathway={selectedTest ? selectedTest.replace(/^[^:]+:\s*/, "") : null}
                  highlightedNodeId={chatHoveredNodeId}
                />
              )}
              {graphLoading && (
                <div className="canvas-loading-overlay">
                  <div className="spinner" />
                </div>
              )}
            </div>
          )}
        </main>
      </div>

      {/* ── Toast ── */}
      {toast && (
        <div className={`toast toast-${toast.type}`}>
          {toast.type === "success" ? "✓" : "✕"} {toast.text}
        </div>
      )}
    </div>
  );
}
