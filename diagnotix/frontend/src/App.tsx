import { useEffect, useRef, useState } from "react";
import "./App.css";
import ChatInput from "./components/ChatInput";
import GraphCanvas from "./components/GraphCanvas";
import { addTest, getGraph, type GraphEdge, type GraphNode } from "./services/api";
import { NODE_COLORS } from "./constants/nodeColors";

// ── Component ─────────────────────────────────────────────────────────────────

export default function App() {
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [newNodeIds, setNewNodeIds] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);
  const [graphLoading, setGraphLoading] = useState(true);
  const [toast, setToast] = useState<{ type: "success" | "error"; text: string } | null>(null);
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ── Load graph on mount ────────────────────────────────────────────────────
  useEffect(() => {
    getGraph()
      .then((data) => {
        setNodes(data.nodes);
        setEdges(data.edges);
      })
      .catch(console.error)
      .finally(() => setGraphLoading(false));
  }, []);

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

      // Merge new nodes (deduplicate by id)
      setNodes((prev) => {
        const existingIds = new Set(prev.map((n) => n.id));
        const incoming = res.new_nodes.filter((n) => !existingIds.has(n.id));
        return [...prev, ...incoming];
      });

      // Merge new edges — use relationship as tie-breaker so parallel edges
      // with different types are preserved.  Cast source/target to string
      // because force-graph mutates them into node objects after first render.
      setEdges((prev) => {
        const existingKeys = new Set(
          prev.map((e) => `${String(e.source)}||${String(e.target)}||${e.relationship ?? ""}`)
        );
        const incoming = res.new_edges.filter(
          (e) => !existingKeys.has(`${e.source}||${e.target}||${e.relationship ?? ""}`)
        );
        return [...prev, ...incoming];
      });

      // Highlight new nodes
      setNewNodeIds(new Set(res.new_nodes.map((n) => n.id)));

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

  // ── Node-type counts for sidebar ───────────────────────────────────────────
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
        <aside className="sidebar">
          <p className="sidebar-title">Node Types</p>
          <ul className="legend">
            {Object.entries(NODE_COLORS).map(([type, color]) => (
              <li key={type} className="legend-item">
                <span className="legend-dot" style={{ background: color }} />
                <span className="legend-label">{type.replace(/_/g, " ")}</span>
                <span className="legend-count">{typeCounts[type] ?? 0}</span>
              </li>
            ))}
          </ul>

          <div className="sidebar-divider" />

          <p className="sidebar-title">Pipeline</p>
          <ol className="pipeline-steps">
            <li>Claude extracts 8–15 triage rules</li>
            <li>Rules appended to JSON store</li>
            <li>Multi-ontology grounding (EBI, Infoway)</li>
            <li>NetworkX graph rebuilt</li>
            <li>New cluster appended live</li>
          </ol>
        </aside>

        {/* Graph canvas */}
        <main className="canvas-area">
          {graphLoading ? (
            <div className="canvas-placeholder">
              <div className="spinner" />
              <p>Loading knowledge graph…</p>
            </div>
          ) : nodes.length === 0 ? (
            <div className="canvas-placeholder">
              <p className="placeholder-hint">
                No graph data yet. Enter a diagnostic test below to generate the first pathway.
              </p>
            </div>
          ) : (
            <GraphCanvas nodes={nodes} edges={edges} newNodeIds={newNodeIds} />
          )}
        </main>
      </div>

      {/* ── Chat bar ── */}
      <ChatInput onSubmit={handleAddTest} disabled={loading} />

      {/* ── Toast ── */}
      {toast && (
        <div className={`toast toast-${toast.type}`}>
          {toast.type === "success" ? "✓" : "✕"} {toast.text}
        </div>
      )}
    </div>
  );
}
