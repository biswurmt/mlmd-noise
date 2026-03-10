import { useEffect, useRef, useState } from "react";
import "./App.css";
import ChatInput from "./components/ChatInput";
import GraphCanvas from "./components/GraphCanvas";
import { NODE_COLORS } from "./constants/nodeColors";
import { addTest, getGraph, getTests, type GraphEdge, type GraphNode, type TestNode } from "./services/api";

const DEFAULT_PATHWAY = "Abdominal Ultrasound";

export default function App() {
  const [nodes, setNodes]           = useState<GraphNode[]>([]);
  const [edges, setEdges]           = useState<GraphEdge[]>([]);
  const [allTestNodes, setAllTestNodes] = useState<TestNode[]>([]);
  const [newNodeIds, setNewNodeIds] = useState<Set<string>>(new Set());
  const [loading, setLoading]       = useState(false);
  const [graphLoading, setGraphLoading] = useState(true);
  const [selectedTest, setSelectedTest] = useState<string | null>(`Test: ${DEFAULT_PATHWAY}`);
  const [toast, setToast] = useState<{ type: "success" | "error"; text: string } | null>(null);
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

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

  // ── Node-type counts for legend ────────────────────────────────────────────
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

          {/* ── Pathway picker ── */}
          {allTestNodes.length > 0 && (
            <>
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
              <div className="sidebar-divider" />
            </>
          )}

          {/* ── Node type legend ── */}
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

          {/* ── Pipeline info ── */}
          <p className="sidebar-title">Pipeline</p>
          <ol className="pipeline-steps">
            <li>Gemini extracts 8–15 triage rules</li>
            <li>Rules appended to JSON store</li>
            <li>Multi-ontology grounding (EBI, Infoway)</li>
            <li>NetworkX graph rebuilt</li>
            <li>New cluster appended live</li>
          </ol>
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
              {nodes.length > 0 && (
                <GraphCanvas
                  nodes={nodes}
                  edges={edges}
                  newNodeIds={newNodeIds}
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
