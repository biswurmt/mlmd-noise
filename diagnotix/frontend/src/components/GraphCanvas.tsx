import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D, { type ForceGraphMethods } from "react-force-graph-2d";
import type { GraphEdge, GraphNode } from "../services/api";
import { DEFAULT_NODE_COLOR, NODE_COLORS } from "../constants/nodeColors";

const LINK_COLORS: Record<string, string> = {
  INDICATES_CONDITION:      "#58a6ff",
  REQUIRES_TEST:            "#e8a838",
  DIRECTLY_INDICATES_TEST:  "#3fb950",
};
const DEFAULT_LINK_COLOR = "#6e7681";

const NEW_HIGHLIGHT_COLOR   = "#ffffff";
const HIGHLIGHT_DURATION_MS = 5_000;

// ── Internal graph types ──────────────────────────────────────────────────────
// react-force-graph mutates link objects during simulation, replacing the
// string "source"/"target" IDs with actual node-object references.

interface FGNode extends Record<string, unknown> {
  id:              string;
  node_type?:      string;
  type?:           string;
  synonyms?:       string[];
  ebi_open_code?:  string;
  snomed_ca_code?: string;
  icd10_code?:     string;
  loinc_code?:     string;
  rxcui?:          string;
  x?:              number;
  y?:              number;
}

interface FGLink extends Record<string, unknown> {
  // After simulation starts these become node-object references.
  source:                  string | FGNode;
  target:                  string | FGNode;
  relationship?:           string;
  guideline_source?:       string;
  literature_weight?:      number;
  test_literature_weight?: number;
  trial_count?:            number;
  source_url?:             string;
}

interface Props {
  nodes:      GraphNode[];
  edges:      GraphEdge[];
  newNodeIds: Set<string>;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function nodeId(n: string | FGNode): string {
  return typeof n === "string" ? n : n.id;
}

function linkWidth(link: FGLink): number {
  const w =
    (link.literature_weight as number | null | undefined) ??
    (link.test_literature_weight as number | null | undefined) ??
    0;
  return w > 0 ? Math.max(1, Math.min(6, Math.log1p(w) * 0.4)) : 1;
}

function linkColor(link: FGLink): string {
  return LINK_COLORS[link.relationship ?? ""] ?? DEFAULT_LINK_COLOR;
}

function nodeTooltip(node: FGNode): string {
  const lines: string[] = [`<b>${node.id}</b>`];
  const typ = node.node_type ?? node.type;
  if (typ) lines.push(`Type: ${typ}`);
  if (node.ebi_open_code)  lines.push(`HP/MONDO: ${node.ebi_open_code}`);
  if (node.snomed_ca_code) lines.push(`SNOMED-CT: ${node.snomed_ca_code}`);
  if (node.icd10_code)     lines.push(`ICD-10: ${node.icd10_code}`);
  if (node.loinc_code)     lines.push(`LOINC: ${node.loinc_code}`);
  if (node.rxcui)          lines.push(`RxCUI: ${node.rxcui}`);
  const syns = node.synonyms;
  if (Array.isArray(syns) && syns.length > 0) {
    lines.push(`Synonyms: ${syns.join(", ")}`);
  }
  return lines.join("<br>");
}

function linkTooltip(link: FGLink): string {
  const lines: string[] = [];
  if (link.relationship)     lines.push(`<b>${link.relationship}</b>`);
  if (link.guideline_source) lines.push(`Guideline: [${link.guideline_source}]`);
  if (link.literature_weight != null)
    lines.push(`Literature evidence: ${Number(link.literature_weight).toLocaleString()} papers`);
  if (link.test_literature_weight != null)
    lines.push(`Test literature: ${Number(link.test_literature_weight).toLocaleString()} papers`);
  if (link.trial_count != null)
    lines.push(`Clinical trials: ${link.trial_count}`);
  const src = nodeId(link.source);
  const tgt = nodeId(link.target);
  lines.push(`<span style="opacity:0.6;font-size:10px">${src} → ${tgt}</span>`);
  return lines.join("<br>") || "Edge";
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function GraphCanvas({ nodes, edges, newNodeIds }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const fgRef        = useRef<ForceGraphMethods<FGNode, FGLink>>();
  const [dims, setDims]           = useState({ width: 800, height: 600 });
  const [highlightIds, setHighlightIds] = useState<Set<string>>(new Set());

  // Responsive container size
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setDims({ width, height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Highlight new nodes, then fade after timeout
  useEffect(() => {
    if (newNodeIds.size === 0) return;
    setHighlightIds(new Set(newNodeIds));
    const t = setTimeout(() => setHighlightIds(new Set()), HIGHLIGHT_DURATION_MS);
    return () => clearTimeout(t);
  }, [newNodeIds]);

  // Configure physics forces whenever the simulation (re)starts
  const handleEngineStart = useCallback(() => {
    const fg = fgRef.current;
    if (!fg) return;
    // Strong repulsion so labels have breathing room
    (fg.d3Force("charge") as any)?.strength(-350);
    // Moderate link distance
    (fg.d3Force("link") as any)?.distance(80).iterations(3);
    // Weak centering so the graph can spread
    (fg.d3Force("center") as any)?.strength(0.05);
  }, []);

  // Fit view after the simulation settles
  const handleEngineStop = useCallback(() => {
    fgRef.current?.zoomToFit(400, 60);
  }, []);

  // Custom node painter
  const paintNode = useCallback(
    (node: FGNode, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const isNew   = highlightIds.has(node.id);
      const typ     = node.node_type ?? node.type ?? "";
      const fill    = isNew ? NEW_HIGHLIGHT_COLOR : (NODE_COLORS[typ] ?? DEFAULT_NODE_COLOR);
      const radius  = isNew ? 9 : 6;
      const nx      = node.x ?? 0;
      const ny      = node.y ?? 0;

      // Glow ring for newly added nodes
      if (isNew) {
        ctx.beginPath();
        ctx.arc(nx, ny, radius + 5, 0, 2 * Math.PI);
        ctx.fillStyle = "rgba(255,255,255,0.12)";
        ctx.fill();
      }

      // Node circle
      ctx.beginPath();
      ctx.arc(nx, ny, radius, 0, 2 * Math.PI);
      ctx.fillStyle = fill;
      ctx.fill();

      // Subtle border
      ctx.strokeStyle = isNew ? "#ffffff" : "rgba(255,255,255,0.18)";
      ctx.lineWidth   = isNew ? 1.5 : 0.8;
      ctx.stroke();

      // Label — visible only when zoomed in enough
      const fontSize = Math.max(2.5, 11 / globalScale);
      if (fontSize > 2.5) {
        const label = String(node.id).replace(/^[^:]+:\s*/, ""); // strip prefix
        ctx.font          = `${fontSize}px Inter, system-ui, sans-serif`;
        ctx.textAlign     = "center";
        ctx.textBaseline  = "top";
        ctx.fillStyle     = "rgba(230,237,243,0.85)";
        ctx.fillText(label, nx, ny + radius + 2);
      }
    },
    [highlightIds]
  );

  // Hit-test area matches the painted radius
  const paintPointerArea = useCallback(
    (node: FGNode, color: string, ctx: CanvasRenderingContext2D) => {
      const radius = highlightIds.has(node.id) ? 9 : 6;
      ctx.beginPath();
      ctx.arc(node.x ?? 0, node.y ?? 0, radius + 3, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();
    },
    [highlightIds]
  );

  // Memoize so force-graph's internal differ only triggers on real data changes
  const graphData = useMemo(
    () => ({ nodes: nodes as FGNode[], links: edges as unknown as FGLink[] }),
    [nodes, edges]
  );

  return (
    <div ref={containerRef} style={{ width: "100%", height: "100%", overflow: "hidden" }}>
      <ForceGraph2D
        ref={fgRef}
        width={dims.width}
        height={dims.height}
        graphData={graphData}
        // ── Node config ──────────────────────────────────────────────────────
        nodeId="id"
        nodeCanvasObject={paintNode}
        nodeCanvasObjectMode={() => "replace"}
        nodePointerAreaPaint={paintPointerArea}
        nodeLabel={nodeTooltip}
        // ── Link config ──────────────────────────────────────────────────────
        linkSource="source"
        linkTarget="target"
        linkColor={linkColor}
        linkWidth={linkWidth}
        linkDirectionalArrowLength={6}
        linkDirectionalArrowRelPos={1}
        linkDirectionalArrowColor={linkColor}
        linkLabel={linkTooltip}
        linkHoverPrecision={4}
        // ── Scene ────────────────────────────────────────────────────────────
        backgroundColor="#0d1117"
        // ── Simulation ──────────────────────────────────────────────────────
        warmupTicks={80}
        cooldownTicks={200}
        onEngineStart={handleEngineStart}
        onEngineStop={handleEngineStop}
      />
    </div>
  );
}
