import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D, { type ForceGraphMethods } from "react-force-graph-2d";
import type { GraphEdge, GraphNode } from "../services/api";
import { DEFAULT_NODE_COLOR, NODE_COLORS } from "../constants/nodeColors";

const LINK_COLORS: Record<string, string> = {
  INDICATES_CONDITION:     "#58a6ff",
  REQUIRES_TEST:           "#e8a838",
  DIRECTLY_INDICATES_TEST: "#3fb950",
};
const DEFAULT_LINK_COLOR    = "#6e7681";
const NEW_HIGHLIGHT_COLOR   = "#ffffff";
const HIGHLIGHT_DURATION_MS = 5_000;

// ── Internal types ────────────────────────────────────────────────────────────
// react-force-graph mutates links: source/target go from string IDs → node refs.

interface FGNode extends Record<string, unknown> {
  id:               string;
  node_type?:       string;
  type?:            string;
  synonyms?:        string[];
  ebi_open_code?:   string;
  snomed_ca_code?:  string;
  icd10_code?:      string;
  loinc_code?:      string;
  rxcui?:           string;
  guideline_source?: string;
  guideline_url?:   string;
  x?: number;
  y?: number;
}

interface FGLink extends Record<string, unknown> {
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

// ── Pure helpers ──────────────────────────────────────────────────────────────

function getId(n: string | FGNode): string {
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

function linkTooltip(link: FGLink): string {
  const lines: string[] = [];
  if (link.relationship)     lines.push(`<b>${link.relationship}</b>`);
  if (link.guideline_source) lines.push(`Guideline: [${link.guideline_source}]`);
  if (link.literature_weight != null)
    lines.push(`Literature: ${Number(link.literature_weight).toLocaleString()} papers`);
  if (link.test_literature_weight != null)
    lines.push(`Test literature: ${Number(link.test_literature_weight).toLocaleString()} papers`);
  if (link.trial_count != null)
    lines.push(`Clinical trials: ${link.trial_count}`);
  lines.push(
    `<span style="opacity:0.6;font-size:10px">${getId(link.source)} → ${getId(link.target)}</span>`
  );
  return lines.join("<br>") || "Edge";
}

// ── Tooltip content ───────────────────────────────────────────────────────────

/**
 * UMLS returns codes as full REST URLs, e.g.:
 *   https://uts-ws.nlm.nih.gov/rest/content/2025AB/source/ICD10CM/D63.1
 * Extract the final path segment as the display label and keep the full URL
 * as the href so the user can click through to the UMLS entry.
 * Non-URL values (e.g. plain codes like "HP:0001658") are returned as-is
 * with no href.
 */
function parseCode(raw: string): { label: string; href: string | null } {
  if (raw.startsWith("http://") || raw.startsWith("https://")) {
    const label = raw.split("/").filter(Boolean).pop() ?? raw;
    return { label, href: raw };
  }
  return { label: raw, href: null };
}

function CodeRow({ prefix, raw }: { prefix: string; raw: string }) {
  const { label, href } = parseCode(raw);
  return (
    <div className="gt-row">
      <span>{prefix}</span>
      {href ? (
        <a className="gt-code-link" href={href} target="_blank" rel="noreferrer">
          <code>{label}</code> ↗
        </a>
      ) : (
        <code>{label}</code>
      )}
    </div>
  );
}

function NodeTooltip({ node }: { node: FGNode }) {
  const typ      = node.node_type ?? node.type;
  const label    = String(node.id).replace(/^[^:]+:\s*/, "");
  const syns     = Array.isArray(node.synonyms) ? node.synonyms as string[] : [];

  return (
    <div className="graph-tooltip">
      <div className="gt-title">{label}</div>
      {typ && <div className="gt-type">{typ.replace(/_/g, " ")}</div>}

      {/* ── Ontology codes ── */}
      {(node.ebi_open_code || node.snomed_ca_code || node.icd10_code ||
        node.loinc_code || node.rxcui) && (
        <div className="gt-section">
          {node.ebi_open_code  && <CodeRow prefix="HP/MONDO"  raw={node.ebi_open_code  as string} />}
          {node.snomed_ca_code && <CodeRow prefix="SNOMED-CT" raw={node.snomed_ca_code as string} />}
          {node.icd10_code     && <CodeRow prefix="ICD-10"    raw={node.icd10_code     as string} />}
          {node.loinc_code     && <CodeRow prefix="LOINC"     raw={node.loinc_code     as string} />}
          {node.rxcui          && <CodeRow prefix="RxCUI"     raw={node.rxcui          as string} />}
        </div>
      )}

      {/* ── Synonyms ── */}
      {syns.length > 0 && (
        <div className="gt-section">
          <div className="gt-label">Also known as</div>
          <div className="gt-synonyms">{syns.join(" · ")}</div>
        </div>
      )}

      {/* ── Supporting documentation (Test nodes) ── */}
      {node.guideline_source && (
        <div className="gt-section">
          <div className="gt-label">Guideline</div>
          <div className="gt-row">
            <span>{node.guideline_source}</span>
          </div>
          {node.guideline_url && (
            <a
              className="gt-link"
              href={node.guideline_url as string}
              target="_blank"
              rel="noreferrer"
            >
              View source ↗
            </a>
          )}
        </div>
      )}
    </div>
  );
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function GraphCanvas({ nodes, edges, newNodeIds }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const fgRef        = useRef<ForceGraphMethods<FGNode, FGLink>>();
  const [dims, setDims]               = useState({ width: 800, height: 600 });
  const [highlightIds, setHighlightIds] = useState<Set<string>>(new Set());
  const [hoveredNode, setHoveredNode]   = useState<FGNode | null>(null);
  const [pinnedNode, setPinnedNode]     = useState<FGNode | null>(null);
  const [tooltipPos, setTooltipPos]     = useState({ x: 0, y: 0 });
  const hideTimerRef     = useRef<ReturnType<typeof setTimeout> | null>(null);
  const tooltipHoveredRef = useRef(false);

  // ── Responsive container ────────────────────────────────────────────────────
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

  // ── New-node highlight ──────────────────────────────────────────────────────
  useEffect(() => {
    if (newNodeIds.size === 0) return;
    setHighlightIds(new Set(newNodeIds));
    const t = setTimeout(() => setHighlightIds(new Set()), HIGHLIGHT_DURATION_MS);
    return () => clearTimeout(t);
  }, [newNodeIds]);

  // ── Graph data ──────────────────────────────────────────────────────────────
  // react-force-graph mutates x/y/vx/vy onto node objects in-place and mutates
  // link.source/target from strings into node references.  Passing the same
  // object references after a pathway change means the library sees "known"
  // nodes, skips re-positioning them, and renders nothing useful.
  //
  // Fix: spread FRESH copies of every node AND link object so the library
  // always treats incoming data as a clean slate.
  const graphData = useMemo(() => ({
    nodes: nodes.map(n => ({ ...n })) as FGNode[],
    links: (edges as unknown as FGLink[]).map(link => ({
      ...link,
      source: getId(link.source),
      target: getId(link.target),
    })),
  }), [nodes, edges]);

  // ── Reheat simulation on data change ───────────────────────────────────────
  // graphData already contains fresh object references (nodes spread above), so
  // ForceGraph2D picks up the new nodes via its prop.  Calling d3ReheatSimulation
  // ensures the engine actually runs from scratch instead of staying frozen.
  useEffect(() => {
    fgRef.current?.d3ReheatSimulation();
  }, [graphData]);

  // ── Physics ─────────────────────────────────────────────────────────────────
  const handleEngineStart = useCallback(() => {
    const fg = fgRef.current;
    if (!fg) return;
    (fg.d3Force("charge") as any)?.strength(-500);
    (fg.d3Force("link")   as any)?.distance(120).iterations(3);
    (fg.d3Force("center") as any)?.strength(0.05);
  }, []);

  // Re-fit when simulation naturally stops
  const handleEngineStop = useCallback(() => {
    fgRef.current?.zoomToFit(400, 80);
  }, []);

  // ── Camera: zoom to fit whenever the displayed cluster changes ─────────────
  useEffect(() => {
    const t = setTimeout(() => {
      fgRef.current?.zoomToFit(800, 50);
    }, 400);
    return () => clearTimeout(t);
  }, [graphData]);

  // ── Hover tooltip ───────────────────────────────────────────────────────────
  // When the mouse leaves a node, we delay hiding by 200 ms so the user has
  // time to move onto the tooltip popup.  If the mouse enters the tooltip
  // (tracked via tooltipHoveredRef) the hide is cancelled and the tooltip
  // stays open until the mouse leaves it.
  const handleNodeHover = useCallback((node: FGNode | null) => {
    if (node) {
      if (hideTimerRef.current) clearTimeout(hideTimerRef.current);
      setHoveredNode(node);
      setPinnedNode(node);
    } else {
      hideTimerRef.current = setTimeout(() => {
        if (!tooltipHoveredRef.current) setHoveredNode(null);
      }, 200);
    }
  }, []);

  const handleTooltipEnter = useCallback(() => {
    tooltipHoveredRef.current = true;
    if (hideTimerRef.current) clearTimeout(hideTimerRef.current);
  }, []);

  const handleTooltipLeave = useCallback(() => {
    tooltipHoveredRef.current = false;
    setHoveredNode(null);
  }, []);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    setTooltipPos({ x: e.clientX - rect.left + 16, y: e.clientY - rect.top + 16 });
  }, []);

  // ── Node painter ─────────────────────────────────────────────────────────────
  const paintNode = useCallback(
    (node: FGNode, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const isNew  = highlightIds.has(node.id);
      const isHovered = hoveredNode?.id === node.id;
      const typ    = node.node_type ?? node.type ?? "";
      const fill   = isNew ? NEW_HIGHLIGHT_COLOR : (NODE_COLORS[typ] ?? DEFAULT_NODE_COLOR);
      const radius = isNew ? 9 : isHovered ? 8 : 6;
      const nx     = node.x ?? 0;
      const ny     = node.y ?? 0;

      // ── 1. Glow ring (new nodes or hovered) ────────────────────────────────
      if (isNew || isHovered) {
        ctx.beginPath();
        ctx.arc(nx, ny, radius + 5, 0, 2 * Math.PI);
        ctx.fillStyle = isNew ? "rgba(255,255,255,0.12)" : "rgba(255,255,255,0.08)";
        ctx.fill();
      }

      // ── 2. Node circle ──────────────────────────────────────────────────────
      ctx.beginPath();
      ctx.arc(nx, ny, radius, 0, 2 * Math.PI);
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.strokeStyle = isNew || isHovered ? "#ffffff" : "rgba(255,255,255,0.25)";
      ctx.lineWidth   = isNew || isHovered ? 1.5 : 0.8 / globalScale;
      ctx.stroke();

      // ── 3. Text label below node ────────────────────────────────────────────
      const fontSize = Math.max(2.5, 11 / globalScale);
      if (fontSize <= 30) {
        const label = String(node.id).replace(/^[^:]+:\s*/, "");
        ctx.font         = `${fontSize}px Inter, system-ui, sans-serif`;
        ctx.textAlign    = "center";
        ctx.textBaseline = "top";
        ctx.fillStyle    = "rgba(230,237,243,0.90)";
        ctx.fillText(label, nx, ny + radius + 2 / globalScale);
      }
    },
    [highlightIds, hoveredNode]
  );

  // Hit-test area
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

  // ── Edge label painter ───────────────────────────────────────────────────────
  // Mode "after" — the default line + arrow is drawn first, then this overlay.
  // At paint time the library has already mutated link.source/target into node
  // object refs, so src.x / tgt.x are available.
  const paintLink = useCallback(
    (
      link:        FGLink,
      ctx:         CanvasRenderingContext2D,
      globalScale: number
    ) => {
      const label = link.relationship;
      if (!label) return;

      const src = link.source as FGNode;
      const tgt = link.target as FGNode;
      if (src?.x == null || tgt?.x == null) return;

      // Keep label a constant ~9 screen-pixels so it's always readable
      const fontSize = 9 / globalScale;
      if (fontSize > 50) return; // skip at extreme zoom-out

      const mx  = (src.x + tgt.x) / 2;
      const my  = (src.y + tgt.y) / 2;
      let angle = Math.atan2(tgt.y! - src.y!, tgt.x - src.x);
      // Prevent upside-down text
      if (angle > Math.PI / 2 || angle < -Math.PI / 2) angle += Math.PI;

      ctx.save();
      ctx.translate(mx, my);
      ctx.rotate(angle);

      ctx.font         = `${fontSize}px Inter, system-ui, sans-serif`;
      ctx.textAlign    = "center";
      ctx.textBaseline = "middle";

      const tw  = ctx.measureText(label).width;
      const pad = fontSize * 0.35;
      ctx.fillStyle = "rgba(13,17,23,0.82)";
      ctx.fillRect(-tw / 2 - pad, -fontSize / 2 - pad * 0.6, tw + pad * 2, fontSize + pad * 1.2);

      ctx.fillStyle = "rgba(230,237,243,0.92)";
      ctx.fillText(label, 0, 0);

      ctx.restore();
    },
    []
  );

  return (
    <div
      ref={containerRef}
      style={{ position: "relative", width: "100%", height: "100%", overflow: "hidden" }}
      onMouseMove={handleMouseMove}
    >
      <ForceGraph2D
        ref={fgRef}
        width={dims.width}
        height={dims.height}
        graphData={graphData}
        // ── Nodes ──────────────────────────────────────────────────────────
        nodeId="id"
        nodeCanvasObject={paintNode}
        nodeCanvasObjectMode={() => "replace"}
        nodePointerAreaPaint={paintPointerArea}
        onNodeHover={handleNodeHover}
        // ── Links ──────────────────────────────────────────────────────────
        linkSource="source"
        linkTarget="target"
        linkColor={linkColor}
        linkWidth={linkWidth}
        linkDirectionalArrowLength={6}
        linkDirectionalArrowRelPos={1}
        linkDirectionalArrowColor={linkColor}
        linkCanvasObject={paintLink}
        linkCanvasObjectMode={() => "after"}
        linkLabel={linkTooltip}
        linkHoverPrecision={4}
        // ── Scene ──────────────────────────────────────────────────────────
        backgroundColor="#0d1117"
        // ── Simulation ────────────────────────────────────────────────────
        warmupTicks={80}
        cooldownTicks={200}
        onEngineStart={handleEngineStart}
        onEngineStop={handleEngineStop}
      />

      {/* ── Hover tooltip overlay ─────────────────────────────────────────── */}
      {(hoveredNode ?? pinnedNode) && (
        <div
          className="graph-tooltip-wrapper"
          style={{
            left:    tooltipPos.x,
            top:     tooltipPos.y,
            opacity: hoveredNode ? 1 : 0,
            pointerEvents: hoveredNode ? "auto" : "none",
          }}
          onMouseEnter={handleTooltipEnter}
          onMouseLeave={handleTooltipLeave}
        >
          <NodeTooltip node={(hoveredNode ?? pinnedNode)!} />
        </div>
      )}
    </div>
  );
}
