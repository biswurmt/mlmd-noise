# frontend/src

React 19 + TypeScript source. Entry point is `main.tsx` → `App.tsx`.

## Layout

```
src/
├── main.tsx            # React root render
├── App.tsx             # Root component — manages all state and layout
├── App.css             # Global dark medical theme styles
├── components/         # UI components (see components/README.md)
├── services/
│   └── api.ts          # fetch wrappers for all backend endpoints
└── constants/
    └── nodeColors.ts   # Node-type → hex colour mapping for legend and canvas
```

## App.tsx — state and layout

`App` owns all top-level state and renders three regions:

- **Header** — brand badge + name/tagline.
- **Sidebar** — switches between Navigate and Analyze modes (see below).
- **Canvas area** — `GraphCanvas` with a loading overlay and a node/edge count badge.

### Navigate mode (default)

| Feature | What it does |
|---------|-------------|
| Node search | Filters nodes by label; clicking a result calls `setFocusedNodeId`, which triggers a client-side 1-hop subgraph focus (no network call). |
| Node type legend | `hiddenTypes` set controls which node types are rendered. Clicking toggles visibility. |
| Pathway picker | Calls `GET /api/graph?pathway=<name>` on each selection. Responses are cached in `pathwayContexts` so switching pathways in Analyze mode reuses already-fetched KG context. |
| Add Pathway | `ChatInput` field; submits to `POST /api/add_test`. On success, new nodes are highlighted (stored in `newNodeIds`) and the new test is auto-selected. |

### Analyze mode

`visitedPathways` tracks every pathway the user has viewed. Each pathway gets an accordion
entry with a `ChatBot` thread. Conversation history is stored per pathway in `pathwayHistories`.

## services/api.ts

All backend calls go through the `apiFetch` helper which throws `Error(detail)` on non-2xx
responses so callers can surface the server error message directly in the UI.

| Export | Endpoint |
|--------|----------|
| `getTests()` | `GET /api/tests` |
| `getGraph(pathway?)` | `GET /api/graph[?pathway=…]` |
| `addTest(diagnosticTest)` | `POST /api/add_test` |
| `sendChatMessage(message, history, context)` | `POST /api/chat` |

`API_BASE` is empty in development — Vite proxies `/api/*` to `localhost:8000` via `vite.config.ts`.
Set `VITE_API_URL` in `.env.production` for deployed environments.

## constants/nodeColors.ts

Maps each KG node type to a hex colour used in the legend and as the node fill colour in
`GraphCanvas`. Also exports `DEFAULT_NODE_COLOR` for unknown types.
