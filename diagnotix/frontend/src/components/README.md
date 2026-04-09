# frontend/src/components

React components used by `App.tsx`.

## Files

### `GraphCanvas.tsx`

Interactive force-directed graph visualisation built on `react-force-graph-2d`.

Key props:
- `nodes` / `edges` — the currently visible subgraph (pre-filtered by `App` for hidden types and focus).
- `newNodeIds` — set of node IDs to highlight with an animation immediately after an add-test pipeline run.
- `activePathway` — label of the currently selected test node (highlighted differently in the canvas).
- `highlightedNodeId` — node hovered in the chat panel (Analyze mode); the canvas dims all other nodes.

---

### `ChatBot.tsx`

Per-pathway chat thread shown in Analyze mode. Renders the conversation history and sends
new messages to `POST /api/chat` via `sendChatMessage`. Calls `onHoverNode` when the user
hovers a node reference in the assistant response, allowing `App` to cross-highlight the
corresponding node in `GraphCanvas`.

Props: `context` (current KG subgraph + pathway), `messages`, `onMessagesChange`, `onHoverNode`.

---

### `ChatInput.tsx`

Single-line text input used in the "Add Pathway" section of the Navigate sidebar. Submits
on Enter or button click, and is disabled while an add-test pipeline is in progress.

Props: `onSubmit(diagnosticTest: string)`, `disabled`.

---

### `KGForm.tsx`

Diagnostic test input form with example chips (e.g. "CT Head", "MRI Brain"). Clicking a chip
pre-fills the input. Used in earlier versions of the app; `ChatInput` now handles the primary
add-pathway interaction in `App.tsx`.
