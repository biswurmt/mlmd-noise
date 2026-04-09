# diagnotix/frontend

React 19 + TypeScript + Vite 6 web app.

```
frontend/
├── src/            # Application source (see src/README.md)
├── index.html
├── vite.config.ts  # Dev server proxy: /api/* → localhost:8000
├── tsconfig.json
└── package.json
```

## Development

```bash
npm install
npm run dev       # http://localhost:5173 (auto-increments if port is in use)
npm run build     # production build → dist/
npm run preview   # preview the production build locally
```

Vite proxies all `/api/*` requests to the FastAPI backend at `localhost:8000` during development,
so no CORS or URL configuration is needed. For deployed environments, set `VITE_API_URL` in
`.env.production`.
