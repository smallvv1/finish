# Toolbox Frontend (Standalone)

## Start

```bash
npm install
npm run dev
```

Default dev URL: `http://127.0.0.1:5173`

## Backend API

- Default in dev: proxied by Vite to `http://127.0.0.1:8000` for `/api` and `/runtime`
- For independent deployment, set `.env`:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## Build

```bash
npm run build
npm run preview
```
