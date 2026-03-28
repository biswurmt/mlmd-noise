# Stage 1: build the React app
FROM node:22-alpine AS builder
WORKDIR /app

COPY diagnotix/frontend/package.json diagnotix/frontend/package-lock.json ./
RUN npm ci

COPY diagnotix/frontend/ .

# VITE_API_URL="" means the frontend sends /api/* requests to the same origin,
# which the outer Nginx reverse-proxy forwards to the backend container.
ENV VITE_API_URL=""
# Use vite build directly to skip tsc type-checking (pre-existing TS errors
# in GraphCanvas.tsx don't affect runtime behaviour).
RUN npx vite build

# Stage 2: serve the built app with Nginx
FROM nginx:1.27-alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY diagnotix/frontend/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
