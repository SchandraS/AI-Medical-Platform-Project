# Multi-stage build: compile the Vite app, then serve the static bundle via nginx.
FROM node:20-slim AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf.template /etc/nginx/conf.d/default.conf.template
COPY docker-entrypoint.sh /docker-entrypoint.sh
# chmod explicitly here rather than relying on the git-tracked executable
# bit: NTFS (Windows) doesn't track it, so a commit made from a Windows
# checkout can silently lose it -- this was a real CI failure on the
# backend's entrypoint script earlier in this project, fixed the same way.
RUN chmod +x /docker-entrypoint.sh

EXPOSE 80

# Use 127.0.0.1, not "localhost": this image's /etc/hosts maps localhost to
# both 127.0.0.1 and ::1, nginx listens on 0.0.0.0:80 (IPv4 only), and wget
# resolving to the IPv6 entry first gets "connection refused" even though
# the server is healthy -- a false-unhealthy that doesn't affect real
# traffic (browsers/nginx-proxy still reach it via IPv4) but does fail the
# container healthcheck itself.
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=5 \
    CMD wget -qO- http://127.0.0.1:80/ || exit 1

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]
