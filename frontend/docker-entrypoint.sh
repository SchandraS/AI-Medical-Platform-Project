#!/bin/sh
# Renders the nginx proxy target from BACKEND_HOST/BACKEND_PORT before nginx
# starts (nginx itself cannot read environment variables directly). Defaults
# match docker-compose's service name/port, so a plain `docker build` +
# `docker run` with no env vars set still produces a config identical to the
# old static nginx.conf -- local docker-compose behavior is unchanged.
#
# Everything below echoes to stdout deliberately: this is the only way to
# confirm this script actually ran and what it did on a platform (e.g.
# Render's free tier) that doesn't offer shell access into the running
# container -- the Logs tab is the only available ground truth there.
set -e

echo "[docker-entrypoint] starting"

: "${BACKEND_SCHEME:=http}"
: "${BACKEND_HOST:=backend}"
: "${BACKEND_PORT:=8000}"
export BACKEND_SCHEME BACKEND_HOST BACKEND_PORT

echo "[docker-entrypoint] BACKEND_SCHEME=$BACKEND_SCHEME BACKEND_HOST=$BACKEND_HOST BACKEND_PORT=$BACKEND_PORT"

if ! command -v envsubst >/dev/null 2>&1; then
  echo "[docker-entrypoint] FATAL: envsubst not found on PATH" >&2
  exit 1
fi

if [ ! -f /etc/nginx/conf.d/default.conf.template ]; then
  echo "[docker-entrypoint] FATAL: template file missing at /etc/nginx/conf.d/default.conf.template" >&2
  exit 1
fi

envsubst '${BACKEND_SCHEME} ${BACKEND_HOST} ${BACKEND_PORT}' \
  < /etc/nginx/conf.d/default.conf.template \
  > /etc/nginx/conf.d/default.conf

echo "[docker-entrypoint] rendered config:"
cat /etc/nginx/conf.d/default.conf

echo "[docker-entrypoint] handing off to: $*"
exec "$@"
